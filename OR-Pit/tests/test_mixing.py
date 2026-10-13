"""Mixing pipeline tests on the SYNTHETIC dataset (generated into a temp dir)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orpit.curriculum import CurriculumManager, sample_gaps
from orpit.data import OnTheFlyDataset, RecipeDataset, collate_mixtures
from orpit.mixing import group_by_speaker, render_recipe, sample_recipe

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), "--out", str(out)], check=True
    )
    return out


def test_curriculum_stages():
    cur = CurriculumManager()
    rng = np.random.default_rng(0)
    for ep, ns, noise, rev in [(0, {2}, False, False), (15, {2, 3}, True, False), (35, {4, 5}, True, True)]:
        cfgs = [cur.get_config(ep, rng) for _ in range(50)]
        assert {c["num_speakers"] for c in cfgs} <= ns
        assert all(c["use_noise"] == noise and c["use_reverb"] == rev for c in cfgs)
        assert all(len(c["gaps"]) == c["num_speakers"] - 1 and min(c["gaps"] + [0]) >= 0 for c in cfgs)
    assert CurriculumManager(reverb_enabled=False).get_config(35, rng)["use_reverb"] is False


def test_gaps_shrink_with_overlap():
    rng = np.random.default_rng(0)
    lo = np.mean([sum(sample_gaps(rng, 3, 0.0)) for _ in range(300)])
    hi = np.mean([sum(sample_gaps(rng, 3, 1.0)) for _ in range(300)])
    assert lo > hi


def test_recipe_deterministic_and_rendering(synth):
    idx = json.load(open(synth / "index_train.json"))
    by = group_by_speaker(idx)
    mc = {"num_speakers": 3, "gaps": [0.1, 0.2], "use_noise": True, "snr_db": 3.0, "use_reverb": True}
    r1 = sample_recipe(np.random.default_rng(5), mc, by, None, 16000, min_active_sec=0.5)
    r2 = sample_recipe(np.random.default_rng(5), mc, by, None, 16000, min_active_sec=0.5)
    assert r1 == r2
    assert len({s["speaker"] for s in r1["sources"]}) == 3
    a, b = render_recipe(r1, str(synth)), render_recipe(r2, str(synth))
    assert np.array_equal(a["mixture"], b["mixture"])
    assert a["mixture"].shape == (16000,) and a["sources"].shape == (3, 16000)
    assert np.isfinite(a["mixture"]).all() and np.abs(a["mixture"]).max() <= 0.9 + 1e-6
    # mixture = reverberant sources + noise (reverberant target)
    assert np.allclose(a["mixture"], a["sources"].sum(0) + a["noise"], atol=1e-5)


def test_snr_is_achieved(synth):
    by = group_by_speaker(json.load(open(synth / "index_train.json")))
    for snr in (-5.0, 0.0, 5.0):
        mc = {"num_speakers": 2, "gaps": [0.3], "use_noise": True, "snr_db": snr, "use_reverb": False}
        out = render_recipe(
            sample_recipe(np.random.default_rng(1), mc, by, None, 16000, min_active_sec=0.5), str(synth)
        )
        got = 10 * np.log10(np.mean(out["sources"].sum(0) ** 2) / np.mean(out["noise"] ** 2))
        assert abs(got - snr) < 0.05


def test_on_the_fly_dataset_and_collate(synth):
    idx = json.load(open(synth / "index_train.json"))
    cur = CurriculumManager(
        [
            {
                "until_epoch": None,
                "num_speakers": [2, 3],
                "overlap": [0.5, 1.0],
                "noise": False,
                "reverb": False,
            }
        ]
    )
    ds = OnTheFlyDataset(idx, None, cur, str(synth), length=6, seg_len=16000, min_active_sec=0.5)
    ds.set_epoch(3)
    assert ds.make_recipe(2) == ds.make_recipe(2)  # deterministic per (seed, epoch, idx)
    ds.set_epoch(4)
    mix, src, n, rec = collate_mixtures([ds[i] for i in range(4)])
    assert mix.shape == (4, 1, 16000) and src.shape[0] == 4 and src.shape[1] == int(n.max())
    for b in range(4):
        assert (src[b, int(n[b]) :] == 0).all()


def test_static_recipes_have_disjoint_speakers(synth):
    tr = {e["speaker"] for e in json.load(open(synth / "index_train.json"))}
    for split in ("val", "test"):
        rec = json.load(open(synth / f"recipes_{split}.json"))
        spk = {s["speaker"] for r in rec for s in r["sources"]}
        assert not spk & tr
        ds = RecipeDataset(rec, str(synth))
        assert ds[0]["mixture"].shape[-1] == rec[0]["seg_len"]


def test_no_silent_targets_in_static_recipes(synth):
    # crops that land in a pause are moved deterministically to an active region
    for split in ("val", "test"):
        ds = RecipeDataset(json.load(open(synth / f"recipes_{split}.json")), str(synth))
        for i in range(len(ds)):
            assert (ds[i]["sources"].abs().amax(dim=1) > 1e-3).all()

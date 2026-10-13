"""Tests for ARCHITECTURE.md Revision 2 options and the Conv-TasNet baseline."""

import numpy as np
import soundfile as sf
import torch

from orpit.baselines.convtasnet import ConvTasNet
from orpit.data import FolderMixtureDataset
from orpit.loss import PITLoss
from orpit.metrics import evaluate_item
from orpit.model import V1, V2, SeparationNetwork, build_model


def test_v1_default_is_spec_and_state_dict_compatible():
    m = SeparationNetwork()
    sd = m.state_dict()
    m2 = build_model({})  # no model section → v1
    m2.load_state_dict(sd)  # strict load: names unchanged
    m3 = build_model({"model": {"preset": "v1"}})
    m3.load_state_dict(sd)


def test_v2_shapes_and_gradients():
    torch.manual_seed(0)
    m = build_model({"model": {"preset": "v2"}}).train()
    est, logits = m(torch.randn(2, 1, 16000))
    assert est.shape == (2, 6, 16000) and logits.shape == (2, 6, 1)
    PITLoss(match_cls_weight=2.0)(est, torch.randn(2, 2, 16000), logits).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_film_tcn_starts_at_v1_mask():
    torch.manual_seed(0)
    v1 = SeparationNetwork(**V1).eval()
    v2 = SeparationNetwork(**dict(V2, gated=False, mask_on="encoded")).eval()
    v2.load_state_dict(v1.state_dict(), strict=False)  # shared modules copied; TCN out is zero-init
    x = torch.randn(1, 1, 16000)
    with torch.no_grad():
        a, _ = v1(x)
        b, _ = v2(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_convtasnet_baseline():
    m = ConvTasNet(n_src=2)
    est, logits = m(torch.randn(2, 1, 16000))
    assert est.shape == (2, 2, 16000) and logits is None
    loss = PITLoss()(est, torch.randn(2, 2, 16000), None)
    assert torch.isfinite(loss)
    x, T = m.pad_input(torch.randn(1, 1, 12345))
    assert x.shape[-1] % 8 == 0 and T == 12345


def test_cls_matching_prefers_confident_slots():
    torch.manual_seed(0)
    ref = torch.randn(1, 1, 4000)
    est = ref.repeat(1, 6, 1) + 0.01 * torch.randn(1, 6, 4000)  # every slot equally good
    logits = torch.full((1, 6, 1), -5.0)
    logits[0, 4, 0] = 5.0
    crit = PITLoss(match_cls_weight=2.0)
    _, parts = crit(est, ref, logits, return_parts=True)
    assert parts["halting"] < 0.05  # slot 4 matched → targets agree with the head


def test_estcount_metric_and_folder_dataset(tmp_path):
    root = tmp_path / "lm"
    for d in ("mix_both", "s1", "s2"):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    s1, s2 = rng.standard_normal(9000) * 0.1, rng.standard_normal(9000) * 0.1
    sf.write(root / "s1" / "a.wav", s1, 16000)
    sf.write(root / "s2" / "a.wav", s2, 16000)
    sf.write(root / "mix_both" / "a.wav", s1 + s2, 16000)
    ds = FolderMixtureDataset(root)
    it = ds[0]
    assert it["sources"].shape == (2, 9000) and it["num_speakers"] == 2
    est = torch.stack([it["sources"][0], it["sources"][1]] + [it["mixture"]] * 4)
    logits = torch.tensor([5.0, 5.0, -5, -5, -5, -5])
    r = evaluate_item(est, it["sources"], it["mixture"], logits)
    assert r["count_correct"] == 1 and r["si_sdri_estcount"] > 30

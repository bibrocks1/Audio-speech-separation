"""Generate fixed-seed mixture recipes (and optionally the wav files) for each split.

    python scripts/make_mixtures.py --config configs/gtx1650.yaml --split val test
    python scripts/make_mixtures.py --config configs/gtx1650.yaml --split train --n 20000   # static train set
    python scripts/make_mixtures.py ... --materialize      # also write wavs under <audio_root>/mixtures/<split>/

Speech/noise sources per split (speaker-disjoint by construction):
  train → LibriSpeech train-clean-100 + WHAM! tr   (curriculum distribution: last stage)
  val   → LibriSpeech dev-clean       + WHAM! cv   (eval_mix distribution)
  test  → LibriSpeech test-clean      + WHAM! tt   (eval_mix distribution)
Val/test speaker counts are stratified evenly over eval_mix.num_speakers.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.audio_io import save_audio  # noqa: E402
from orpit.config import load_config, resolve  # noqa: E402
from orpit.curriculum import sample_mix_config  # noqa: E402
from orpit.mixing import group_by_speaker, load_rir_bank, render_recipe, sample_recipe  # noqa: E402

SPLITS = {
    "train": ("train.json", "noise_tr.json", "rir_bank_train"),
    "val": ("val.json", "noise_cv.json", "rir_bank_eval"),
    "test": ("test.json", "noise_tt.json", "rir_bank_eval"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gtx1650.yaml")
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--n", type=int, default=None, help="number of mixtures (default from eval_mix)")
    ap.add_argument("--index_dir", default=None, help="default <meta_dir>/indices")
    ap.add_argument("--out_dir", default=None, help="default <meta_dir>/recipes")
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    meta = resolve(cfg["paths"]["meta_dir"])
    index_dir = Path(args.index_dir) if args.index_dir else meta / "indices"
    out_dir = Path(args.out_dir) if args.out_dir else meta / "recipes"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root = resolve(cfg["paths"]["audio_root"])
    sr = cfg["audio"]["sample_rate"]
    seg_len = int(cfg["audio"]["segment_sec"] * sr)
    em = cfg["eval_mix"]

    for split in args.split:
        sp_file, nz_file, rir_key = SPLITS[split]
        speech = json.load(open(index_dir / sp_file))
        noise = json.load(open(index_dir / nz_file)) if (index_dir / nz_file).exists() else None
        rir_rel = cfg["data"].get(rir_key)
        rir_path = audio_root / rir_rel if rir_rel else None
        bank_size = int(load_rir_bank(str(rir_path)).shape[0]) if rir_path and rir_path.exists() else 0
        if split == "train":
            stage = dict(cfg["curriculum"]["stages"][-1])
            n = args.n or 20000
            seed = cfg["train"]["seed"] + 999
        else:
            stage = {
                "num_speakers": em["num_speakers"],
                "overlap": em["overlap"],
                "noise": em["noise"],
                "snr_db": em["snr_db"],
                "reverb": em["reverb"] and cfg["data"].get("reverb", True),
            }
            n = args.n or em[f"n_{split}"]
            seed = em[f"seed_{split}"]
        if stage.get("noise") and not noise:
            print(
                f"[{split}] WARNING: no noise index {nz_file}; using synthetic white noise (labelled in recipes)"
            )
        by_spk = group_by_speaker(speech)
        rng = np.random.default_rng(seed)
        recipes = []
        for i in range(n):
            st = dict(stage)
            if split != "train":
                st["num_speakers"] = [stage["num_speakers"][i % len(stage["num_speakers"])]]
            mc = sample_mix_config(rng, st)
            r = sample_recipe(
                rng,
                mc,
                by_spk,
                noise,
                seg_len,
                sr,
                rir_bank_size=bank_size,
                min_active_sec=cfg["audio"]["min_active_sec"],
            )
            r["id"] = f"{split}_{i:05d}"
            recipes.append(r)
        with open(out_dir / f"{split}.json", "w") as f:
            json.dump(recipes, f)
        counts = {
            k: sum(r["num_speakers"] == k for r in recipes)
            for k in sorted({r["num_speakers"] for r in recipes})
        }
        print(
            f"[{split}] {n} recipes (seed {seed}) → {out_dir / (split + '.json')}  speakers: {counts}  "
            f"noise={('WHAM' if noise else 'synthetic') if stage.get('noise') else 'off'} "
            f"reverb={('bank' if bank_size else 'synthetic') if stage.get('reverb') else 'off'}"
        )

        if args.materialize:
            mdir = audio_root / "mixtures" / split
            for r in recipes:
                out = render_recipe(
                    r, str(audio_root), str(rir_path) if bank_size else None, cfg["data"]["reverb_target"]
                )
                d = mdir / r["id"]
                save_audio(d / "mix.wav", out["mixture"], sr)
                for k, s in enumerate(out["sources"]):
                    save_audio(d / f"s{k + 1}.wav", s, sr)
            print(f"[{split}] wavs written to {mdir}")


if __name__ == "__main__":
    main()

"""Tiny SYNTHETIC dataset for smoke tests (NOT speech; labelled 'SYNTHETIC' everywhere).

Each synthetic "speaker" is a harmonic source with its own f0 (90–260 Hz), vibrato and
spectral tilt, gated by a syllable-rate (3–6 Hz) envelope with pauses. Train / val / test
use disjoint synthetic speakers.

    python scripts/make_synthetic.py [--out data/synthetic] [--seconds 3]
Writes <out>/{train,val,test}/*.wav, index_{train,val,test}.json, recipes_{val,test}.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.audio_io import save_audio  # noqa: E402
from orpit.config import resolve  # noqa: E402
from orpit.curriculum import sample_mix_config  # noqa: E402
from orpit.mixing import group_by_speaker, sample_recipe  # noqa: E402

SR = 16000


def synth_utterance(rng, f0, tilt, seconds):
    t = np.arange(int(seconds * SR)) / SR
    vib = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
    drift = 1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f0 * vib * drift) / SR
    x = sum((k**-tilt) * np.sin(k * phase + rng.uniform(0, 6.28)) for k in range(1, 25) if k * f0 < 7000)
    rate = rng.uniform(3, 6)
    env = np.clip(np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.28)), 0, None) ** 0.7
    pauses = np.repeat(rng.random(int(seconds * 2) + 1) > 0.25, SR // 2)[: len(t)]
    env = env * pauses
    x = x * env
    return (0.3 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment_sec", type=float, default=1.0)
    args = ap.parse_args()
    out = resolve(args.out)
    rng = np.random.default_rng(args.seed)
    layout = {"train": 8, "val": 4, "test": 4}
    spk_id = 0
    for split, n_spk in layout.items():
        index = []
        for _ in range(n_spk):
            spk_id += 1
            f0, tilt = rng.uniform(90, 260), rng.uniform(0.8, 1.6)
            for u in range(6):
                wav = synth_utterance(rng, f0, tilt, args.seconds)
                rel = f"{split}/SYN{spk_id:02d}-0-{u}.wav"
                save_audio(out / rel, wav, SR)
                index.append(
                    {
                        "path": rel,
                        "speaker": f"SYN{spk_id:02d}",
                        "sample_rate": SR,
                        "num_frames": len(wav),
                        "synthetic": True,
                    }
                )
        json.dump(index, open(out / f"index_{split}.json", "w"), indent=0)
        if split in ("val", "test"):
            by = group_by_speaker(index)
            recipes = []
            for i in range(8):
                stage = {"num_speakers": [2 + i % 2], "overlap": [0.6, 1.0], "noise": False, "reverb": False}
                r = sample_recipe(
                    rng,
                    sample_mix_config(rng, stage),
                    by,
                    None,
                    int(args.segment_sec * SR),
                    SR,
                    min_active_sec=0.5,
                )
                r["id"], r["synthetic"] = f"{split}_{i:03d}", True
                recipes.append(r)
            json.dump(recipes, open(out / f"recipes_{split}.json", "w"))
    (out / "README_SYNTHETIC.txt").write_text(
        "SYNTHETIC harmonic test signals for smoke tests only - not speech.\n"
    )
    print(f"[synthetic] wrote SYNTHETIC dataset to {out}")


if __name__ == "__main__":
    main()

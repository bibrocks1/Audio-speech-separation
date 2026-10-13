"""Input-level statistics: does mixture loudness give away the speaker count?

    python scripts/level_stats.py --config configs/v2/final_orpit_v2.yaml \
        --librimix data/audio/LibriMix [--n 400]
Prints RMS (dBFS) of own-test mixtures per speaker count and of LibriMix test mixtures.
CPU only; safe to run while a GPU job trains.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import load_config  # noqa: E402
from orpit.train import build_eval_dataset  # noqa: E402


def db(x):
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12)


def summary(name, vals):
    q = np.percentile(vals, [10, 50, 90])
    print(
        f"{name:28s} n={len(vals):4d}  RMS dBFS p10 {q[0]:6.1f}  median {q[1]:6.1f}  p90 {q[2]:6.1f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--librimix", default=None)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = build_eval_dataset(cfg, "test")
    by_n = {}
    for i in range(min(args.n, len(ds))):
        it = ds[i]
        by_n.setdefault(it["num_speakers"], []).append(db(it["mixture"].numpy()))
    for n in sorted(by_n):
        summary(f"own test {n} spk", by_n[n])
    if args.librimix:
        for sub in ("Libri2Mix", "Libri3Mix"):
            d = Path(args.librimix) / sub / "wav16k" / "min" / "test" / "mix_both"
            files = sorted(d.glob("*.wav"))[: args.n]
            if files:
                summary(f"{sub} test mix_both", [db(sf.read(str(f), dtype="float32")[0]) for f in files])


if __name__ == "__main__":
    main()

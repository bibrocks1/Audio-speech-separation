"""Rebuild the official Libri2Mix / Libri3Mix *test* sets (16 kHz, min mode) from the LibriMix
metadata, following JorisCos/LibriMix scripts/create_librimix_from_metadata.py (MIT):
sources × gain, WHAM! noise channel 0 × gain (looped with a 1 s Hann cross-fade if shorter),
all truncated to the shortest source, mix_clean = Σ sources, mix_both = mix_clean + noise.

    python scripts/get_librimix_metadata.py
    python scripts/download_wham_librimix.py --root data/audio
    python scripts/make_librimix_test.py --root data/audio [--n_src 2 3]
Output: <root>/LibriMix/Libri{N}Mix/wav16k/min/test/{s1..sN,noise,mix_clean,mix_both}/<id>.wav
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import resolve  # noqa: E402

RATE = 16000


def extend_noise(noise, max_length):
    """Same as LibriMix: concatenate with a 1 s Hann cross-fade until long enough."""
    noise_ex = noise
    window = np.hanning(RATE + 1)
    i_w, d_w = window[: len(window) // 2 + 1], window[len(window) // 2 :: -1]
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate(
            (
                noise_ex[: len(noise_ex) - len(d_w)],
                np.multiply(noise_ex[len(noise_ex) - len(d_w) :], d_w) + np.multiply(noise[: len(i_w)], i_w),
                noise[len(i_w) :],
            )
        )
    return noise_ex[:max_length]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n_src", nargs="+", type=int, default=[2, 3])
    args = ap.parse_args()
    root = Path(args.root)
    for n in args.n_src:
        meta = resolve(f"data/librimix/metadata/Libri{n}Mix/libri{n}mix_test-clean.csv")
        out = root / "LibriMix" / f"Libri{n}Mix" / "wav16k" / "min" / "test"
        dirs = [f"s{i + 1}" for i in range(n)] + ["noise", "mix_clean", "mix_both"]
        for d in dirs:
            (out / d).mkdir(parents=True, exist_ok=True)
        rows = list(csv.DictReader(open(meta)))
        for k, r in enumerate(rows):
            mid = r["mixture_ID"]
            if (out / "mix_both" / f"{mid}.wav").exists():
                continue
            srcs = [
                sf.read(str(root / "LibriSpeech" / r[f"source_{i + 1}_path"]), dtype="float32")[0]
                for i in range(n)
            ]
            max_len = max(len(s) for s in srcs)
            noise = sf.read(str(root / "wham_noise" / r["noise_path"]), dtype="float32", stop=max_len)[0]
            if noise.ndim > 1:
                noise = noise[:, 0]
            if len(noise) < max_len:
                noise = extend_noise(noise, max_len)
            sigs = [s * float(r[f"source_{i + 1}_gain"]) for i, s in enumerate(srcs)]
            sigs.append(noise * float(r["noise_gain"]))
            L = min(len(s) for s in sigs[:n])
            sigs = [s[:L] for s in sigs]
            for i in range(n):
                sf.write(str(out / f"s{i + 1}" / f"{mid}.wav"), sigs[i], RATE, subtype="FLOAT")
            sf.write(str(out / "noise" / f"{mid}.wav"), sigs[n], RATE, subtype="FLOAT")
            clean = np.sum(sigs[:n], axis=0)
            sf.write(str(out / "mix_clean" / f"{mid}.wav"), clean, RATE, subtype="FLOAT")
            sf.write(str(out / "mix_both" / f"{mid}.wav"), clean + sigs[n], RATE, subtype="FLOAT")
            if (k + 1) % 500 == 0:
                print(f"  Libri{n}Mix test {k + 1}/{len(rows)}", flush=True)
        print(f"[Libri{n}Mix] {len(rows)} test mixtures → {out}")


if __name__ == "__main__":
    main()

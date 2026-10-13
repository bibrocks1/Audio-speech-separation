"""Precompute a bank of simulated room impulse responses with pyroomacoustics.

Each room: random shoebox (x,y ∈ [3,8] m, z ∈ [2.5,3.5] m), RT60 ∈ [0.2, 0.6] s, one
microphone and 6 source positions (≥0.5 m from walls, 1–3 m from the mic). Precomputing
avoids running the image-source method inside the data loader.

    python scripts/make_rir_bank.py --out data/audio/rir_banks/train.npz --rooms 600 --seed 1
    python scripts/make_rir_bank.py --out data/audio/rir_banks/eval.npz  --rooms 150 --seed 2
"""

import argparse
import time
from pathlib import Path

import numpy as np


def random_room(rng, sr, max_len, max_order_cap=20):
    import pyroomacoustics as pra

    dim = np.array([rng.uniform(3, 8), rng.uniform(3, 8), rng.uniform(2.5, 3.5)])
    rt60 = rng.uniform(0.2, 0.6)
    absorption, max_order = pra.inverse_sabine(rt60, dim)
    room = pra.ShoeBox(
        dim, fs=sr, materials=pra.Material(absorption), max_order=min(max_order, max_order_cap)
    )
    mic = np.array([rng.uniform(0.5, dim[0] - 0.5), rng.uniform(0.5, dim[1] - 0.5), rng.uniform(1.0, 1.6)])
    room.add_microphone(mic)
    placed = 0
    while placed < 6:
        p = np.array([rng.uniform(0.5, dim[0] - 0.5), rng.uniform(0.5, dim[1] - 0.5), rng.uniform(1.2, 1.9)])
        if 1.0 <= np.linalg.norm(p - mic) <= 3.0:
            room.add_source(p)
            placed += 1
    room.compute_rir()
    rirs = np.zeros((6, max_len), dtype=np.float32)
    for k in range(6):
        r = np.asarray(room.rir[0][k], dtype=np.float32)
        r = r[:max_len]
        rirs[k, : len(r)] = r / (np.abs(r).max() + 1e-9)
    return rirs, rt60, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rooms", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max_sec", type=float, default=0.6)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    L = int(args.max_sec * args.sr)
    bank = np.zeros((args.rooms, 6, L), dtype=np.float16)
    rt60s, dims, t0 = [], [], time.time()
    for i in range(args.rooms):
        r, rt60, dim = random_room(rng, args.sr, L)
        bank[i] = r.astype(np.float16)
        rt60s.append(rt60)
        dims.append(dim)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.rooms} rooms ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, rirs=bank, rt60=np.array(rt60s), dims=np.array(dims))
    print(f"[rir] {args.rooms} rooms → {args.out} ({Path(args.out).stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()

"""Data-loader sanity check: shapes, sample rate, NaNs, clipping, speaker-disjoint splits,
distinct speakers within each mixture, achieved SNR.

    python scripts/sanity_check_data.py --config configs/gtx1650.yaml [--n 50]
Exit code 1 if any check fails.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import load_config, resolve  # noqa: E402
from orpit.train import build_eval_dataset, build_train_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gtx1650.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    sr = cfg["audio"]["sample_rate"]
    T = int(cfg["audio"]["segment_sec"] * sr)
    fails = []

    def check(cond, msg):
        print(("  PASS " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # 1) speaker-disjoint splits (indices, or speakers referenced by val/test recipes)
    spk = {}
    idx_dir = resolve(cfg["paths"]["meta_dir"]) / "indices"
    for split, path in (
        ("train", resolve(cfg["data"]["train_index"])),
        ("val", idx_dir / "val.json"),
        ("test", idx_dir / "test.json"),
    ):
        if path.exists():
            spk[split] = {e["speaker"] for e in json.load(open(path))}
    for split in ("val", "test"):
        rp = resolve(cfg["data"][f"{split}_recipes"])
        if rp.exists():
            spk.setdefault(split, set()).update(
                s["speaker"] for r in json.load(open(rp)) for s in r["sources"]
            )
    print("[splits] speakers:", {k: len(v) for k, v in spk.items()})
    names = list(spk)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            check(not (spk[names[i]] & spk[names[j]]), f"{names[i]} ∩ {names[j]} speakers empty")

    # 2) sample-rate of a few raw files
    root = Path(resolve(cfg["paths"]["audio_root"]))
    tr_idx = json.load(open(resolve(cfg["data"]["train_index"])))
    for e in tr_idx[:: max(1, len(tr_idx) // 5)][:5]:
        check(sf.info(str(root / e["path"])).samplerate == sr, f"{e['path']} is {sr} Hz")

    # 3) rendered items from train (every curriculum stage) and val
    train_ds = build_train_dataset(cfg)
    sets = []
    if hasattr(train_ds, "curriculum"):
        for st_i in range(len(train_ds.curriculum.stages)):
            ep = 0 if st_i == 0 else (train_ds.curriculum.stages[st_i - 1]["until_epoch"] or 0)
            sets.append((f"train/stage{st_i + 1}", train_ds, ep))
    else:
        sets.append(("train/static", train_ds, 0))
    sets.append(("val", build_eval_dataset(cfg, "val"), None))
    for name, ds, ep in sets:
        if ep is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(ep)
        bad_shape = nan = clip = dup = silent = 0
        counts, snr_err = [], []
        for i in range(min(args.n, len(ds))):
            it = ds[i]
            m, s, n, r = it["mixture"].numpy(), it["sources"].numpy(), it["num_speakers"], it["recipe"]
            bad_shape += int(m.shape != (T,) or s.shape != (n, T))
            nan += int(not (np.isfinite(m).all() and np.isfinite(s).all()))
            clip += int(np.abs(m).max() >= 1.0)
            dup += int(len({x["speaker"] for x in r["sources"]}) != n)
            silent += int((np.abs(s).max(axis=1) == 0).any())
            counts.append(n)
            if r.get("noise"):
                speech = s.sum(0) if not r.get("reverb") else None
                if speech is not None:
                    noise = m - speech
                    snr = 10 * np.log10(np.mean(speech**2) / max(1e-12, np.mean(noise**2)))
                    snr_err.append(abs(snr - r["noise"]["snr_db"]))
        k = min(args.n, len(ds))
        print(
            f"[{name}] {k} items, speaker counts {sorted(set(counts))}, "
            f"noise={'WHAM' if any(it['recipe'].get('noise', {}) and it['recipe']['noise'].get('path') for it in [ds[0]]) else 'none/synthetic'}"
        )
        check(bad_shape == 0, f"{name}: shapes mixture [{T}], sources [n, {T}]")
        check(nan == 0, f"{name}: no NaN/Inf")
        check(clip == 0, f"{name}: no clipping (|x| < 1)")
        check(dup == 0, f"{name}: speakers distinct within each mixture")
        check(silent == 0, f"{name}: no all-silent target")
        if snr_err:
            check(
                max(snr_err) < 0.1,
                f"{name}: achieved SNR within 0.1 dB of recipe (max err {max(snr_err):.3f})",
            )
    print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

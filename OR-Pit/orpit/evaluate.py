"""Evaluate a checkpoint on fixed test (or val) recipes, or on a LibriMix-style folder.

    python -m orpit.evaluate --checkpoint runs/gtx1650/best.pt [--split test] [--limit N] [--pesq] [--stoi]
    python -m orpit.evaluate --checkpoint ... --folder data/audio/LibriMix/Libri2Mix/wav16k/min/test \
                           --mix_dir mix_both --name libri2mix_noisy

Reports SI-SDR / SI-SDRi (Hungarian-matched over the 6 output slots, i.e. oracle
assignment) overall and per speaker count, plus halting-classifier count accuracy.
Writes <run_dir>/eval_<split>.csv (per mixture) and eval_<split>.json (summary).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from orpit.config import apply_env, load_config
from orpit.data import FolderMixtureDataset, collate_mixtures
from orpit.metrics import evaluate_item, summarise
from orpit.model import build_model
from orpit.train import build_eval_dataset, pick_device


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--threshold",
        default="0.5",
        help="halting threshold, or 'auto' = value chosen on validation (threshold.json)",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="(use with --split val) choose the halting threshold that maximises count "
        "accuracy on these mixtures and write <run>/threshold.json",
    )
    ap.add_argument("--pesq", action="store_true")
    ap.add_argument("--stoi", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--folder", default=None, help="LibriMix-style root with <mix_dir>/ and s1/, s2/, ...")
    ap.add_argument("--mix_dir", default="mix_both")
    ap.add_argument(
        "--chunk_sec",
        type=float,
        default=8.0,
        help="fallback chunk length if a full-length mixture does not fit in GPU memory",
    )
    ap.add_argument(
        "--input_rms_db",
        type=float,
        default=None,
        help="scale every input mixture to this RMS level (dBFS) before the model; "
        "training mixtures sit at about -20..-17 dBFS (SI-SDR metrics are scale-invariant)",
    )
    ap.add_argument("--name", default=None, help="output file stem (default eval_<split>)")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    out_dir = Path(args.checkpoint).parent
    if args.threshold == "auto":
        tf = out_dir / "threshold.json"
        args.threshold = json.load(open(tf))["threshold"] if tf.exists() else 0.5
        print(f"[threshold] {args.threshold} ({'from ' + str(tf) if tf.exists() else 'default'})")
    args.threshold = float(args.threshold)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.config:
        cfg = load_config(args.config, args.set)
    else:
        cfg = apply_env(ck["config"])
        if args.set:  # apply overrides on top of the stored config
            import tempfile

            import yaml

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f)
            cfg = load_config(f.name, args.set)

    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    if args.folder:
        ds = FolderMixtureDataset(args.folder, args.mix_dir, limit=args.limit)
    else:
        ds = build_eval_dataset(cfg, args.split)
        if getattr(model, "n_src", None):
            ds.recipes = [r for r in ds.recipes if r["num_speakers"] == model.n_src]
        if args.limit:
            ds.recipes = ds.recipes[: args.limit]
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_mixtures, num_workers=0)

    rows, t0, n_chunked = [], time.time(), 0
    with torch.no_grad():
        for i, (mix, refs, n_spk, _) in enumerate(loader):
            mix = mix.to(device)
            if args.input_rms_db is not None:
                mix = mix * (10 ** (args.input_rms_db / 20) / mix.pow(2).mean().sqrt().clamp_min(1e-8))
            x, T = model.pad_input(mix) if hasattr(model, "pad_input") else (mix, mix.shape[-1])
            try:
                est, logits = model(x)
                est = est[..., :T]
            except torch.cuda.OutOfMemoryError:
                # long mixture: chunked separation with Hungarian slot alignment (orpit.inference)
                from orpit.inference import separate_long

                del x
                torch.cuda.empty_cache()
                est, probs = separate_long(model, mix[0, 0], device, args.chunk_sec)
                est = est.unsqueeze(0).to(device)
                logits = (
                    torch.logit(probs.clamp(1e-6, 1 - 1e-6)).view(1, -1, 1).to(device)
                    if getattr(model, "has_halting", True)
                    else None
                )
                n_chunked += 1
            n = int(n_spk[0])
            r = evaluate_item(
                est[0],
                refs[0, :n].to(device),
                mix[0, 0],
                None if logits is None else logits[0, :, 0],
                args.threshold,
                pesq=args.pesq,
                stoi=args.stoi,
            )
            r["index"] = i
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(
                    f"  {i + 1}/{len(ds)}  running SI-SDRi {sum(x['si_sdri'] for x in rows) / len(rows):.2f} dB"
                )

    summary = summarise(rows)
    name = args.name or f"eval_{args.split}"
    if args.calibrate and rows and "probs" in rows[0]:
        sweep = {}
        for k in range(1, 100):
            th = k / 100
            sweep[th] = sum(int(sum(p > th for p in r["probs"]) == r["n_true"]) for r in rows) / len(rows)
        best = max(sweep.values())
        th = min((t for t, a in sweep.items() if a == best), key=lambda t: abs(t - 0.5))
        cal = {
            "threshold": th,
            "val_count_acc": best,
            "val_count_acc_at_0.5": sweep[0.5],
            "n_mixtures": len(rows),
            "data": args.folder or args.split,
            "sweep": {f"{t:.2f}": round(a, 4) for t, a in sweep.items()},
        }
        json.dump(cal, open(out_dir / "threshold.json", "w"), indent=1)
        print(f"[calibrate] threshold {th:.2f}: count acc {best:.3f} (0.5 → {sweep[0.5]:.3f})")
    summary["threshold"] = args.threshold
    summary["n_chunked_oom_fallback"] = n_chunked
    summary["input_rms_db"] = args.input_rms_db
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "epoch": ck.get("epoch"),
            "data": args.folder and f"{args.folder}/{args.mix_dir}" or args.split,
            "n_mixtures": len(rows),
            "seconds": round(time.time() - t0, 1),
            "matching": "oracle Hungarian over all output slots",
            "model": (cfg.get("model") or {"preset": "v1"}),
        }
    )
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    keys = [k for k in rows[0] if not isinstance(rows[0][k], (list, dict))]
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Audio demos for the results page — *representative*, not cherry-picked: for each test set the
mixture whose SI-SDRi is closest to that set's median (from the evaluation CSV) is used.

    python scripts/make_demos.py --orpit runs/final_orpit_v2/best.pt \
        --baseline runs/final_convtasnet/best.pt --librimix data/audio/LibriMix --out results/demos

Writes <out>/<demo>/{mixture,ref_k,orpit_k,convtasnet_k}.wav (16-bit, peak-normalised to 0.9;
model outputs are shown in reference order via Hungarian matching) and <out>/demos.json with
per-source SI-SDRi and the halting probabilities of every OR-PiT slot.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import apply_env, resolve  # noqa: E402
from orpit.data import FolderMixtureDataset  # noqa: E402
from orpit.loss import hungarian_match, pairwise_si_sdr, si_sdr  # noqa: E402
from orpit.model import build_model  # noqa: E402
from orpit.train import build_eval_dataset  # noqa: E402


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = apply_env(ck["config"])
    m = build_model(cfg).to(device).eval()
    m.load_state_dict(ck["model_state_dict"])
    tf = Path(ckpt).parent / "threshold.json"
    th = json.load(open(tf))["threshold"] if tf.exists() else 0.5
    return m, cfg, th


def median_index(csv_path, n_true=None):
    rows = [r for r in csv.DictReader(open(csv_path)) if n_true is None or int(r["n_true"]) == n_true]
    if not rows:
        return None, None
    med = statistics.median(float(r["si_sdri"]) for r in rows)
    r = min(rows, key=lambda r: abs(float(r["si_sdri"]) - med))
    return int(r["index"]), med


def norm(y):
    p = float(np.abs(y).max())
    return (y / p * 0.9 if p > 0 else y).astype(np.float32)


@torch.no_grad()
def run(model, mix, device):
    x, T = model.pad_input(mix.view(1, 1, -1).to(device))
    est, logits = model(x)
    probs = None if logits is None else torch.sigmoid(logits[0, :, 0]).cpu().tolist()
    return est[0, :, :T].float().cpu(), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orpit", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--librimix", default=None)
    ap.add_argument("--out", default="results/demos")
    ap.add_argument(
        "--input_rms_db",
        type=float,
        default=None,
        help="scale LibriMix inputs to this RMS level before the models (as in eval *_norm)",
    )
    ap.add_argument("--eval_suffix", default="", help="e.g. _norm: pick medians from eval_<set>_norm.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    orpit, cfg, th = load(args.orpit, dev)
    base = load(args.baseline, dev)[0] if args.baseline and Path(args.baseline).exists() else None
    run_dir = Path(args.orpit).parent
    own = build_eval_dataset(cfg, "test")
    demos = []
    for n in (2, 3, 4, 5):
        if (run_dir / "eval_test.csv").exists():
            i, med = median_index(run_dir / "eval_test.csv", n)
            if i is not None:
                demos.append(
                    (f"own_test_{n}spk", own, i, med, f"Own test set (test-clean + WHAM!), {n} speakers")
                )
    if args.librimix:
        for stem, sub, desc in (
            ("eval_libri2mix_noisy", "Libri2Mix", "Libri2Mix test (noisy, min)"),
            ("eval_libri3mix_noisy", "Libri3Mix", "Libri3Mix test (noisy, min)"),
        ):
            if (run_dir / f"{stem}{args.eval_suffix}.csv").exists():
                i, med = median_index(run_dir / f"{stem}{args.eval_suffix}.csv")
                ds = FolderMixtureDataset(Path(args.librimix) / sub / "wav16k" / "min" / "test", "mix_both")
                demos.append((stem.replace("eval_", ""), ds, i, med, desc))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, ds, i, med, desc in demos:
        item = ds[i]
        mix, refs = item["mixture"], item["sources"]
        if args.input_rms_db is not None and name.startswith("libri"):
            mix = mix * (10 ** (args.input_rms_db / 20) / mix.pow(2).mean().sqrt().clamp_min(1e-8))
        n = refs.shape[0]
        d = out / name
        d.mkdir(exist_ok=True)
        sf.write(str(d / "mixture.wav"), norm(mix.numpy()), 16000, subtype="PCM_16")
        base_sdr = si_sdr(mix.unsqueeze(0).expand_as(refs), refs)
        entry = {
            "name": name,
            "description": desc,
            "index": i,
            "id": item["recipe"].get("id"),
            "set_median_si_sdri": round(med, 2),
            "n_speakers": n,
            "seconds": round(mix.numel() / 16000, 2),
            "threshold": th,
            "files": {"mixture": f"{name}/mixture.wav"},
        }
        for k in range(n):
            sf.write(str(d / f"ref_{k + 1}.wav"), norm(refs[k].numpy()), 16000, subtype="PCM_16")
        for tag, model in (("orpit", orpit), ("convtasnet", base)):
            if model is None or (getattr(model, "n_src", None) and model.n_src != n):
                continue
            est, probs = run(model, mix, dev)
            sdr = pairwise_si_sdr(est, refs)
            rows, cols = hungarian_match(sdr)
            slot_for_ref = {int(c): int(r) for r, c in zip(rows, cols)}
            res = []
            for k in range(n):
                s = slot_for_ref[k]
                sf.write(str(d / f"{tag}_{k + 1}.wav"), norm(est[s].numpy()), 16000, subtype="PCM_16")
                res.append(
                    {
                        "slot": s,
                        "si_sdri": round(float(sdr[s, k] - base_sdr[k]), 2),
                        "prob": None if probs is None else round(probs[s], 3),
                    }
                )
            entry[tag] = {
                "per_source": res,
                "mean_si_sdri": round(float(np.mean([r["si_sdri"] for r in res])), 2),
            }
            if probs is not None:
                entry[tag]["slot_probs"] = [round(p, 3) for p in probs]
                entry[tag]["predicted_count"] = int(sum(p > th for p in probs))
        meta.append(entry)
        print(
            json.dumps({k: entry[k] for k in ("name", "n_speakers", "set_median_si_sdri")}),
            {t: entry[t]["mean_si_sdri"] for t in ("orpit", "convtasnet") if t in entry},
        )
    json.dump(meta, open(out / "demos.json", "w"), indent=1)
    print(f"→ {out}")


if __name__ == "__main__":
    main()

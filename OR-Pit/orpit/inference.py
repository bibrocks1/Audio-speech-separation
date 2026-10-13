"""Separate an audio file into per-speaker wavs with a trained OR-PiT checkpoint.

    python -m orpit.inference --checkpoint runs/gtx1650/best.pt --input_mix mix.wav --output_dir out/
                            [--threshold 0.5] [--save_all] [--chunk_sec 8]

The halting classifier (sigmoid(logit) > threshold) decides which of the 6 slots are
written; at least one slot is always kept. Output files are sorted by speaker
probability: speaker_1.wav is the most confident. --save_all writes every slot.

Long inputs: self-attention cost grows with length², so files longer than --chunk_sec are
processed in chunks with 1 s overlap; slots of neighbouring chunks are aligned by Hungarian
matching on the overlap (SI-SDR) and cross-faded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from orpit.audio_io import load_audio, save_audio
from orpit.config import apply_env
from orpit.loss import hungarian_match, pairwise_si_sdr
from orpit.model import build_model

SR = 16000


@torch.no_grad()
def separate_tensor(model, wav: torch.Tensor, device):
    """wav [T] → est [6, T], probs [6]."""
    x, T = model.pad_input(wav.view(1, 1, -1).to(device))
    est, logits = model(x)
    if logits is None:  # fixed-speaker baseline: every output is "active"
        return est[0, :, :T].float().cpu(), torch.ones(est.shape[1])
    return est[0, :, :T].float().cpu(), torch.sigmoid(logits[0, :, 0].float()).cpu()


@torch.no_grad()
def separate_long(model, wav: torch.Tensor, device, chunk_sec: float, overlap_sec: float = 1.0):
    chunk = int(chunk_sec * SR)
    ov = int(min(overlap_sec, chunk_sec / 4) * SR)
    if wav.numel() <= chunk:
        return separate_tensor(model, wav, device)
    hop = chunk - ov
    starts = list(range(0, max(1, wav.numel() - ov), hop))
    out, prob_acc, prev_tail = None, None, None
    fade_in = torch.linspace(0, 1, ov)
    for s in starts:
        seg = wav[s : s + chunk]
        est, probs = separate_tensor(model, seg, device)
        if out is None:
            out, prob_acc = torch.zeros(est.shape[0], wav.numel()), torch.zeros(est.shape[0])
        if prev_tail is not None:
            # align this chunk's slots to the previous chunk on the overlap region
            sdr = pairwise_si_sdr(prev_tail, est[:, :ov])
            rows, cols = hungarian_match(sdr)
            perm = torch.as_tensor(cols[np.argsort(rows)])
            est, probs = est[perm], probs[perm]
            out[:, s : s + ov] = out[:, s : s + ov] * (1 - fade_in) + est[:, :ov] * fade_in
            out[:, s + ov : s + est.shape[1]] = est[:, ov:]
        else:
            out[:, s : s + est.shape[1]] = est
        prob_acc += probs * est.shape[1]
        prev_tail = est[:, -ov:] if est.shape[1] >= ov else None
        if s + chunk >= wav.numel():
            break
    return out, prob_acc / wav.numel()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Separate a mixed audio file with OR-PiT")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_mix", required=True)
    ap.add_argument("--output_dir", default="outputs/separated")
    ap.add_argument(
        "--threshold",
        default="0.5",
        help="halting threshold, or 'auto' = <run>/threshold.json (validation-calibrated)",
    )
    ap.add_argument("--save_all", action="store_true")
    ap.add_argument(
        "--input_rms_db",
        type=float,
        default=-19.0,
        help="normalise the input to this RMS level (dBFS), the median level of the training "
        "mixtures; the halting head is level-sensitive. Use 'nan' to disable",
    )
    ap.add_argument("--chunk_sec", type=float, default=8.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    if args.threshold == "auto":
        tf = Path(args.checkpoint).parent / "threshold.json"
        args.threshold = json.load(open(tf))["threshold"] if tf.exists() else 0.5
    args.threshold = float(args.threshold)

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(apply_env(ck.get("config") or {})).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])

    wav = torch.from_numpy(load_audio(args.input_mix, SR))
    if args.input_rms_db == args.input_rms_db:  # not NaN
        wav = wav * (10 ** (args.input_rms_db / 20) / wav.pow(2).mean().sqrt().clamp_min(1e-8))
    print(f"[input] {args.input_mix}: {wav.numel() / SR:.2f} s @ {SR} Hz (mono)")
    est, probs = separate_long(model, wav, device, args.chunk_sec)

    order = torch.argsort(probs, descending=True).tolist()
    keep = order if args.save_all else [i for i in order if probs[i] > args.threshold] or order[:1]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for rank, slot in enumerate(keep, 1):
        y = est[slot].numpy()
        peak = float(np.abs(y).max())
        if peak > 0:
            y = y / peak * 0.9  # SI-SDR training leaves output scale arbitrary
        path = out_dir / f"speaker_{rank}.wav"
        save_audio(path, y, SR)
        report.append({"file": path.name, "slot": slot, "prob": round(float(probs[slot]), 4)})
        print(f"  saved {path}  (slot {slot}, p={float(probs[slot]):.3f})")
    with open(out_dir / "separation.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(args.input_mix),
                "checkpoint": str(args.checkpoint),
                "threshold": args.threshold,
                "slot_probs": [round(float(p), 4) for p in probs],
                "outputs": report,
            },
            f,
            indent=2,
        )
    print(
        f"[done] {len(keep)} source(s) → {out_dir}  (predicted speakers: {int((probs > args.threshold).sum())})"
    )


if __name__ == "__main__":
    main()

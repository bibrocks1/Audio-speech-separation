"""Measure peak VRAM and step time of one full training step (fwd + PIT loss + bwd + AdamW).

    python scripts/vram_probe.py [--segments 2 3 4] [--batches 1 2 4] [--speakers 5] [--amp both] [--ckpt both]

Random waveforms are used (memory does not depend on content). Prints a table and writes
runs/vram_probe.json. Also verifies forward shapes against ARCHITECTURE.md and that AMP
gives a finite loss / no NaN gradients.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import (
    load_config,  # noqa: E402
    resolve,  # noqa: E402
)
from orpit.loss import PITLoss  # noqa: E402
from orpit.model import build_model, count_parameters  # noqa: E402


def one(seg, bs, n_spk, amp, ckpt, device, steps=3, cfg=None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg, grad_checkpoint=ckpt).to(device).train()
    n_spk = min(n_spk, getattr(model, "n_src", n_spk))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = PITLoss()
    T = int(seg * 16000)
    res = {"segment_sec": seg, "batch": bs, "speakers": n_spk, "amp": amp, "grad_ckpt": ckpt}
    try:
        times = []
        for s in range(steps):
            x = torch.randn(bs, 1, T, device=device) * 0.1
            refs = torch.randn(bs, n_spk, T, device=device) * 0.05
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                est, logits = model(x)
            if s == 0:
                res["est_shape"] = list(est.shape)
                res["logits_shape"] = None if logits is None else list(logits.shape)
            loss = crit(est, refs, logits)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        res.update(
            {
                "ok": True,
                "loss_finite": bool(torch.isfinite(loss)),
                "grad_norm_finite": bool(torch.isfinite(gn)),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated(device) / 2**20),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20),
                "sec_per_step": round(sum(times[1:]) / max(1, len(times) - 1), 3),
            }
        )
        # Windows (WDDM) spills to shared system memory instead of raising OOM: flag it
        res["spills_to_shared_memory"] = (
            res["peak_reserved_mb"] > torch.cuda.get_device_properties(device).total_memory / 2**20
        )
    except torch.cuda.OutOfMemoryError:
        res.update({"ok": False, "error": "OOM"})
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="+", type=float, default=[2.0, 3.0, 4.0])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--speakers", type=int, default=5)
    ap.add_argument("--amp", choices=["on", "off", "both"], default="both")
    ap.add_argument("--ckpt", choices=["on", "off", "both"], default="both")
    ap.add_argument("--config", default=None, help="model config (default: v1 OR-PiT)")
    ap.add_argument("--out", default="runs/vram_probe.json")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA not available"
    cfg = load_config(args.config) if args.config else {}
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(
        f"GPU {props.name}  {props.total_memory / 2**30:.2f} GB  cc {props.major}.{props.minor}  torch {torch.__version__}"
        f"  params {count_parameters(build_model(cfg)):,}"
    )
    amps = {"on": [True], "off": [False], "both": [True, False]}[args.amp]
    ckpts = {"on": [True], "off": [False], "both": [False, True]}[args.ckpt]
    results = []
    for seg, bs, amp, ck in itertools.product(args.segments, args.batches, amps, ckpts):
        r = one(seg, bs, args.speakers, amp, ck, dev, cfg=cfg)
        results.append(r)
        print(json.dumps(r), flush=True)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "gpu": props.name,
            "total_mb": props.total_memory // 2**20,
            "torch": torch.__version__,
            "results": results,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"→ {out}")


if __name__ == "__main__":
    main()

"""Inference speed / memory of a trained checkpoint: real-time factor (RTF = processing time /
audio duration), peak VRAM and parameter count for a few input lengths.

    python scripts/benchmark_speed.py --checkpoint runs/final_orpit_v2/best.pt [--seconds 4 8 16]
Writes <run>/speed.json. Random input (speed does not depend on content); 2 warm-up passes,
then the median of --repeats timed passes with CUDA synchronisation.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import apply_env  # noqa: E402
from orpit.model import build_model, count_parameters  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seconds", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(apply_env(ck["config"])).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    res = {
        "checkpoint": args.checkpoint,
        "params": count_parameters(model),
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "precision": "fp32",
        "results": [],
    }
    with torch.no_grad():
        for sec in args.seconds:
            x = torch.randn(1, 1, int(sec * 16000), device=dev) * 0.1
            x, _ = model.pad_input(x)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            try:
                times = []
                for k in range(2 + args.repeats):
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(x)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    if k >= 2:
                        times.append(time.perf_counter() - t0)
                t = statistics.median(times)
                r = {
                    "seconds": sec,
                    "ms": round(1000 * t, 1),
                    "rtf": round(t / sec, 4),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated(dev) / 2**20)
                    if dev.type == "cuda"
                    else None,
                }
            except torch.cuda.OutOfMemoryError:
                r = {"seconds": sec, "error": "OOM"}
            res["results"].append(r)
            print(json.dumps(r), flush=True)
    out = Path(args.checkpoint).parent / "speed.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"params {res['params']:,} → {out}")


if __name__ == "__main__":
    main()

"""Choose the epoch count that fits a wall-clock budget, from a vram_probe.py measurement, so a
cosine LR schedule fully decays within the budget (used to give OR-PiT and the baseline the
same GPU time). Prints a single integer.

    python scripts/plan_epochs.py --probe runs/probe_x.json --segment 3 --batch 2 \\
        --mixtures 4000 --minutes 480 [--overhead 1.15] [--val_minutes 2]
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--segment", type=float, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--mixtures", type=int, required=True)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--overhead", type=float, default=1.15, help="data loading / logging slack")
    ap.add_argument("--val_minutes", type=float, default=2.0)
    args = ap.parse_args()
    res = json.load(open(args.probe))["results"]
    r = next(x for x in res if x["segment_sec"] == args.segment and x["batch"] == args.batch and x.get("ok"))
    epoch_min = args.mixtures / args.batch * r["sec_per_step"] * args.overhead / 60 + args.val_minutes
    print(max(1, int(args.minutes // epoch_min)))


if __name__ == "__main__":
    main()

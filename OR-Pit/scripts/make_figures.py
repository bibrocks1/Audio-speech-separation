"""Render the README figures from the evaluation results.

    python scripts/make_figures.py [--results results/results.json] [--runs runs] [--out results/figures]

Reads results/results.json (written by scripts/make_results.py) and, for the training curves,
runs/<run>/metrics.csv. Needs matplotlib (pip install matplotlib).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import resolve  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1"
INK, MUTED, GRID = "#1b1b1a", "#6f6e69", "#e6e5df"


def style(ax, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def overall(e):
    return (e or {}).get("overall", {}).get("si_sdri")


def ablation(R, out):
    rows = [
        ("A", "v1 (spec)"),
        ("B", "+ frontend\nmasks"),
        ("C", "+ gated cgMLP,\nhalting matching"),
        ("D", "+ FiLM-TCN\n(= v2)"),
    ]
    vals = [overall(R.get(f"ablation_{t}")) for t, _ in rows]
    if None in vals:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=160)
    bars = ax.bar(
        [f"{t}\n{n}" for t, n in rows],
        vals,
        width=0.55,
        color=[GREY, GREY, GREY, BLUE],
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"+{v:.2f} dB", ha="center", fontsize=9, color=INK)
    style(ax, "SI-SDRi (dB)")
    ax.set_ylim(0, max(vals) + 0.7)
    ax.set_title(
        "Cumulative ablation, equal data and GPU time (own test, 2–5 speakers)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out / "ablation.png")
    plt.close(fig)


def per_count(R, out):
    v1, v2 = R.get("gtx1650_budget", {}).get("own_test"), R.get("final_orpit_v2", {}).get("own_test")
    ctn = R.get("final_convtasnet", {}).get("own_test")
    if not v2:
        return
    ns = [2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=160)
    w = 0.26
    series = [("OR-PiT v1 (spec)", v1, ORANGE, -w), ("OR-PiT v2", v2, BLUE, 0.0)]
    for name, e, c, dx in series:
        if e:
            ys = [e.get(f"{n}spk", {}).get("si_sdri") for n in ns]
            ax.bar([i + dx for i in range(4)], ys, width=w, color=c, label=name)
    if ctn and ctn.get("2spk"):
        y = ctn["2spk"]["si_sdri"]
        ax.bar([w], [y], width=w, color=AQUA, label="Conv-TasNet (2-speaker only)")
    ax.set_xticks(range(4), [f"{n} speakers" for n in ns])
    style(ax, "SI-SDRi (dB)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    ax.set_title("Own test set by speaker count", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "per_speaker_count.png")
    plt.close(fig)


def curves(runs, out):
    def load(run):
        p = runs / run / "metrics.csv"
        if not p.exists():
            return None
        rows = list(csv.DictReader(open(p)))
        return [float(r["elapsed_min"]) / 60 for r in rows], [float(r["val_si_sdri"]) for r in rows]

    series = [("OR-PiT v2", "final_orpit_v2", BLUE), ("OR-PiT v1 (spec)", "gtx1650_budget", ORANGE)]
    data = [(n, load(r), c) for n, r, c in series]
    if not any(d for _, d, _ in data):
        return
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=160)
    for name, d, c in data:
        if d:
            ax.plot(d[0], d[1], color=c, linewidth=2, marker="o", markersize=3.5, label=name)
            ax.annotate(
                f"{d[1][-1]:+.2f} dB",
                (d[0][-1], d[1][-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                color=INK,
                va="center",
            )
    style(ax, "val SI-SDRi (dB)")
    ax.set_xlabel("training time on a GTX 1650 (hours)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    ax.set_title(
        "Validation SI-SDRi (400 fixed dev-clean mixtures, 2–5 speakers)", fontsize=10, color=INK, loc="left"
    )
    fig.tight_layout()
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    R = json.load(open(resolve(args.results)))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ablation(R, out)
    per_count(R, out)
    curves(resolve(args.runs), out)
    print("→", out, sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()

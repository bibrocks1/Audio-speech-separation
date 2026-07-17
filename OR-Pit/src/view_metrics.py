"""
view_metrics.py
───────────────
Inspect training and validation metrics stored inside checkpoint .pt files
without loading the model or initialising CUDA.  Safe to run on any machine,
including a CPU-only laptop.

Usage:
    python -m src.view_metrics                          # scans checkpoints/
    python -m src.view_metrics --checkpoint_dir /path/to/checkpoints
    python -m src.view_metrics --sort val               # sort by val loss
    python -m src.view_metrics --sort train             # sort by train loss
"""

import os
import glob
import argparse

import torch


def load_metrics(checkpoint_dir: str) -> list[dict]:
    """
    Scan checkpoint_dir for *.pt files, load each on CPU, and extract the
    metrics dict.  Returns a list of dicts sorted by the 'epoch' key.
    """
    pattern = os.path.join(checkpoint_dir, "*.pt")
    paths   = glob.glob(pattern)

    if not paths:
        return []

    rows = []
    for path in paths:
        try:
            # map_location='cpu' — no GPU required.
            # weights_only=False — we need the full dict, not just tensors.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  [WARN] Could not load {os.path.basename(path)}: {exc}")
            continue

        rows.append(
            {
                "file":        os.path.basename(path),
                "epoch":       ckpt.get("epoch",       None),
                "train_loss":  ckpt.get("epoch_loss",  None),   # key used in train.py
                "val_loss":    ckpt.get("val_loss",     None),
            }
        )

    return rows


def _fmt(value, decimals: int = 4) -> str:
    """Format a numeric value or return 'N/A' if missing."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _sort_key(row: dict, sort_by: str):
    """Return a sortable key; pushes None values to the end."""
    val = row.get(sort_by)
    if val is None:
        return float("inf")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("inf")


def print_table(rows: list[dict], sort_by: str = "epoch") -> None:
    """Print a fixed-width ASCII table of checkpoint metrics."""
    if not rows:
        print("  (no checkpoints found)")
        return

    rows_sorted = sorted(rows, key=lambda r: _sort_key(r, sort_by))

    # ── Column widths ─────────────────────────────────────────────────────────
    col_file  = max(len(r["file"]) for r in rows_sorted)
    col_file  = max(col_file, len("File"))
    col_epoch = 7    # "Epoch  "
    col_train = 12   # "Train Loss  "
    col_val   = 12   # "Val Loss    "

    header = (
        f"{'File':<{col_file}}  "
        f"{'Epoch':>{col_epoch}}  "
        f"{'Train Loss':>{col_train}}  "
        f"{'Val Loss':>{col_val}}"
    )
    separator = "─" * len(header)

    print()
    print(separator)
    print(header)
    print(separator)

    for r in rows_sorted:
        epoch_str = str(r["epoch"]) if r["epoch"] is not None else "N/A"
        print(
            f"{r['file']:<{col_file}}  "
            f"{epoch_str:>{col_epoch}}  "
            f"{_fmt(r['train_loss']):>{col_train}}  "
            f"{_fmt(r['val_loss']):>{col_val}}"
        )

    print(separator)
    print(f"  {len(rows_sorted)} checkpoint(s) found.")

    # ── Best epoch callout ────────────────────────────────────────────────────
    val_rows = [r for r in rows_sorted if r["val_loss"] is not None]
    if val_rows:
        best = min(val_rows, key=lambda r: float(r["val_loss"]))
        print(
            f"\n  ★  Best val loss: {_fmt(best['val_loss'])} "
            f"at epoch {best['epoch']}  ({best['file']})"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="View training/validation metrics from checkpoint .pt files."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory to scan for *.pt files (default: checkpoints/).",
    )
    parser.add_argument(
        "--sort",
        choices=["epoch", "train_loss", "val_loss"],
        default="epoch",
        help="Column to sort results by (default: epoch).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        print(f"[ERROR] Directory not found: {args.checkpoint_dir}")
        return

    print(f"Scanning: {os.path.abspath(args.checkpoint_dir)}")
    rows = load_metrics(args.checkpoint_dir)
    print_table(rows, sort_by=args.sort)


if __name__ == "__main__":
    main()

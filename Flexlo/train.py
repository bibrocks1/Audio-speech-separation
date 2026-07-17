"""
Full start-to-end training loop for the FlexIO-Lite recursive OR-PIT
separator, built on top of your existing scripts/dataset.py and
scripts/dynamic_mixer.py.

Usage:
    python scripts/train.py --epochs 40 --batch_size 4 --lr 1e-3

What this does, in order:
  1. Loads your DynamicMixtureDataset (curriculum-driven mixing you already
     wrote), which yields a mixture + variable-length list of clean sources
     + per-recursion-depth stopping labels for each example.
  2. Each batch: runs the OR-PIT recursive loop -- repeatedly extract one
     speaker + one residual, match against remaining ground-truth sources,
     backprop the SI-SNR loss + the stopping-classifier BCE loss.
  3. Periodically validates on the static eval sets you built with
     build_eval_sets.py, reporting SI-SNRi *per speaker-count level*
     (exactly how your project will be graded) and stopping-classifier
     precision/recall.
  4. Checkpoints the model, logs metrics to a CSV, and keeps the best
     checkpoint by mean validation SI-SNRi.

IMPORTANT: this script must live in your project's `scripts/` folder
(alongside dataset.py, dynamic_mixer.py, etc) since it imports from them
using the same sys.path pattern your other scripts already use.
"""

import os
import sys
import csv
import json
import glob
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf


def load_wav_simple(path, target_sr=16000):
    """Load a wav file as a [1, L] float32 tensor at target_sr. Uses
    soundfile directly (not torchaudio.load) since some torchaudio versions
    require an extra torchcodec dependency that may not be installed --
    same reasoning as the load_wav() helper in dynamic_mixer.py."""
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.transpose(0, 1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav

# --- make workspace root importable, same pattern as your other scripts ---
if '__file__' in globals():
    WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    WORKSPACE_ROOT = os.getcwd()
sys.path.append(WORKSPACE_ROOT)

from scripts.dataset import get_dataloader
from model import (
    FlexIOLiteRecursiveSeparator,
    si_snr,
    or_pit_step_loss,
    compute_si_snri,
    StoppingStats,
    summarize_si_snri_list,
)


# --------------------------------------------------------------------------
# Training-step logic (one batch through the recursive OR-PIT loop)
# --------------------------------------------------------------------------

def run_training_step(model, mixtures, sources, recursion_labels, device,
                       stop_loss_weight=0.5, optimizer=None):
    """
    mixtures: [B, 1, L] tensor
    sources: list of length B, each a list of [1, L] tensors (variable length per item)
    recursion_labels: list of length B, each a list[bool] of length == n_speakers for that item
    optimizer: if provided, this function calls loss.backward() + optimizer.step()
        *per recursion step* (rather than accumulating every step's graph and
        doing one big backward at the end). This keeps peak memory roughly
        constant regardless of how deep the recursion goes (2 vs 6 speakers),
        which matters a lot once you scale to higher speaker counts. If None,
        the caller is responsible for backward/step.

    Returns: dict of scalar metrics for logging (averaged across all steps).
    """
    B = mixtures.shape[0]
    mixtures = mixtures.to(device)

    remaining = [[s.to(device) for s in sources[b]] for b in range(B)]
    n_speakers = [len(r) for r in remaining]
    max_depth = max(n_speakers)

    current_mix = mixtures.clone()

    step_sep_losses = []
    step_stop_losses = []
    step_si_snris = []
    stop_stats = StoppingStats()

    for k in range(max_depth):
        active_idx = [b for b in range(B) if k < n_speakers[b]]
        if not active_idx:
            break

        if optimizer is not None:
            optimizer.zero_grad()

        sub_mix = current_mix[active_idx]
        target_pred, residual_pred, stop_logit = model.separate_step(sub_mix)

        sub_remaining = [remaining[b] for b in active_idx]
        sep_loss, chosen_idx = or_pit_step_loss(target_pred, residual_pred, sub_remaining)

        stop_labels = torch.tensor(
            [float(recursion_labels[b][k]) for b in active_idx], device=device,
        )
        stop_loss = F.binary_cross_entropy_with_logits(stop_logit, stop_labels)

        step_loss = sep_loss + stop_loss_weight * stop_loss

        if optimizer is not None:
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        step_sep_losses.append(sep_loss.item())
        step_stop_losses.append(stop_loss.item())

        with torch.no_grad():
            for i, b in enumerate(active_idx):
                matched_source = remaining[b][chosen_idx[i]].unsqueeze(0)
                snri = compute_si_snri(
                    target_pred[i:i + 1], matched_source, sub_mix[i:i + 1],
                )
                step_si_snris.append(snri)

                pred_bool = torch.sigmoid(stop_logit[i]).item() > 0.5
                true_bool = bool(recursion_labels[b][k])
                stop_stats.update(pred_bool, true_bool)

        # Advance recursion: drop the matched source, feed the predicted
        # residual forward as next step's mixture. Always detached, since
        # each step is now optimized independently (per-step backward above).
        new_current_mix = current_mix.clone()
        for i, b in enumerate(active_idx):
            chosen = chosen_idx[i]
            del remaining[b][chosen]
            new_current_mix[b] = residual_pred[i].detach()
        current_mix = new_current_mix

    metrics = {
        "sep_loss": float(np.mean(step_sep_losses)),
        "stop_loss": float(np.mean(step_stop_losses)),
        "total_loss": float(np.mean(step_sep_losses) + stop_loss_weight * np.mean(step_stop_losses)),
        "mean_si_snri": float(np.mean(step_si_snris)) if step_si_snris else float("nan"),
        "stop_accuracy": stop_stats.accuracy,
        "stop_precision": stop_stats.precision,
        "stop_recall": stop_stats.recall,
    }
    return metrics


# --------------------------------------------------------------------------
# Validation on the static eval sets built by build_eval_sets.py
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, eval_root, device):
    """
    eval_root: e.g. metadata/eval_sets/dev-clean
    Walks each Nmix/ subfolder, runs deterministic recursion for exactly
    N steps (N known from meta.json), reports SI-SNRi per N and stopping
    classifier precision/recall per N.
    """
    model.eval()
    results = {}

    if not os.path.isdir(eval_root):
        print(f"  [WARN] Eval root not found: {eval_root} -- skipping validation.")
        return results

    n_dirs = sorted(
        d for d in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, d)) and d.endswith("mix")
    )

    for n_dir in n_dirs:
        n_speakers = int(n_dir.replace("mix", ""))
        mix_dir = os.path.join(eval_root, n_dir)
        mix_files = sorted(glob.glob(os.path.join(mix_dir, "*_mixture.wav")))

        si_snri_values = []
        stop_stats = StoppingStats()

        for mix_path in mix_files:
            mix_id = os.path.basename(mix_path).replace("_mixture.wav", "")
            meta_path = os.path.join(mix_dir, f"{mix_id}_meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            n_spk = meta["n_speakers"]
            true_recursion_labels = meta["recursion_labels"]

            mixture = load_wav_simple(mix_path).to(device).unsqueeze(0)  # [1, 1, L]

            src_paths = sorted(glob.glob(os.path.join(mix_dir, f"{mix_id}_s*.wav")))
            true_sources = []
            for sp in src_paths:
                wav = load_wav_simple(sp)
                true_sources.append(wav.to(device))

            remaining = list(true_sources)
            current_mix = mixture

            for k in range(n_spk):
                target_pred, residual_pred, stop_logit = model.separate_step(current_mix)

                sisnrs = [
                    si_snr(target_pred, s.unsqueeze(0)).item() for s in remaining
                ]
                best_i = int(np.argmax(sisnrs))
                mix_si = si_snr(current_mix, remaining[best_i].unsqueeze(0)).item()
                si_snri_values.append(sisnrs[best_i] - mix_si)

                pred_bool = torch.sigmoid(stop_logit).item() > 0.5
                true_bool = bool(true_recursion_labels[k])
                stop_stats.update(pred_bool, true_bool)

                del remaining[best_i]
                current_mix = residual_pred

        results[n_speakers] = {
            "si_snri": summarize_si_snri_list(si_snri_values),
            "stopping": stop_stats.summary(),
        }

    model.train()
    return results


def print_eval_results(results, tag=""):
    print(f"\n  --- Validation results {tag} ---")
    for n_speakers in sorted(results.keys()):
        r = results[n_speakers]
        s = r["si_snri"]
        st = r["stopping"]
        print(
            f"    {n_speakers}-speaker: "
            f"SI-SNRi = {s['mean']:.2f} +/- {s['std']:.2f} dB (n={s['n']}) | "
            f"stop acc={st['accuracy']:.3f} prec={st['precision']:.3f} rec={st['recall']:.3f}"
        )


def overall_mean_si_snri(results):
    vals = [r["si_snri"]["mean"] for r in results.values() if not np.isnan(r["si_snri"]["mean"])]
    return float(np.mean(vals)) if vals else float("-inf")


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_dir", type=str, default=os.path.join(WORKSPACE_ROOT, "metadata"))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(WORKSPACE_ROOT, "cache"))
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(WORKSPACE_ROOT, "checkpoints"))
    parser.add_argument("--log_dir", type=str, default=os.path.join(WORKSPACE_ROOT, "logs"))

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_len_sec", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--stop_loss_weight", type=float, default=0.5)
    parser.add_argument("--total_curriculum_steps", type=int, default=100000,
                         help="Matches DynamicMixtureDataset's curriculum schedule length.")
    parser.add_argument("--steps_per_epoch", type=int, default=2500)
    parser.add_argument("--val_every_epochs", type=int, default=2)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0,
                         help="Keep at 0: the curriculum step_getter is only "
                              "correct in the main process. See note below.")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint .pt file to resume from.")

    # Model size -- tune these to fit your GPU. Defaults are a reasonably
    # strong Conv-TasNet-scale config; shrink for a laptop/free-tier GPU or
    # for smoke-testing the pipeline before a real run.
    parser.add_argument("--enc_channels", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_stacks", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = FlexIOLiteRecursiveSeparator(
        enc_channels=args.enc_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_stacks=args.num_stacks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    start_epoch = 0
    best_si_snri = float("-inf")

    if args.resume is not None and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_si_snri = ckpt.get("best_si_snri", float("-inf"))

    # ---------------------------------------------------------------
    # Curriculum-aware global step (see NOTE: num_workers must be 0 for
    # this to correctly drive DynamicMixtureDataset's curriculum -- with
    # num_workers > 0, each worker process gets its own frozen copy of this
    # counter's value from when the DataLoader was created, since Python
    # multiprocessing doesn't share this object across processes.)
    # ---------------------------------------------------------------
    global_step_holder = {"value": start_epoch * args.steps_per_epoch}

    def step_getter():
        return global_step_holder["value"]

    train_loader = get_dataloader(
        metadata_dir=args.metadata_dir,
        cache_dir=args.cache_dir,
        split="train-clean-100",
        chunk_len_sec=args.chunk_len_sec,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        total_steps=args.total_curriculum_steps,
        step_getter=step_getter,
        shuffle=True,
    )
    train_iter = iter(train_loader)

    eval_root = os.path.join(args.metadata_dir, "eval_sets", "dev-clean")
    log_csv_path = os.path.join(args.log_dir, "train_log.csv")
    csv_is_new = not os.path.exists(log_csv_path)
    csv_file = open(log_csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow([
            "epoch", "step", "phase_type", "sep_loss", "stop_loss", "total_loss",
            "mean_si_snri", "stop_accuracy", "stop_precision", "stop_recall", "lr",
        ])

    print(f"\nStarting training: {args.epochs} epochs x {args.steps_per_epoch} steps/epoch")
    print(f"Curriculum total steps: {args.total_curriculum_steps}")
    print(f"Logging to: {log_csv_path}")
    print(f"Checkpoints to: {args.checkpoint_dir}\n")

    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        running = {"sep_loss": [], "stop_loss": [], "total_loss": [], "mean_si_snri": []}

        for local_step in range(args.steps_per_epoch):
            try:
                mixtures, sources, noises, recursion_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                mixtures, sources, noises, recursion_labels = next(train_iter)

            metrics = run_training_step(
                model, mixtures, sources, recursion_labels, device,
                stop_loss_weight=args.stop_loss_weight,
                optimizer=optimizer,
            )

            global_step_holder["value"] += 1

            for k in ["sep_loss", "stop_loss", "total_loss", "mean_si_snri"]:
                if not np.isnan(metrics[k]):
                    running[k].append(metrics[k])

            if (local_step + 1) % args.log_every_steps == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch} step {local_step + 1}/{args.steps_per_epoch} "
                    f"(global {global_step_holder['value']}) | "
                    f"loss={metrics['total_loss']:.3f} sep={metrics['sep_loss']:.3f} "
                    f"stop={metrics['stop_loss']:.3f} SI-SNRi={metrics['mean_si_snri']:.2f}dB "
                    f"stop_acc={metrics['stop_accuracy']:.2f} lr={lr_now:.2e}"
                )
                csv_writer.writerow([
                    epoch, global_step_holder["value"], "train",
                    metrics["sep_loss"], metrics["stop_loss"], metrics["total_loss"],
                    metrics["mean_si_snri"], metrics["stop_accuracy"],
                    metrics["stop_precision"], metrics["stop_recall"], lr_now,
                ])
                csv_file.flush()

        epoch_time = time.time() - epoch_start_time
        print(
            f"\nEpoch {epoch} done in {epoch_time:.1f}s | "
            f"avg loss={np.mean(running['total_loss']):.3f} "
            f"avg SI-SNRi={np.mean(running['mean_si_snri']):.2f}dB"
        )

        # -----------------------------------------------------------
        # Validation
        # -----------------------------------------------------------
        if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1:
            val_results = evaluate(model, eval_root, device)
            print_eval_results(val_results, tag=f"(epoch {epoch})")

            mean_val_si_snri = overall_mean_si_snri(val_results)
            scheduler.step(mean_val_si_snri)

            for n_speakers, r in val_results.items():
                csv_writer.writerow([
                    epoch, global_step_holder["value"], f"val_{n_speakers}mix",
                    "", "", "",
                    r["si_snri"]["mean"], r["stopping"]["accuracy"],
                    r["stopping"]["precision"], r["stopping"]["recall"],
                    optimizer.param_groups[0]["lr"],
                ])
            csv_file.flush()

            # Save "latest" checkpoint always, "best" only if improved
            latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_si_snri": best_si_snri,
                "val_results": val_results,
            }, latest_path)

            if mean_val_si_snri > best_si_snri:
                best_si_snri = mean_val_si_snri
                best_path = os.path.join(args.checkpoint_dir, "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_si_snri": best_si_snri,
                    "val_results": val_results,
                }, best_path)
                print(f"  New best mean SI-SNRi: {best_si_snri:.2f} dB -- saved {best_path}")

    csv_file.close()
    print("\nTraining finished.")


if __name__ == "__main__":
    main()

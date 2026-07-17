import os
# Must be set before PyTorch is imported or CUDA initialises.
# expandable_segments:True tells the CUDA allocator to grow existing
# allocations instead of reserving fixed-size blocks, which eliminates
# the heavy fragmentation that causes OOM during the backward pass on
# activation-heavy models like BranchformerBlock + TransformerDecoder.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import json
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dynamic_mixer import DynamicMixDataset, collate_custom_mix
from src.curriculum import CurriculumManager
from src.model import SeparationNetwork
from src.loss import PITLoss


def create_dummy_indices():
    """Create empty fallback index stubs so the dataset never hard-crashes."""
    os.makedirs("data/indices", exist_ok=True)
    if not os.path.exists("data/indices/speech_index.json"):
        with open("data/indices/speech_index.json", "w") as f:
            json.dump([], f)
    if not os.path.exists("data/indices/noise_index.json"):
        with open("data/indices/noise_index.json", "w") as f:
            json.dump([], f)


def find_latest_checkpoint(checkpoint_dir: str):
    """
    Scans checkpoint_dir for files named checkpoint_epoch_<N>.pt and returns
    the path with the highest epoch number, or None if none exist.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _epoch_num(path):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path)
        return int(match.group(1)) if match else -1

    return max(candidates, key=_epoch_num)


def prune_checkpoints(checkpoint_dir: str, keep: int = 3):
    """
    Deletes the oldest checkpoint files, retaining only the `keep` most recent
    ones (sorted by epoch number embedded in the filename).
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    candidates = glob.glob(pattern)

    def _epoch_num(path):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path)
        return int(match.group(1)) if match else -1

    candidates.sort(key=_epoch_num)                          # oldest first
    to_delete = candidates[:-keep] if len(candidates) > keep else []
    for path in to_delete:
        os.remove(path)
        print(f"  [checkpoint] Deleted old checkpoint: {os.path.basename(path)}")


# ── Windows multiprocessing guard ────────────────────────────────────────────
# DataLoader workers are spawned via multiprocessing.spawn on Windows, which
# re-imports this module in every worker process.  Without this guard each
# worker would re-enter the training code, recursively spawning more workers
# and deadlocking.  All dataset / model / training code MUST live here.
if __name__ == "__main__":

    CHECKPOINT_DIR     = "checkpoints"
    NUM_EPOCHS         = 50   # stage1: 0-9 | stage2: 10-29 | stage3: 30-49
    ACCUMULATION_STEPS = 8    # effective batch = batch_size(2) × 8 = 16

    create_dummy_indices()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    if use_cuda:
        print(f"Using device: {device} — {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device}")

    curriculum = CurriculumManager()

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Speech: LibriSpeech train-clean-360  (104,014 files)
    # Noise : WHAMR!                       ( 20,000 files)
    # 95 / 5 deterministic split — disjoint, no shuffle, stable across restarts.
    SPEECH_INDEX = "data/indices/speech_index.json"
    NOISE_INDEX  = "data/indices/noise_index.json"

    train_dataset = DynamicMixDataset(
        speech_index_path=SPEECH_INDEX,
        noise_index_path=NOISE_INDEX,
        curriculum=curriculum,
        max_length_sec=4.0,
        split="train",        # first 95% → ~98,813 speech / ~19,000 noise
    )

    val_dataset = DynamicMixDataset(
        speech_index_path=SPEECH_INDEX,
        noise_index_path=NOISE_INDEX,
        curriculum=curriculum,
        max_length_sec=4.0,
        split="val",          # last 5%  → ~5,201 speech / ~1,000 noise
    )

    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # persistent_workers=False — workers re-spawn each epoch so they pick up
    # the updated self.epoch from dataset.set_epoch(), keeping curriculum stages
    # correct.  With persistent=True the workers would cache the epoch-0 config
    # forever, trapping the model in Stage 1.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=collate_custom_mix,
        num_workers=4,
        persistent_workers=False,
        pin_memory=use_cuda,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        collate_fn=collate_custom_mix,
        num_workers=4,
        persistent_workers=False,
        pin_memory=use_cuda,
        shuffle=False,        # deterministic val order every epoch
    )

    model     = SeparationNetwork().to(device)
    criterion = PITLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        print(f"[checkpoint] Resuming from {latest_ckpt}")
        ckpt        = torch.load(latest_ckpt, map_location=device)
        start_epoch = ckpt["epoch"] + 1          # next epoch after the saved one
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        prev_train = ckpt.get("epoch_loss", float("nan"))
        prev_val   = ckpt.get("val_loss",   float("nan"))
        print(
            f"[checkpoint] Resuming at epoch {start_epoch} "
            f"(train loss: {prev_train:.4f} | val loss: {prev_val:.4f})"
        )
    else:
        start_epoch = 0
        print("[checkpoint] No checkpoint found — starting from epoch 0.")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):

        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        train_dataset.set_epoch(epoch)

        epoch_loss       = 0.0
        accumulated_loss = 0.0

        # zero_grad OUTSIDE the batch loop so gradients accumulate across
        # ACCUMULATION_STEPS micro-batches before each optimizer step.
        optimizer.zero_grad()

        for batch_idx, (mixed_audio, target_waveforms, configs) in enumerate(train_dataloader):

            mixed_audio      = mixed_audio.to(device, non_blocking=True)
            target_waveforms = target_waveforms.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                est_sources, speaker_probs = model(mixed_audio)
                # Divide by ACCUMULATION_STEPS so the summed gradient over the
                # window equals a true batch-8 gradient.
                loss = criterion(est_sources, target_waveforms, speaker_probs)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            epoch_loss       += loss.item() * ACCUMULATION_STEPS
            accumulated_loss += loss.item()

            is_update_step = (
                (batch_idx + 1) % ACCUMULATION_STEPS == 0
                or (batch_idx + 1) == len(train_dataloader)
            )

            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                print(
                    f"  Epoch {epoch+1}/{NUM_EPOCHS} "
                    f"| batch {batch_idx+1:>6d}/{len(train_dataloader)} "
                    f"| accum loss: {accumulated_loss:.4f}"
                )
                accumulated_loss = 0.0

        avg_train_loss = epoch_loss / len(train_dataloader)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_dataset.set_epoch(epoch)   # keep curriculum stage consistent

        val_loss_total = 0.0

        with torch.no_grad():
            for mixed_audio, target_waveforms, configs in val_dataloader:

                mixed_audio      = mixed_audio.to(device, non_blocking=True)
                target_waveforms = target_waveforms.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    est_sources, speaker_probs = model(mixed_audio)
                    val_loss = criterion(est_sources, target_waveforms, speaker_probs)

                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / len(val_dataloader)

        # ── Epoch summary ─────────────────────────────────────────────────────
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss:   {avg_val_loss:.4f}"
        )

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "epoch_loss":           avg_train_loss,
                "val_loss":             avg_val_loss,
            },
            ckpt_path,
        )
        print(f"[checkpoint] Saved {ckpt_path}")

        # Retain only the 3 most recent checkpoints to conserve disk space.
        prune_checkpoints(CHECKPOINT_DIR, keep=3)

    print("Training completed successfully.")

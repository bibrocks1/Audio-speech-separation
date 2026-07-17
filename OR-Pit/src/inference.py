"""
inference.py
────────────
Load a trained OR-PiT checkpoint and separate a mixed audio file into individual
speaker tracks.

Usage:
    python -m src.inference \
        --input_mix /path/to/mixed_audio.wav

    python -m src.inference \
        --checkpoint ./checkpoints/epoch_20.pt \
        --input_mix /path/to/mixed_audio.wav \
        --output_dir ./separated_outputs/
"""

import os
import argparse

import torch
import torchaudio
import torchaudio.transforms as T

from src.model import SeparationNetwork


def load_audio(path: str, target_sr: int = 16000):
    """
    Load audio file and resample to target_sr if needed.
    Returns: (waveform, sample_rate) where waveform is mono and on CPU.
    """
    waveform, sr = torchaudio.load(path)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        print(f"  Resampling from {sr} Hz to {target_sr} Hz…")
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    print(f"  Loaded: {waveform.shape} (mono) @ {sr} Hz")
    return waveform, sr


def separate(
    checkpoint_path: str,
    input_mix_path: str,
    output_dir: str,
    device: torch.device,
):
    """
    Load checkpoint, separate the mixed audio, and save individual sources.
    """
    print(f"\n[SEPARATION]")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Input mix  : {input_mix_path}")
    print(f"  Device     : {device}")

    # ── Load audio ────────────────────────────────────────────────────────────
    print(f"\n[LOAD INPUT]")
    mixed_waveform, sr = load_audio(input_mix_path, target_sr=16000)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[LOAD MODEL]")
    model = SeparationNetwork().to(device)
    model.eval()

    # Load checkpoint
    print(f"  Loading checkpoint…")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded state from epoch {ckpt.get('epoch', 'unknown')}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    print(f"\n[INFERENCE]")

    # Prepare input: [1, Time] -> [1, 1, Time] (Batch, Channels, Samples)
    mixed_input = mixed_waveform.unsqueeze(0).to(device)  # [1, 1, Time]

    with torch.no_grad():
        # est_sources: [Batch, Max_Speakers, Time]
        est_sources, speaker_probs = model(mixed_input)

    print(f"  Output shape : {est_sources.shape}")
    print(f"  (Batch=1, Max_Speakers={est_sources.shape[1]}, Samples={est_sources.shape[2]})")

    # ── Save separated sources ────────────────────────────────────────────────
    print(f"\n[SAVE OUTPUTS]")
    os.makedirs(output_dir, exist_ok=True)

    # Extract batch 0, move to CPU
    separated = est_sources[0].detach().cpu()  # [Max_Speakers, Time]

    num_speakers = separated.shape[0]
    for i in range(num_speakers):
        source = separated[i].unsqueeze(0)  # [1, Time] for torchaudio.save

        output_path = os.path.join(output_dir, f"output_source_{i+1}.wav")
        torchaudio.save(output_path, source, sr)
        print(f"  Saved: {output_path}  ({source.shape[1]} samples)")

    print(f"\n✓ Separation complete. {num_speakers} source(s) saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Separate a mixed audio file using a trained OR-PiT model."
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/best_model.pt",
        help="Path to the checkpoint .pt file (default: ./checkpoints/best_model.pt).",
    )
    parser.add_argument(
        "--input_mix",
        required=True,
        help="Path to the mixed/dirty audio file to separate.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/",
        help="Output directory for separated source files (default: ./outputs/).",
    )
    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return

    if not os.path.exists(args.input_mix):
        print(f"[ERROR] Input file not found: {args.input_mix}")
        return

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Separate ──────────────────────────────────────────────────────────────
    separate(
        checkpoint_path=args.checkpoint,
        input_mix_path=args.input_mix,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()

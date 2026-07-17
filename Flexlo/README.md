# FlexIO-Lite (Recursive OR-PIT Separator)

**Memory-Efficient Recursive Speech Separation with Stopping Classifier**

## Overview

FlexIO-Lite is a **recursive speech separation** model that extracts speakers one-at-a-time from mixed audio. Unlike fixed-output models, FlexIO-Lite operates through an **iterative loop**: at each step, it separates one speaker and feeds the residual back as the new mixture.

### Key Innovations

- **Recursive Extraction**: Single model handles arbitrary speaker counts (2–6+) with constant memory
- **FiLM-Conditioned TCN**: Prompt vector modulates all convolutional blocks for flexible control
- **Per-Step Backward**: Memory-efficient training that scales to any speaker count
- **Stopping Classifier**: Automatically determines when all speakers have been extracted

### Key Differentiator from OR-PiT and UME

| Feature | Flexlo | OR-PiT | UME |
|---------|--------|--------|-----|
| **Output** | Audio waveforms (recursive) | Audio waveforms (fixed N) | Text + time anchors |
| **Speaker Count** | Stopping classifier determines | Halting classifier predicts | Built-in diarization |
| **Architecture** | TCN + FiLM | Branchformer + Attractors | MoE + LLM |
| **Memory** | Constant (per-step backward) | Scales with speaker count | Scales with sequence length |
| **Training** | Recursive loop, O(N) per step | O(N³) Hungarian algorithm | O(N) per frame |

---

## Curriculum Learning & Dynamic Mixing

### Four-Phase Curriculum

The `dataset.py` module implements curriculum learning with increasing difficulty:

```python
# Phase 1 (0-25% of training):
# - 2 speakers only
# - Full overlap (1.0)
# - No noise, no reverb
n_speakers = 2
overlap_ratio = 1.0
use_noise = False
use_reverb = False

# Phase 2 (25-50%):
# - 2-3 speakers
# - Variable overlap (0.0 to 1.0)
# - No augmentations
n_speakers = random(2, 3)
overlap_ratio = random(0.0, 1.0)

# Phase 3 (50-75%):
# - 3-4 speakers
# - Variable overlap
# - Noise + reverb active
n_speakers = random(3, 4)
use_noise = True
use_reverb = True

# Phase 4 (75-100%):
# - 4-6 speakers (maximum capacity)
# - Full overlap range
# - All augmentations
n_speakers = random(4, 6)
```

### Dynamic Mixing Pipeline

```python
# scripts/dynamic_mixer.py

def generate_mixture(source_index, noise_files, rir_files, n_speakers, overlap_ratio):
    # 1. Sample N speakers (ensure different speaker IDs)
    chosen_speakers = set()
    sources = []
    for _ in range(n_speakers):
        meta = sample_utterance(source_index, exclude=chosen_speakers)
        chosen_speakers.add(meta["speaker_id"])
        sources.append(load_wav(meta["path"]))
    
    # 2. Normalize loudness (RMS between -33 and -25 dB)
    sources = [normalize_loudness_rms(s, random(-33, -25)) for s in sources]
    
    # 3. Apply overlap staggering
    aligned = apply_overlap_staggering(sources, overlap_ratio, chunk_len_samples)
    
    # 4. Optional: Add reverb (separate RIR per speaker)
    if use_reverb:
        for i, src in enumerate(aligned):
            rir = load_wav(random.choice(rir_files))
            aligned[i] = convolve(src, rir)
    
    # 5. Sum sources → mixture
    mixture = sum(aligned)
    
    # 6. Optional: Add noise at target SNR
    if use_noise:
        noise = load_wav(random.choice(noise_files))
        mixture, noise = add_noise_at_snr(mixture, noise, snr_db=random(10, 25))
    
    # 7. Peak normalize to -1 dBFS
    scale = 0.89 / max(abs(mixture))
    mixture *= scale
    aligned = [s * scale for s in aligned]
    
    # 8. Generate recursion labels
    # [True, True, ..., True, False] for N speakers
    recursion_labels = [True] * (n_speakers - 1) + [False]
    
    return mixture, aligned, noise, recursion_labels
```

### Recursion Labels

The **recursion_labels** encode when to stop:

```python
# For N speakers:
# Step 0: Extract speaker 1 → residual has N-1 speakers → label = True (continue)
# Step 1: Extract speaker 2 → residual has N-2 speakers → label = True (continue)
# ...
# Step N-1: Extract speaker N → residual has 0 speakers → label = False (stop)

recursion_labels = [True] * (n_speakers - 1) + [False]
```

---

## Quick Start

### Installation

```bash
cd Flexlo
pip install torch torchaudio soundfile numpy scipy
```

### Prepare Data

1. Run the data preparation script:

```bash
python prepare_data.py --librispeech_dir /path/to/LibriSpeech
```

This creates:
- `metadata/source_index.json` (speech file metadata)
- `cache/noise_bank/` (noise files)
- `cache/rir_bank/` (room impulse responses)

2. Build evaluation sets:

```bash
python build_eval_sets.py --split dev-clean --num_mixtures 100
```

### Run Training

```bash
python train.py --epochs 40 --batch_size 4 --lr 1e-3
```

---

## Kaggle Deployment

### Step 1: Configure Paths

```python
# Auto-detects Kaggle paths
metadata_dir, cache_dir = resolve_kaggle_paths(
    metadata_dir="metadata",
    cache_dir="cache"
)
```

### Step 2: Set Up DataLoader

```python
from dataset import get_dataloader

# Curriculum-aware global step counter
global_step = {"value": 0}
def step_getter():
    return global_step["value"]

train_loader = get_dataloader(
    metadata_dir=metadata_dir,
    cache_dir=cache_dir,
    split="train-clean-100",
    chunk_len_sec=4.0,
    batch_size=4,
    num_workers=0,  # Important: must be 0 for correct curriculum
    total_steps=100000,
    step_getter=step_getter,
    shuffle=True
)
```

### Step 3: Run Training with GPU

```python
import torch
from train import run_training_step

model = FlexIOLiteRecursiveSeparator(
    enc_channels=256,
    hidden_channels=512,
    num_blocks=8,
    num_stacks=2
).to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(40):
    for mixtures, sources, noises, recursion_labels in train_loader:
        metrics = run_training_step(
            model, mixtures, sources, recursion_labels,
            device="cuda",
            stop_loss_weight=0.5,
            optimizer=optimizer
        )
        
        global_step["value"] += 1
        
        if global_step["value"] % 50 == 0:
            print(f"Step {global_step['value']}: "
                  f"loss={metrics['total_loss']:.3f} "
                  f"SI-SNRi={metrics['mean_si_snri']:.2f}dB")
```

### Step 4: Validate on Static Eval Sets

```python
from train import evaluate, print_eval_results

val_results = evaluate(model, "metadata/eval_sets/dev-clean", device="cuda")
print_eval_results(val_results)

# Output:
#   --- Validation results ---
#     2-speaker: SI-SNRi = 8.52 +/- 2.31 dB (n=200) | stop acc=0.912 prec=0.895 rec=0.923
#     3-speaker: SI-SNRi = 6.84 +/- 2.89 dB (n=200) | stop acc=0.867 prec=0.841 rec=0.891
```

---

## Inference

### Recursive Separation Loop

```python
import torch
from model import FlexIOLiteRecursiveSeparator

# Load trained model
model = FlexIOLiteRecursiveSeparator().to(device)
model.load_state_dict(torch.load("checkpoints/best.pt")["model_state_dict"])
model.eval()

# Load mixed audio
mixture = load_audio("mixed.wav").unsqueeze(0).to(device)  # [1, 1, T]

separated_sources = []
current_mix = mixture
max_iterations = 10

with torch.no_grad():
    for step in range(max_iterations):
        # Extract one speaker + residual
        target, residual, stop_logit = model.separate_step(current_mix)
        
        # Save extracted speaker
        separated_sources.append(target.squeeze(0).cpu())
        
        # Check stopping criterion
        if torch.sigmoid(stop_logit).item() < 0.5:
            print(f"Stopped after {step + 1} speakers")
            break
        
        # Feed residual as new mixture
        current_mix = residual

# Save separated sources
for i, source in enumerate(separated_sources):
    torchaudio.save(f"separated_speaker_{i+1}.wav", source, 16000)
```

### SI-SNRi Evaluation

```python
from model import si_snr, compute_si_snri

# For each extracted source, match to best ground truth
remaining_targets = list(ground_truth_sources)
snri_values = []

for est_source in separated_sources:
    # Find best matching ground truth
    best_snri = -float('inf')
    best_idx = 0
    
    for i, true_source in enumerate(remaining_targets):
        snri = compute_si_snri(
            est_source.unsqueeze(0),
            true_source.unsqueeze(0),
            mixture
        )
        if snri > best_snri:
            best_snri = snri
            best_idx = i
    
    snri_values.append(best_snri)
    del remaining_targets[best_idx]

print(f"Mean SI-SNRi: {np.mean(snri_values):.2f} dB")
```

---

## Model Checkpoints

### Checkpoint Structure

```python
checkpoint = {
    "epoch": 15,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_si_snri": 7.85,  # Best validation SI-SNRi
    "val_results": {
        2: {"si_snri": {"mean": 8.52, "std": 2.31, "n": 200}},
        3: {"si_snri": {"mean": 6.84, "std": 2.89, "n": 200}},
    }
}
```

### Resume Training

```python
python train.py --resume checkpoints/latest.pt
```

### Best Checkpoint Selection

The model keeps track of the best checkpoint by **mean validation SI-SNRi**:

```python
mean_si_snri = np.mean([r["si_snri"]["mean"] for r in val_results.values()])
if mean_si_snri > best_si_snri:
    torch.save(checkpoint, "checkpoints/best.pt")
```

---

## Training Logs

Training metrics are logged to `logs/train_log.csv`:

```csv
epoch,step,phase_type,sep_loss,stop_loss,total_loss,mean_si_snri,stop_accuracy,stop_precision,stop_recall,lr
0,50,train,-8.234,0.412,8.646,7.82,0.845,0.823,0.867,1.00e-03
0,100,train,-8.567,0.389,8.956,8.12,0.856,0.841,0.872,1.00e-03
```

---

## Architecture Summary

```
Input Waveform [B, 1, T]
    ↓ Encoder: Conv1d(1→256, k=16, s=8)
Embedding H [B, 256, T']
    ↓ SeparatorCore (TCN + FiLM)
    ├─ GroupNorm → Bottleneck Conv1d
    ├─ 16× TCNBlocks with dilated convolutions
    │   └─ FiLM conditioning from prompt vector
    ├─ Mask Output: sigmoid → split into mask_target, mask_residual
    └─ Apply Masks: H ⊙ mask
Target Emb, Residual Emb [B, 256, T']
    ↓ Decoder: ConvTranspose1d(256→1, k=16, s=8)
Target Wav, Residual Wav [B, 1, T]
    ↓ StoppingClassifier
Stop Logit [B]  →  sigmoid > 0.5 ? continue : stop
```

---

## Citation

```bibtex
@misc{flexlo2024,
  title={FlexIO-Lite: Memory-Efficient Recursive Speech Separation with Stopping Classifier},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/flexlo}}
}
```

---

## License

MIT License

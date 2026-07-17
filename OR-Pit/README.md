# OR-PiT (One-and-Rest Permutation Invariant Training)

**Speaker-Count-Agnostic Neural Speech Separation**

## Overview

OR-PiT is a neural speech separation model that extracts individual speakers from mixed audio without knowing the speaker count in advance. A **single trained model** handles anywhere from 2 to 6+ overlapping speakers.

### Key Innovations

- **Permutation Invariant Training with Hungarian Algorithm**: O(N³) instead of O(N!) complexity
- **Attractor-Based Speaker Extraction**: Learnable speaker prototypes generalize to unseen voices
- **Branchformer Encoder**: Combines self-attention (global) with convolutional gated MLP (local)
- **Halting Classifier**: Automatically determines how many speakers are present

### Key Differentiator from Flexlo and UME

| Feature | OR-PiT | Flexlo | UME |
|---------|--------|--------|-----|
| **Output** | Audio waveforms (fixed N) | Audio waveforms (recursive) | Text + time anchors |
| **Speaker Count** | Predicted by model | Stopping classifier | Built-in diarization |
| **Architecture** | Branchformer + Attractors | TCN + FiLM | MoE + LLM |
| **Inference** | Single forward pass | Recursive loop | Single forward pass |
| **Training Complexity** | O(N³) per batch | O(N) per step | O(N) per frame |

---

## Curriculum Learning & Dynamic Mixing

### Progressive Training Stages

The `src/curriculum.py` module implements a 4-stage curriculum:

```python
# Stage 1 (0-25% of training): 2 speakers, full overlap, no noise/reverb
num_speakers = 2
overlap = 1.0  # fully overlapped
use_noise = False
use_reverb = False

# Stage 2 (25-50%): 2-3 speakers, variable overlap, no augmentations
num_speakers = random(2, 3)
overlap = random(0.0, 1.0)
use_noise = False
use_reverb = False

# Stage 3 (50-75%): 3-4 speakers, full overlap range, noise + reverb
num_speakers = random(3, 4)
overlap = random(0.0, 1.0)
use_noise = True
use_reverb = True

# Stage 4 (75-100%): 4-6 speakers, full augmentations
num_speakers = random(4, 6)
overlap = random(0.0, 1.0)
use_noise = True
use_reverb = True
```

### Dynamic Mixing Pipeline

```python
# src/dynamic_mixer.py

class DynamicMixDataset(Dataset):
    def __getitem__(self, idx):
        # 1. Sample speakers from index
        speech_files = random.sample(speech_index, num_speakers)
        
        # 2. Load and preprocess audio
        signals = [load_and_preprocess(f) for f in speech_files]
        
        # 3. Apply time shifting based on overlap ratio
        aligned = apply_overlap_staggering(signals, overlap_ratio)
        
        # 4. Optional: Add synthetic reverb (RIR convolution)
        if use_reverb:
            rir = generate_synthetic_rir(duration=0.3s)
            aligned = [fftconvolve(sig, rir) for sig in aligned]
        
        # 5. Optional: Add noise at specified SNR
        if use_noise:
            noise = load_noise_sample()
            mixture = add_noise_at_snr(sum(aligned), noise, snr_db)
        
        return mixture, aligned_speakers, config
```

### Acoustic Simulation

**Synthetic RIR (Room Impulse Response):**
```python
def _generate_synthetic_rir(self):
    rir_length = int(0.3 * sample_rate)  # 300ms tail
    rir = torch.randn(1, rir_length)
    decay = torch.exp(-torch.linspace(0, 10, rir_length))
    return rir * decay  # exponential decay
```

**SNR Scaling:**
```python
# Scale noise to target SNR
p_speech = mean(mixture²)
p_noise = mean(noise²)
scale = sqrt((p_speech / p_noise) × 10^(-snr_db/10))
scaled_noise = noise × scale
```

---

## Quick Start

### Installation

```bash
cd OR-Pit
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Prepare Data

1. Download LibriSpeech (or similar speech corpus)
2. Download WHAMR! noise corpus
3. Run the indexer:

```bash
python src/indexer.py --speech_dir /path/to/librispeech --noise_dir /path/to/whamr
```

This creates:
- `data/indices/speech_index.json` (104,014 entries)
- `data/indices/noise_index.json` (20,000 entries)

### Run Training

```bash
python src/train.py --epochs 40 --batch_size 2 --lr 1e-3
```

---

## Kaggle Deployment

### Step 1: Build Index Files

```bash
python src/build_kaggle_indices.py \
    --speech_dir /kaggle/input/librispeech \
    --noise_dir /kaggle/input/whamr
```

### Step 2: Configure for 16GB GPU

The default config is optimized for **Tesla T4 (16 GB VRAM)**:

```python
# In src/train.py
batch_size = 2           # Micro-batch
accumulation_steps = 8   # Effective batch = 16
max_length_sec = 4.0     # 4-second windows
```

### Step 3: Enable GPU and Run

```bash
# In Kaggle notebook
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!python src/train.py --epochs 40 --batch_size 2
```

### Step 4: Monitor Training

```bash
# View metrics without GPU
python src/view_metrics.py --checkpoint_dir checkpoints/
```

Output:
```
──────────────────────────────────────────────────────────
File                     Epoch    Train Loss      Val Loss
──────────────────────────────────────────────────────────
checkpoint_epoch_1.pt        1        2.5618        2.3846
checkpoint_epoch_2.pt        2        2.2466        1.8423
checkpoint_epoch_7.pt        7        1.3973        1.3578
──────────────────────────────────────────────────────────
  ★  Best val loss: 1.3578 at epoch 7
```

---

## Inference

### Separate a Mixed Audio File

```bash
python -m src.inference \
    --checkpoint ./checkpoints/best_model.pt \
    --input_mix /path/to/mixed_audio.wav \
    --output_dir ./separated_outputs/
```

### Output

The script generates:
```
separated_outputs/
├── output_source_1.wav
├── output_source_2.wav
├── output_source_3.wav
├── ...
└── output_source_N.wav
```

where N is determined by the model's halting classifier.

### Python API

```python
import torch
from src.model import SeparationNetwork
import torchaudio

# Load model
model = SeparationNetwork().to(device)
ckpt = torch.load("checkpoints/best.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load mixed audio
mixture, sr = torchaudio.load("mixed.wav")
if sr != 16000:
    mixture = torchaudio.functional.resample(mixture, sr, 16000)

# Separate
with torch.no_grad():
    separated, speaker_probs = model(mixture.unsqueeze(0).unsqueeze(0).to(device))

# separated: [1, N_speakers, Time]
for i in range(separated.shape[1]):
    torchaudio.save(f"speaker_{i+1}.wav", separated[0, i].cpu(), 16000)
```

---

## Model Checkpoints

### Checkpoint Structure

```python
checkpoint = {
    "epoch": 7,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),  # for AMP
    "epoch_loss": 1.3973,
    "val_loss": 1.3578
}
```

### Resume Training

```bash
python src/train.py --resume ./checkpoints/checkpoint_epoch_7.pt
```

### Automatic Pruning

Only the **3 most recent checkpoints** are kept to save disk space.

---

## Training Metrics

### Validation Set

The validation set is **permanently locked to Stage 3 curriculum** (3-4 speakers, noise + reverb) to provide a static benchmark:

```python
val_dataset.set_epoch(50)  # Permanently Stage 3
```

### SI-SDR Improvement

The model is evaluated on **SI-SNRi (SI-SNR improvement)**:

```
SI-SNRi = SI-SNR(separated, target) - SI-SNR(mixture, target)
```

This measures how much better the separated output is compared to the raw mixture.

---

## Architecture Summary

```
Input Waveform [B, 1, T]
    ↓ Conv1d (learned filterbank)
Mixture Embedding [B, 256, T']
    ↓ BranchformerBlock
    ↓ Self-Attention + ConvGatedMLP
Refined Embedding [B, 256, T']
    ↓ Temporal Pooling (160×)
Pooled Embedding [B, 256, 50]
    ↓ TransformerDecoderAttractor
Attractors [B, 6, 256]  +  Speaker Probs [B, 6, 1]
    ↓ Soft Mask Computation
    ↓ einsum + sigmoid
Masks [B, 6, 256, T']
    ↓ ConvTranspose1d
Separated Waveforms [B, 6, T]
```

---

## Citation

```bibtex
@misc{orpit2024,
  title={OR-PiT: One-and-Rest Permutation Invariant Training for Speaker-Count-Agnostic Speech Separation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/or-pit}}
}
```

---

## License

MIT License

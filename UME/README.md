# UME (Universal Model for Meeting Transcription)

**End-to-End Speech Separation + ASR with Generative LLM Decoding**

## Overview

UME is a **unified speech separation and transcription pipeline** that directly outputs transcripts interleaved with temporal time anchors, eliminating the need for separate diarization systems. It combines:

- **Sparse Mixture-of-Experts (MoE)** architecture upcycled from pre-trained ASR weights
- **Dynamic Threshold Gating** for adaptive expert activation
- **Generative LLM backend** (TagSpeech) for transcript generation
- **Sortformer** for speaker ordering and diarization

### Key Differentiator from OR-PiT and Flexlo

| Feature | UME | OR-PiT | Flexlo |
|---------|-----|--------|--------|
| **Output** | Text + time anchors | Audio waveforms | Audio waveforms |
| **Primary Use** | Meeting transcription | Source separation | Source separation |
| **Architecture** | MoE + LLM decoder | Branchformer + Attractors | TCN + FiLM |
| **Diarization** | Built-in (time anchors) | None | Stopping classifier |
| **Pre-trained** | Yes (OWSM weights) | No | No |
| **Training** | 2-stage (warmup + escalation) | End-to-end | Recursive loop |

---

## Curriculum Learning & Dynamic Mixing

### Three-Tier Dataset Generation

The `progressive_dataset.py` module generates curriculum datasets with increasing difficulty:

```python
# Tier 1: 1-Speaker Clean Baseline
- Single speaker, no overlap
- Moderate reverb (T60 = 50-100 ms)
- SNR: 5-15 dB

# Tier 2: 2-Speaker Overlapped
- Two speakers with variable overlap (10% → 90%)
- Curriculum: first half capped at 10% overlap, second half progressive
- SNR: 0-12 dB

# Tier 3: 3-Speaker Dense
- Three speakers with dense overlap
- Multiple entry points (staggered starts)
- SNR: -3 to 8 dB (challenging)
```

### Acoustic Simulation Pipeline

```python
# Fast RIR simulation
rir = fast_rir_simulate(decay=0.05-0.1)
reverb_sig = np.convolve(clean_sig, rir, mode='same')

# CHiME-4 style modulated noise
noise = gaussian_noise × (0.6(1 + sin(2π·2t)) + 0.15 cos(2π·10t))
noisy_sig = clean_sig + noise × scale_for_snr(snr_db)
```

---

## Quick Start

### Installation

```bash
cd UME
pip install torch torchaudio scipy numpy
```

### Generate Curriculum Dataset

```python
import progressive_dataset as pd

pd.generate_curriculum_datasets(
    num_samples=100,           # samples per tier
    output_dir="curriculum_dataset"
)
```

### Run Full Pipeline

```bash
python run_pipeline.py
```

This will:
1. Generate curriculum datasets (Tier 1, 2, 3)
2. Run Stage 1 warmup on 1-speaker clean data
3. Freeze encoder and activate Sidecar separator
4. Run Stage 2 escalation on 2 and 3-speaker mixtures

---

## Kaggle Deployment

### Step 1: Prepare Data

Upload your LibriSpeech or similar speech corpus to Kaggle. The dataset should contain:
- Audio files (`.wav` or `.flac`)
- Organized by speaker ID

### Step 2: Create Kaggle Notebook

```python
# In Kaggle notebook
import os
import sys

# Mount your dataset
dataset_path = "/kaggle/input/your-speech-dataset"

# Clone/copy UME code
# Run curriculum generation
import progressive_dataset as pd
pd.generate_curriculum_datasets(
    num_samples=1000,
    output_dir="curriculum_dataset"
)
```

### Step 3: Enable GPU Accelerator

1. Go to notebook settings → Accelerator → GPU P100 or T4
2. Enable "Save to GitHub" for persistence

### Step 4: Run Training

```python
import run_pipeline

# Or run stages manually:
import train_regimen as tr
import ume_architecture as ume

model = ume.iVPipeline(vocab_size=100, hidden_dim=256)

# Stage 1
tr.train_stage1_warmup(model, loader1, epochs=5, device="cuda")

# Freeze and activate separator
tr.freeze_encoder_and_insert_sidecar(model)

# Stage 2
tr.train_stage2_escalation(model, loader2, loader3, epochs=5, device="cuda")
```

---

## Inference

UME outputs **transcripts with temporal time anchors** rather than separated audio:

```python
import torch
from ume_architecture import iVPipeline

# Load trained model
model = iVPipeline(vocab_size=100, hidden_dim=256)
model.load_state_dict(torch.load("checkpoints/best.pt")["model_state_dict"])
model.eval()

# Prepare input (mixture waveform → frame embeddings)
# Note: Requires pre-processing to 256-dim frame embeddings

with torch.no_grad():
    logits, balance_loss = model(
        mix_frames,           # [B, Frames, 256]
        target_tokens,        # [B, SeqLen] (use dummy during inference)
        clap_context,         # [B, 256] room context
        num_speakers=2
    )

# Decode logits to tokens
predicted_tokens = torch.argmax(logits, dim=-1)

# Convert to text using vocabulary
transcript = decode_tokens(predicted_tokens)
# Output: "Hello [TIME_1] how are you [TIME_2] I'm fine [EOS]"
```

### Time Anchor Interpretation

- `[TIME_k]` tokens indicate **speaker transitions**
- The time value can be approximated from frame position: `time = frame_index × hop_size / sample_rate`

---

## Model Checkpoints

After training, checkpoints are saved with:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, f"checkpoints/ume_epoch_{epoch}.pt")
```

### View Training Metrics

```bash
# On CPU (no GPU needed)
python -c "
import torch
ckpt = torch.load('checkpoints/ume_epoch_5.pt', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
# Add custom metric printing as needed
"
```

---

## Architecture Summary

```
Input Waveform (16 kHz)
    ↓ Log-Mel Spectrogram
    ↓ Feature Projection
Dense Encoder (OWSM pre-trained)
    ↓ MoE Upcycling
Sparse MoE Layer (4 experts)
    ↓ Dynamic Threshold Routing
Sidecar TCN Separator (Stage 2)
    ↓
Sortformer (Speaker Ordering)
    ↓
TagSpeech LLM Decoder
    ↓
Transcript + Time Anchors
```

---

## Citation

If you use UME in your research, please cite:

```bibtex
@misc{ume2024,
  title={UME: Universal Model for Meeting Transcription},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/ume}}
}
```

---

## License

MIT License

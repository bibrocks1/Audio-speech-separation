# UME (Universal Model for Meeting Transcription) Architecture

## Overview

The UME (iV Pipeline) is an end-to-end speech separation and recognition system that combines **Sparse Mixture-of-Experts (MoE)** architecture with **generative LLM decoding**. Unlike traditional separation models that output audio waveforms, UME directly emits **transcripts interleaved with temporal time anchors**, making it uniquely suited for meeting transcription and diarization tasks.

---

## Acoustic Physics

### Time-Domain Waveform Processing

The UME pipeline begins with a **pre-trained Dense ASR Encoder** (derived from OWSMv3.1), which processes 80-dimensional log-Mel filterbank features extracted from the input waveform at 16 kHz.

**Key acoustic properties:**

1. **Frame-level representation**: The input waveform is segmented into overlapping frames (typically 25 ms windows with 10 ms hop), yielding ~100 frames per second.

2. **Dense feature projection**: Each frame is projected into a 256-dimensional hidden embedding space via a linear layer:
   ```
   h_t = LayerNorm(Linear(x_t))  where x_t ∈ ℝ^80, h_t ∈ ℝ^256
   ```

3. **Reverberation context (CLAP)**: A separate **CLAP acoustic context embedding** captures room characteristics (reverberation time T60, direct-to-reverberant ratio DRR). This is concatenated with the frame embeddings for routing decisions:
   ```
   router_input = [h_t; clap_context] ∈ ℝ^512
   ```

### Overlapping Soundwave Modeling

The MoE layer handles overlapping speakers through **Dynamic Threshold Gating**, which activates a variable number of experts per frame based on the complexity of the acoustic scene:

- **Single-speaker mode (warmup)**: Top-1 routing — only the highest-scoring expert processes the frame
- **Multi-speaker mode (N ≥ 2)**: Dynamic threshold τ = μ(probs) + 0.1·σ(probs) activates all experts with probability > τ

This allows the model to allocate more computational capacity to frames with dense speaker overlap, while remaining efficient on clean speech.

---

## Network Topology

### Stage 1: Dense Encoder → Sparse MoE Upcycling

```
Input Waveform (16 kHz)
    ↓
Log-Mel Spectrogram (80-dim)
    ↓
Feature Projection → 256-dim embeddings
    ↓
Dense FFN Layer (pre-trained OWSM weights)
    ↓
    ┌─────────────────────────────────┐
    │  Sparse MoE Layer (4 experts)   │
    │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
    │  │ E₁  │  │ E₂  │  │ E₃  │  │ E₄  │
    │  └─────┘  └─────┘  └─────┘  └─────┘
    │         ↑ Dynamic Router ↑        │
    └─────────────────────────────────┘
    ↓
MoE Output (256-dim)
```

**Upcycling process:**
- The dense FFN weights from the pre-trained ASR model are **replicated** into 4 expert layers
- Each expert starts as an identical copy, then specializes during fine-tuning
- The router is initialized randomly and trained from scratch

**Load balancing penalty:**
To prevent expert collapse, a curriculum-weighted auxiliary loss is applied:
```
L_balance = N · Σᵢ fᵢ · Pᵢ

where:
  N = number of experts (4)
  fᵢ = fraction of routing decisions to expert i
  Pᵢ = average probability allocated to expert i
```

### Stage 2: Sidecar Separator Branch

After Stage 1 warmup, the encoder is **frozen** and a lightweight **Temporal Convolutional Network (TCN)** sidecar branch is trained:

```
MoE Output (256-dim)
    ↓
    ├──────────────────────────┐
    │                          ↓
    │              Sidecar TCN (residual)
    │              Conv1d(256→256, k=3)
    │              ReLU
    │              Conv1d(256→256, k=3)
    │                          ↓
    └──────────────────→ Add ←─┘
                         ↓
               Separated Embedding
```

**Rationale for residual design:**
- The frozen encoder preserves the pre-trained linguistic knowledge
- The TCN sidecar learns only the **interference patterns** specific to overlapped speech
- This prevents catastrophic forgetting while enabling separation

### Stage 3: Sortformer & Speaker Kernels

```
Separated Embedding
    ↓
┌─────────────────────────────────────┐
│        Sortformer Module            │
│  1. Arrival Time Sorting (ATS)      │
│     - Estimate entry score per frame│
│     - Sort speakers chronologically │
│                                     │
│  2. Sinusoidal Speaker Kernels      │
│     - Learnable embeddings per      │
│       speaker (max_speakers=3)      │
│     - Temporal sinusoidal binding   │
└─────────────────────────────────────┘
    ↓
Speaker-ordered Embeddings
```

**Speaker kernel math:**
```
speaker_frames[t] = Σₖ sin(4π·t/T) · kernel_k

where:
  T = total frames
  k = speaker index
  kernel_k ∈ ℝ^256 (learnable)
```

### Stage 4: Generative LLM Backend (TagSpeech)

```
Speaker-ordered Embeddings
    ↓
┌─────────────────────────────────────┐
│   TagSpeech Transformer Decoder     │
│   (2 layers, 4 heads, d=256)        │
│                                     │
│   Input: [SOS, text_1, time_1, ...] │
│   Output: [text_1, time_1, ..., EOS]│
└─────────────────────────────────────┘
    ↓
Logits (vocab_size=1000)
    ↓
+ CALM Bias Log Probabilities
    ↓
Final Transcript with Time Anchors
```

**Time anchor tokens:**
Special tokens inserted into the transcript to mark speaker transitions:
```
[SOS] Hello [TIME_1] how are you [TIME_2] I'm fine [EOS]
```

---

## Mathematics of Loss

### 1. Cross-Entropy Loss (Transcript Generation)

Standard sequence-to-sequence cross-entropy over the token vocabulary:

```
L_CE = -Σₜ log P(target_t | target_{<t}, encoder_output)
```

### 2. Load Balancing Penalty (MoE Routing)

```
L_balance = N · Σᵢ fᵢ · Pᵢ

where:
  fᵢ = (1/BT) Σ_{b,t} 1[expert i selected at (b,t)]
  Pᵢ = (1/BT) Σ_{b,t} p_i^{(b,t)}
```

**Curriculum weighting:**
```
λ_c(t) = 0.1 / (epoch + 1)

L_total = L_CE + λ_c · L_balance
```

The penalty is strongest in early epochs (encouraging uniform expert usage) and decays over time (allowing specialization).

---

## Training Regimen

### Stage 1: Clean Baseline Optimization (1-Speaker Warmup)

- **Data**: Single-speaker clean utterances (Tier 1)
- **Objective**: Adapt pre-trained ASR weights to the MoE architecture
- **Active parameters**: All (encoder + MoE + decoder)
- **Epochs**: 2–3

### Stage 2: Overlapped Escalation (N ≥ 2)

- **Data**: 2-speaker and 3-speaker mixtures (Tier 2 & 3)
- **Objective**: Train separation capability without disrupting linguistic knowledge
- **Active parameters**: Sidecar TCN only (encoder + MoE frozen)
- **Epochs**: 2–3

---

## Key Innovations

1. **MoE Upcycling**: Repurposes pre-trained single-speaker ASR weights into a multi-speaker separation model without training from scratch

2. **Dynamic Threshold Gating**: Adaptively activates experts based on frame complexity, enabling efficient processing of variable-density mixtures

3. **Sidecar Separator**: Residual TCN branch preserves encoder knowledge while learning interference patterns

4. **Generative Transcription**: Direct text output with temporal anchors eliminates the need for a separate diarization system

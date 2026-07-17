# OR-PiT (One-and-Rest Permutation Invariant Training) Architecture

## Overview

OR-PiT is a **speaker-count-agnostic speech separation** model that recursively extracts one speaker at a time from a mixed audio waveform. Unlike traditional PIT-based methods that require knowing the speaker count in advance and scale factorially (N!), OR-PiT makes only **N predictions** for N speakers, enabling a single trained model to handle anywhere from 2 to 6+ speakers.

---

## Acoustic Physics

### Time-Domain Waveform Processing

The OR-PiT encoder processes raw waveforms directly without STFT, using a **learned convolutional filterbank**:

```
Input Waveform [Batch, 1, Samples]
    ↓
Conv1d(kernel=16, stride=8, out_channels=256)
    ↓
Learned Embedding [Batch, 256, Frames]
```

**Key acoustic properties:**

1. **Receptive field**: The 16-sample kernel at 16 kHz covers 1 ms of audio, capturing fine-grained spectral details
2. **Temporal compression**: Stride=8 reduces 64,000 samples (4 seconds) to 8,000 frames, a 8× reduction
3. **Learned basis**: Unlike fixed STFT, the convolutional filters adapt to optimize separation performance

### Overlapping Soundwave Modeling

OR-PiT models overlapping speakers through **iterative attractor-based extraction**:

1. **Speaker queries**: Learnable embedding vectors (max_speakers=6) act as "what does speaker k look like?" prototypes
2. **Cross-attention**: Each query attends to the mixture embedding to locate its corresponding speaker
3. **Soft masking**: Attractors generate time-frequency masks that isolate individual speakers

**Physical interpretation:**
- The attractor represents the **spectral-temporal fingerprint** of a speaker
- Cross-attention finds "where in this mixture is someone matching this fingerprint?"
- The mask extracts "what belongs to this speaker" while suppressing interference

---

## Network Topology

### Full Pipeline Architecture

```
Input Waveform [B, 1, T]
    ↓
┌─────────────────────────────────────────────────────────┐
│                 MultiScaleEncoder                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Frontend Conv1d(1→256, k=16, s=8)                   │ │
│  └─────────────────────────────────────────────────────┘ │
│  ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │            BranchformerBlock(256)                   │ │
│  │  ┌────────────────┐    ┌─────────────────────────┐  │ │
│  │  │ Self-Attention │    │ Convolutional Gated MLP │  │ │
│  │  │   (4 heads)    │    │  (depthwise conv, k=3)  │  │ │
│  │  └───────┬────────┘    └───────────┬─────────────┘  │ │
│  │          └──────────────┬──────────┘                 │ │
│  │                         ↓                            │ │
│  │         Residual Fusion: x + attn + cgmlp           │ │
│  │                  LayerNorm                           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
    ↓
Mixture Embedding Y [B, 256, F]

    ↓
┌─────────────────────────────────────────────────────────┐
│           TransformerDecoderAttractor                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Temporal Pooling: AvgPool1d(k=160, s=160)           │ │
│  │ 8000 frames → 50 frames (for tractable attention)   │ │
│  └─────────────────────────────────────────────────────┘ │
│  ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Speaker Queries: nn.Parameter([max_spk=6, 256])     │ │
│  │ "What does speaker k look like?"                    │ │
│  └─────────────────────────────────────────────────────┘ │
│  ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │   TransformerDecoder (3 layers, 8 heads)            │ │
│  │   Cross-attention: queries attend to pooled mixture  │ │
│  └─────────────────────────────────────────────────────┘ │
│  ↓                                                       │
│  Attractors [B, 6, 256]    Speaker Probs [B, 6, 1]      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│              SoftMaskSeparator                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Mask Computation:                                    │ │
│  │   masks = einsum('bsc,bcf->bscf', attractors, Y)    │ │
│  │   masks = 0.1 + 0.9 * sigmoid(masks)               │ │
│  └─────────────────────────────────────────────────────┘ │
│  ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Apply Masks: masked_Y = masks * Y                   │ │
│  └─────────────────────────────────────────────────────┘ │
│  ↓                                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Decoder: ConvTranspose1d(256→1, k=16, s=8)         │ │
│  └─────────────────────────────────────────────────────┘ │
│  ↓                                                       │
│  Separated Waveforms [B, 6, T]                          │
└─────────────────────────────────────────────────────────┘
```

### Branchformer Block Detail

The Branchformer is a **dual-branch architecture** that combines self-attention (global dependencies) with convolutional gated MLP (local patterns):

**Branch 1: Self-Attention**
```
Input: [B, 256, F] → transpose → [B, F, 256]
MultiheadAttention(embed_dim=256, num_heads=4)
  Q, K, V projections → scaled dot-product attention → output projection
Output: [B, F, 256] → transpose → [B, 256, F]
```

**Branch 2: Convolutional Gated MLP (cgMLP)**
```
Input: [B, 256, F]
  ↓ LayerNorm
  ↓ Conv1d(256→512, k=1)  [expansion]
  ↓ GELU
  ↓ DepthwiseConv1d(512→512, k=3, groups=512)  [local mixing]
  ↓ Conv1d(512→256, k=1)  [projection]
Output: [B, 256, F]
```

**Fusion:**
```
out = x + attn_out + cgmlp_out
out = LayerNorm(out)
```

The residual connection preserves the original signal while both branches add complementary information.

### Attractor Mechanism

The **attractor** is the key innovation that enables speaker-count-agnostic separation:

**Speaker queries** are learnable vectors that encode "what speaker k typically looks like" in the 256-dim embedding space. These are **NOT** tied to specific identities — they're generic prototypes that adapt during training.

**Cross-attention** matches each query to its corresponding speaker in the mixture:
```
Query: speaker_query_k ∈ ℝ^256  "Find speaker k"
Key/Value: mixture_embedding ∈ ℝ^{F×256}  "Where is speaker k?"
Output: attractor_k ∈ ℝ^256  "Speaker k's representation"
```

**Halting classifier** determines which speakers are actually present:
```
speaker_prob_k = Linear(attractor_k) ∈ ℝ
# Raw logit, passed to BCEWithLogitsLoss
```

This enables the model to output exactly the right number of speakers (no more, no less).

### Soft Mask Separator

The separator applies **attractor-conditioned soft masks** to the mixture:

```python
# Compute masks via outer product
masks = torch.einsum('bsc,bcf->bscf', attractors, Y)
# Shape: [Batch, max_speakers, 256, Frames]

# Bounded sigmoid: [0.1, 1.0]
masks = 0.1 + 0.9 * torch.sigmoid(masks)

# Apply masks
masked_Y = masks * Y.unsqueeze(1)

# Decode to waveforms
separated = ConvTranspose1d(masked_Y)
```

**Interference suppression bound (0.1 floor):**
The mask never goes below 0.1, ensuring that even "inactive" speakers retain a small signal. This prevents hard zeros that can cause gradient issues and preserves information for the next extraction step.

---

## Mathematics of Loss

### Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

SI-SDR measures the quality of separated audio in a **scale-invariant** manner (performs correctly even if the model outputs are scaled differently from targets):

```
SI-SDR(est, ref) = 10 · log10( ||s_target||² / ||e_noise||² )

where:
  est = estimated signal
  ref = reference (ground truth) signal
  
  # Mean-center both signals
  est = est - mean(est)
  ref = ref - mean(ref)
  
  # Project est onto ref to find optimal scaling
  α = (est · ref) / (ref · ref)
  
  s_target = α · ref        # scaled reference (clean component)
  e_noise = est - s_target  # error (noise + interference)
```

**Energy clamping (numerical stability):**
```
ref_energy = (ref · ref).clamp(min=1e-10)
target_energy = ||s_target||².clamp(min=1e-10)
noise_energy = ||e_noise||².clamp(min=1e-10)
```

This prevents NaN gradients when reference signals are near-silent (e.g., padded targets).

### Permutation Invariant Training (PIT) Loss

PIT solves the **label assignment problem**: given N estimated sources and M true sources, which permutation matches them best?

**Standard PIT (N! permutations):**
```
L_PIT = min_{π ∈ S_N} (1/N) Σᵢ L(est_i, ref_{π(i)})

where S_N is the symmetric group on N elements.
```

For 6 speakers: 6! = 720 permutations to evaluate.

**OR-PiT (N linear assignments):**

Instead of trying all permutations, OR-PiT uses the **Hungarian algorithm** (linear sum assignment) to find the optimal one-to-one matching:

```python
# Build cost matrix: SI-SDR between all (est, ref) pairs
cost_matrix = si_sdr(est[expand], ref[expand])  # [N, M]

# Hungarian algorithm finds optimal assignment in O(N³)
row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

# Sum SI-SDR for matched pairs
loss = -Σ_{r,c matched} si_sdr(est[r], ref[c])
```

**Why this works:**
- The Hungarian algorithm finds the **globally optimal** one-to-one matching
- Gradients flow only through the matched pairs (detached cost matrix for solver)
- Complexity: O(N³) instead of O(N!) — tractable for any speaker count

### Halting Loss

The halting classifier is trained to predict **which speakers are actually present**:

```python
# Target: one-hot for matched speakers
target_probs[b, matched_row, 0] = 1.0
target_probs[b, unmatched_row, 0] = 0.0

# Binary cross-entropy (with logits for AMP stability)
halting_loss = BCEWithLogitsLoss(speaker_probs, target_probs)
```

**Total loss:**
```
L_total = L_PIT + L_halting
```

---

## Temporal Downsampling for Efficiency

A critical optimization in the attractor module:

```python
# Before: cross-attention on 8000 frames
# Memory: O(B × max_speakers × 8000 × 256) ≈ 12 GB for batch=4

# After: average pooling with stride 160
self.temporal_pool = nn.AvgPool1d(kernel_size=160, stride=160)
Y_pooled = self.temporal_pool(Y)  # [B, 256, 50]

# Memory: O(B × max_speakers × 50 × 256) ≈ 0.3 GB for batch=4
```

This 160× reduction makes the model tractable on CPU and smaller GPUs without sacrificing separation quality (the attractor aggregates information across time anyway).

---

## Key Innovations

1. **Speaker-count-agnostic**: Single model handles 2–6+ speakers without modification

2. **Efficient PIT**: O(N³) Hungarian algorithm instead of O(N!) permutation search

3. **Attractor-based extraction**: Learnable speaker prototypes that generalize to unseen voices

4. **Branchformer encoder**: Combines global self-attention with local convolutions for rich representations

5. **Temporal pooling**: Makes cross-attention tractable on limited hardware

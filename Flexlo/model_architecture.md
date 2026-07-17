# FlexIO-Lite Architecture (Recursive OR-PIT)

## Overview

FlexIO-Lite is a **recursive speech separation** model that implements the **One-and-Rest PIT (OR-PIT)** training paradigm with a **FiLM-conditioned Temporal Convolutional Network (TCN)** backbone. Unlike the fixed-output OR-PiT model, FlexIO-Lite operates through **iterative extraction**: at each step, it separates one speaker from the residual, then feeds the residual back as the new mixture.

This recursive design enables a **single trained model** to handle an **arbitrary number of speakers** with constant memory footprint — the model only ever predicts two things per step (one target, one residual), regardless of how many speakers actually overlap.

---

## Acoustic Physics

### Time-Domain Waveform Processing

FlexIO-Lite uses a **learned encoder-decoder filterbank** similar to Conv-TasNet:

```
Input Waveform [B, 1, T]
    ↓
Encoder: Conv1d(1→256, kernel=16, stride=8)
    ↓
Mixture Embedding H [B, 256, T']
    ↓
Separator Core (TCN + FiLM)
    ↓
Target Emb H_t, Residual Emb H_r
    ↓
Decoder: ConvTranspose1d(256→1, kernel=16, stride=8)
    ↓
Target Waveform, Residual Waveform
```

**Key acoustic properties:**

1. **Learned basis functions**: The 256 encoder channels form an adaptive basis that optimizes for separation rather than reconstruction (unlike fixed STFT)

2. **Stride=8 compression**: Reduces temporal resolution by 8×, making the TCN computationally tractable while preserving enough detail for speaker discrimination

3. **Perfect reconstruction**: The encoder-decoder pair is designed such that `decode(encode(x)) ≈ x` for clean signals, ensuring the bottleneck doesn't destroy information

### Overlapping Soundwave Modeling

The recursive extraction process models overlapping speakers as a **binary tree decomposition**:

```
Mixture (N speakers)
    ↓ separate_step()
├── Target Speaker 1
└── Residual (N-1 speakers)
        ↓ separate_step()
    ├── Target Speaker 2
    └── Residual (N-2 speakers)
            ↓ separate_step()
        ...
```

**Physical interpretation:**
- Each step isolates "one voice from the chorus"
- The residual contains everything else (other speakers + any remaining interference)
- The stopping classifier decides: "Is there still someone left in the residual?"

---

## Network Topology

### Full Recursive Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    FlexIOLiteRecursiveSeparator                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      Shared Encoder                         │ │
│  │  Conv1d(1→256, k=16, s=8) + ReLU                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                SeparatorCore (TCN + FiLM)                  │ │
│  │                                                             │ │
│  │  Input Norm (GroupNorm) → Bottleneck Conv1d(256→512)      │ │
│  │       ↓                                                     │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │ TCN Block × 16 (2 stacks × 8 blocks)                │   │ │
│  │  │                                                      │   │ │
│  │  │  Dilated Depthwise Separable Conv:                  │   │ │
│  │  │    dilation = 2^i, i ∈ {0,1,2,3,4,5,6,7}           │   │ │
│  │  │    receptive_field = Σ 2^i · kernel_size            │   │ │
│  │  │                   ≈ 2^8 = 256 frames in embedding   │   │ │
│  │  │                   ≈ 2048 samples in waveform (128ms)│   │ │
│  │  │                                                      │   │ │
│  │  │  FiLM Conditioning:                                 │   │ │
│  │  │    γ, β = Linear(prompt)                            │   │ │
│  │  │    out = x * (1 + γ) + β                            │   │ │
│  │  │                                                      │   │ │
│  │  │  Residual: out = x + TCN(x)                         │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │       ↓                                                     │ │
│  │  Mask Output: Conv1d(512→512) → sigmoid                   │ │
│  │  Split into: mask_target, mask_residual                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Apply Masks:                                               │ │
│  │    target_emb = H ⊙ mask_target                            │ │
│  │    residual_emb = H ⊙ mask_residual                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      Shared Decoder                         │ │
│  │  ConvTranspose1d(256→1, k=16, s=8)                         │ │
│  │  Pad/crop to exact input length                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  target_wav, residual_wav                                       │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  StoppingClassifier                         │ │
│  │  GlobalAvgPool(residual_emb) → MLP(256→128→128→1)         │ │
│  │  Output: stop_logit ∈ ℝ                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### FiLM Conditioning Mechanism

**Feature-wise Linear Modulation (FiLM)** allows a prompt vector to dynamically rescale and shift the TCN features:

```python
class FiLM(nn.Module):
    def __init__(self, prompt_dim, feature_dim):
        self.to_gamma_beta = nn.Linear(prompt_dim, feature_dim * 2)
    
    def forward(self, x, prompt):
        # x: [B, C, T], prompt: [B, prompt_dim]
        gamma_beta = self.to_gamma_beta(prompt)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # [B, C, 1]
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta
```

**Interpretation:**
- **γ (gamma)**: "How much should I amplify/suppress this feature?"
- **β (beta)**: "What offset should I add to shift the decision boundary?"

The generic prompt (`self.generic_prompt`) learns a "give me any speaker" conditioning that works for all voices.

### TCN Block with Dilated Convolutions

The **receptive field** of the TCN determines how much temporal context the model sees:

```
Stack 1:
  Block 0: dilation=1   → receptive field = 3 frames
  Block 1: dilation=2   → receptive field = 3 + 2×2 = 7 frames
  Block 2: dilation=4   → receptive field = 7 + 2×4 = 15 frames
  Block 3: dilation=8   → receptive field = 15 + 2×8 = 31 frames
  Block 4: dilation=16  → receptive field = 31 + 2×16 = 63 frames
  Block 5: dilation=32  → receptive field = 63 + 2×32 = 127 frames
  Block 6: dilation=64  → receptive field = 127 + 2×64 = 255 frames
  Block 7: dilation=128 → receptive field = 255 + 2×128 = 511 frames

Stack 2 (repeat): total receptive field ≈ 1023 frames

In samples: 1023 frames × 8 (stride) ≈ 8184 samples ≈ 511 ms @ 16 kHz
```

This 500+ ms context is sufficient to capture syllable-level patterns for speaker discrimination.

### Stopping Classifier

The stopping classifier reads the **residual embedding** after separation to predict whether a real speaker remains:

```python
class StoppingClassifier(nn.Module):
    def forward(self, residual_emb):
        # residual_emb: [B, 256, T']
        pooled = residual_emb.mean(dim=-1)  # [B, 256] global average pool
        logit = self.mlp(pooled)  # [B, 1]
        return logit.squeeze(-1)  # [B]
```

**Why post-separation (not pre-separation)?**
After separation, the residual embedding contains cleaner information about "what's left." If the model successfully extracted a speaker, the residual should clearly show whether another remains. Pre-separation classification is harder because it must infer from the messy mixture.

---

## Mathematics of Loss

### SI-SNR (Scale-Invariant SNR)

FlexIO-Lite uses the same SI-SDR formulation as OR-PiT, but with a slightly different notation:

```python
def si_snr(estimate, target):
    # Mean-center
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    
    # Project estimate onto target
    s_target = (estimate · target) / ||target||² × target
    
    # Noise = estimate - projection
    e_noise = estimate - s_target
    
    # Ratio in dB
    return 10 × log10(||s_target||² / ||e_noise||²)
```

**SI-SNR improvement (SI-SNRi):**
```
SI-SNRi = SI-SNR(separated, target) - SI-SNR(mixture, target)
```

This measures how much better the separated output is compared to just returning the mixture unchanged.

### OR-PIT Step Loss

At each recursion step, the model predicts a **target speaker** and a **residual**. The loss finds the best matching ground-truth speaker:

```python
def or_pit_step_loss(target_pred, residual_pred, remaining_sources):
    """
    remaining_sources: list of ground-truth speakers still in the mixture
    """
    best_loss = inf
    best_idx = 0
    
    for i, candidate in enumerate(remaining_sources):
        # If we match this speaker, what should the residual be?
        if len(remaining_sources) > 1:
            candidate_residual = sum(remaining_sources[j] for j ≠ i)
        else:
            candidate_residual = zeros  # last speaker
        
        # Combined loss
        loss_i = si_snr_loss(target_pred, candidate) + \
                 si_snr_loss(residual_pred, candidate_residual)
        
        if loss_i < best_loss:
            best_loss = loss_i
            best_idx = i
    
    return best_loss, best_idx
```

**Why is this O(N) instead of O(N!)?**
- Standard PIT tries all N! permutations of N outputs against N targets
- OR-PIT only tries N candidates: "which of the N remaining speakers is the target this step?"
- The residual is implicitly determined by the target choice

### Stopping Loss

Binary cross-entropy for the stopping classifier:

```python
stop_labels = [True, True, ..., True, False]  # N-1 True, 1 False
# True = "keep recursing, there's still someone left"
# False = "stop, the residual is empty/noise"

stop_loss = BCEWithLogitsLoss(stop_logit, stop_labels)
```

### Total Step Loss

```python
step_loss = separation_loss + stop_loss_weight × stop_loss
# stop_loss_weight typically 0.5
```

---

## Recursive Training Loop

The training process runs the recursive loop **per-step backward** to keep memory constant:

```python
for k in range(max_depth):  # max_depth = max speakers in batch
    # Forward pass for this step only
    target_pred, residual_pred, stop_logit = model.separate_step(current_mix)
    
    # Find best matching speaker
    sep_loss, chosen_idx = or_pit_step_loss(target_pred, residual_pred, remaining)
    
    # Stopping classifier loss
    stop_loss = BCEWithLogitsLoss(stop_logit, recursion_labels[k])
    
    # Backward + step immediately (frees computation graph)
    (sep_loss + stop_loss_weight × stop_loss).backward()
    optimizer.step()
    
    # Update for next step
    del remaining[chosen_idx]  # remove matched speaker
    current_mix = residual_pred.detach()  # residual becomes new mixture
```

**Memory efficiency:**
- Peak memory is constant regardless of speaker count (only one step's graph exists at a time)
- Each step's gradients are computed and discarded before the next step

---

## Key Innovations

1. **Recursive extraction**: Single model handles arbitrary speaker counts with constant memory

2. **FiLM conditioning**: Prompt vector modulates all TCN blocks, enabling flexible control

3. **Per-step backward**: Memory-efficient training that scales to any speaker count

4. **OR-PIT loss**: O(N) instead of O(N!) assignment problem

5. **Stopping classifier**: Automatically determines when to stop recursion

---

## Differences from OR-PiT

| Feature | OR-PiT | FlexIO-Lite |
|---------|--------|-------------|
| Output | Fixed N speakers at once | Recursive one-at-a-time |
| Architecture | Branchformer + Transformer decoder | TCN + FiLM |
| Memory | Scales with speaker count | Constant (per-step backward) |
| Inference | Single forward pass | Recursive loop |
| Stopping | Attractor halting classifier | Post-separation classifier |
| Conditioning | None (blind) | Optional speaker embedding |

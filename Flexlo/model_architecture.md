# FlexIO Recursive Architecture (LSTM + FiLM + Dual-Prompt)

## Overview

FlexIO Recursive is an **iterative speech separation and target extraction** model. It implements the **One-and-Rest PIT (OR-PIT)** training paradigm within a prompt-conditioned **Bi-directional LSTM sequence model** backbone. Rather than extracting all $N$ speakers simultaneously, the model operates recursively: at each step, it isolates one target speaker and reconstructs a residual mixture containing the remaining speakers. This residual is fed back as the input mixture for the next step.

By utilizing this recursive loop, the model can scale to an **arbitrary, unknown number of speakers** at inference time while maintaining a constant peak memory footprint—predicting only one target and one residual per step.

---

## Acoustic Physics

### Time-Domain Waveform Processing

The model uses a learned encoder-decoder filterbank operating in the time domain:

```
Input Waveform [B, 1, T]
    ↓
Encoder: Conv1d(1→256, kernel=16, stride=8)
    ↓
Mixture Embedding H [B, 256, T_latent]
    ↓
Separator Core (Bi-LSTM Stack + FiLM Fusion)
    ↓
Target Emb H_t, Residual Emb H_r
    ↓
Decoder (Target): ConvTranspose1d(256→1, k=16, s=8)  ──> Target Waveform [B, 1, T]
Decoder (Residual): ConvTranspose1d(256→1, k=16, s=8) ──> Residual Waveform [B, 1, T]
```

**Key acoustic properties:**
1. **Learned basis functions:** The 256 encoder filters act as an adaptive, data-driven time-frequency representation, capturing fine acoustic characteristics optimized for separation.
2. **Stride=8 compression:** Compresses temporal resolution by 8×, reducing the frame rate to make recurrent sequence modeling computationally efficient while retaining temporal resolution.
3. **Dual-Decoder Reconstruction:** Separate decoders for the target speaker and residual ensure the acoustic properties of the residual mixture (which contains the remaining overlapping speakers) are preserved for the next recursion step.

### Overlapping Soundwave Decomposition

The recursive separation process models overlapping audio as a **binary tree decomposition**:

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
- Each step isolates "one voice from the chorus."
- The residual signal represents the acoustic sum of all remaining active sound sources plus ambient noise.
- The stopping classifier decides: "Does the residual embedding still contain a real active speaker?"

---

## Network Topology

### Full Recursive Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     FlexIORecursiveModel                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      Shared Encoder                         │ │
│  │  Conv1d(1→256, k=16, s=8, p=8) + ReLU                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Prompt Resolution                       │ │
│  │  If target mode: use speaker embedding p ∈ ℝ^(B×256)      │ │
│  │  If blind mode: expand self.generic_prompt ∈ ℝ^(1×256)      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │             FiLM Prompt Conditioning (Fusion)              │ │
│  │  γ, β = Linear(prompt)                                      │ │
│  │  H_cond = H * γ + β                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Separator Core (Stacked Bi-LSTM)              │ │
│  │  Block 1: GroupNorm → Bi-LSTM → Linear → Add Residual      │ │
│  │  Block 2: GroupNorm → Bi-LSTM → Linear → Add Residual      │ │
│  │  Block 3: GroupNorm → Bi-LSTM → Linear → Add Residual      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Mask Generation                           │ │
│  │  Conv1d(256→512) → Sigmoid                                 │ │
│  │  Split into: mask_target [B, 256, T_l], mask_res [B, 256, T_l]│ │
│  │  Apply: h_target = h_mix ⊙ mask_t, h_res = h_mix ⊙ mask_r  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Dual-Decoders                           │ │
│  │  Target: ConvTranspose1d(256→1, k=16, s=8, p=8)             │ │
│  │  Residual: ConvTranspose1d(256→1, k=16, s=8, p=8)           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Stopping Head Classifier                  │ │
│  │  GlobalAvgPool(h_residual) → MLP(256→128→ReLU→Linear(1))   │ │
│  │  Output: stop_logit ∈ ℝ                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### FiLM Conditioning Mechanism

**Feature-wise Linear Modulation (FiLM)** allows a prompt vector to dynamically rescale and shift the latent mixture features:

```python
class FiLMBlock(nn.Module):
    def __init__(self, prompt_dim, num_features):
        super().__init__()
        self.gamma_project = nn.Linear(prompt_dim, num_features)
        self.beta_project = nn.Linear(prompt_dim, num_features)
        
    def forward(self, x, prompt):
        # x: [B, C, T], prompt: [B, prompt_dim]
        gamma = self.gamma_project(prompt).unsqueeze(-1)  # [B, C, 1]
        beta = self.beta_project(prompt).unsqueeze(-1)    # [B, C, 1]
        return gamma * x + beta
```

**Interpretation:**
- **$\gamma$ (gamma):** Modulates channel-wise gains, selectively amplifying or suppressing specific acoustic basis functions.
- **$\beta$ (beta):** Applies channel-wise bias offsets, shifting feature thresholds.
- **Generic Prompt (`self.generic_prompt`):** A trainable vector of shape `[1, 256]` that learns the average acoustic signature needed to isolate "any" target speaker when no target profile is specified.

### Recurrent Separator Block

The separator core consists of stacked `SeparatorBlock` modules modeled with bidirectional recurrence and residual skip connections:

```python
class SeparatorBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_channels * 2, in_channels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        
    def forward(self, x):
        # x: [B, C, T]
        residual = x
        x_lstm = x.transpose(1, 2)  # [B, T, C] for LSTM
        x_out, _ = self.lstm(x_lstm)
        x_proj = self.proj(x_out).transpose(1, 2)  # [B, C, T]
        out = F.relu(self.norm(x_proj) + residual)
        return out
```

By leveraging bidirectional recurrence, the separator block tracks long-term context both forwards and backwards, which helps the network align speech features under high overlap.

### Post-Separation Stopping Classifier

The stopping classifier evaluates the **residual latent embedding** after separation has already occurred:

```python
class StoppingHead(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, h_residual):
        # h_residual: [B, 256, T_latent]
        pooled = torch.mean(h_residual, dim=-1)  # [B, 256] Global Avg Pool
        logit = self.mlp(pooled)  # [B, 1]
        return logit
```

**Why post-separation (instead of pre-separation)?**
Classifying the residual after extracting a speaker is significantly cleaner. If the target speaker was successfully extracted, the residual contains only the remaining speakers. This cleaner signal makes detecting speaker presence easier than trying to predict speaker counts from the original mixture.

---

## Mathematics of Loss

### SI-SNR (Scale-Invariant Signal-to-Noise Ratio)

The loss function uses mean-centered Scale-Invariant SNR to measure separation accuracy:

$$\mathbf{s}_{target} = \mathbf{s} - \text{mean}(\mathbf{s})$$
$$\hat{\mathbf{s}}_{estimate} = \hat{\mathbf{s}} - \text{mean}(\hat{\mathbf{s}})$$

We project the estimate onto the target vector:
$$\mathbf{e}_{target} = \frac{\langle \hat{\mathbf{s}}_{estimate}, \mathbf{s}_{target} \rangle}{\|\mathbf{s}_{target}\|^2} \mathbf{s}_{target}$$
$$\mathbf{e}_{noise} = \hat{\mathbf{s}}_{estimate} - \mathbf{e}_{target}$$

$$\text{SI-SNR} = 10 \log_{10} \left( \frac{\|\mathbf{e}_{target}\|^2}{\|\mathbf{e}_{noise}\|^2 + \epsilon} \right)$$

### OR-PIT Step Loss

At recursion step $k$, the model predicts a target $\hat{\mathbf{s}}_{target}$ and a residual $\hat{\mathbf{s}}_{residual}$. The OR-PIT loss matches the target output against one of the remaining active ground-truth sources, forcing the remaining sum of sources into the residual output:

$$\mathcal{L}_{sep} = \min_{i \in \mathcal{I}_{rem}} \left[ -\text{SI-SNR}(\hat{\mathbf{s}}_{target}, \mathbf{s}_i) - \text{SI-SNR}\left(\hat{\mathbf{s}}_{residual}, \sum_{j \in \mathcal{I}_{rem}, j \neq i} \mathbf{s}_j\right) \right]$$

where $\mathcal{I}_{rem}$ is the set of indices of active ground-truth speakers remaining in this step.

### Stopping Loss

The stopping classifier is trained using Binary Cross Entropy (BCE) with logits:

$$\mathcal{L}_{stop} = \text{BCEWithLogitsLoss}(l_{stop}, y_{stop})$$

where $y_{stop} \in \{0, 1\}$ represents speaker presence ($1 = \text{speakers remain}$, $0 = \text{no speakers remain}$).

### Total Step Loss

$$\mathcal{L}_{total} = \mathcal{L}_{sep} + \alpha \cdot \mathcal{L}_{stop} \quad (\text{where } \alpha = 0.5)$$

---

## Recursive Training Loop

Training runs recursively, calculating loss and executing backpropagation **per-step** to maintain constant peak memory usage:

```python
for k in range(max_depth):  # max_depth = speaker count in mixture
    # 1. Forward pass for current step only
    target_pred, residual_pred, stop_logit = model(current_mix, prompt)
    
    # 2. Compute OR-PIT separation loss
    sep_loss, chosen_idx = or_pit_step_loss(target_pred, residual_pred, remaining_gt_sources)
    
    # 3. Compute stopping classifier loss
    stop_label = torch.tensor([1.0 if len(remaining_gt_sources) > 1 else 0.0])
    stop_loss = bce_loss(stop_logit, stop_label)
    
    # 4. Backward pass & step optimizer (clears current computation graph)
    total_loss = sep_loss + 0.5 * stop_loss
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # 5. Prepare variables for next step
    del remaining_gt_sources[chosen_idx]  # remove matched speaker
    current_mix = residual_pred.detach()  # detach residual to act as new input mix
```

---

## Key Innovations

1. **Recursive Extraction:** A single model processes an arbitrary number of speakers with a constant peak memory footprint.
2. **Dual-Conditioning Prompt Routing:** Dynamically shifts between blind separation (using a trainable generic prompt parameter) and target extraction (using ECAPA speaker embeddings) via a shared prompt fusion slot.
3. **Per-Step Backward Pass:** Memory-efficient backpropagation that enables deep recursion scaling without memory overflow.
4. **Post-Separation Halting Classifier:** Evaluates residual latent embeddings after extraction, providing high-precision speaker counting and automatic termination.

---

## Model Differences

| Feature | Classic OR-PiT | FlexIO Recursive |
| :--- | :--- | :--- |
| **Output Scheme** | Fixed $N$ speakers simultaneously | Recursive one-at-a-time loop |
| **Sequence Model** | Branchformer + Transformer decoder | Bi-directional LSTM Stack |
| **Peak Memory Scale** | Scales with speaker count ($O(N)$) | Constant peak memory ($O(1)$) |
| **Conditioning** | Blind separation only | Dual-mode (Blind vs. Target Embedding) |
| **Halting Mechanism** | Attractor halting vector | Post-separation latent classifier |

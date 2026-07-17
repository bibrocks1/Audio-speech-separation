# Task List — Audio Separation Pipeline Mock-Run Audit

## Overview

This task list covers the import audit, bug fixes, and mock-run verification for the 1-epoch training loop.

---

## Task 1: Import Audit

- [x] 1.1 Verify `src/train.py` imports — all use `from src.X import ...` absolute style, correct for `python -m src.train` invocation from project root.
- [x] 1.2 Verify `src/model.py` imports — all use `from src.X import ...` absolute style, no relative import issues.
- [x] 1.3 Confirm `DynamicMixDataset` returns `target_waveforms` as time-domain waveforms (`[num_speakers, Time]`), not spectral masks — **confirmed correct**, SI-SDR loss requirement satisfied.
- [x] 1.4 Confirm `collate_custom_mix` returns `(mixed_audios [B,1,T], target_waveforms [B,Spk,T], configs)` — **confirmed correct**.
- [x] 1.5 Verify encoder/separator symmetry: Conv1d(k=16, s=8, p=4) → ConvTranspose1d(k=16, s=8, p=4) — both 32000→4000→32000 and 64000→8000→64000 are lossless round-trips — **confirmed**.

---

## Task 2: Bug Fixes Applied

- [x] 2.1 **`src/attractor.py` — Cross-attention sequence length (performance/correctness bug)**
  - **Root cause**: The `TransformerDecoderAttractor` fed the raw encoder output of 8000 frames as the cross-attention memory, making each forward pass O(6 × 8000) across 3 decoder layers — this caused a >5 minute CPU stall for batch_size=4.
  - **Fix**: Added `nn.AvgPool1d(kernel_size=160, stride=160)` temporal downsampling before the decoder, reducing memory length from 8000 → 50 frames. The `pool_stride` is a constructor parameter (default 160) for easy adjustment.
  - **File**: `d:\Audio_Separation\src\attractor.py`

- [x] 2.2 **`src/loss.py` — Infeasible cost matrix in `linear_sum_assignment` (correctness bug)**
  - **Root cause**: When padded-zero target waveforms (from the fallback synthetic path) are used as references, `si_sdr` produces NaN/Inf values (division by zero in reference energy). `scipy.optimize.linear_sum_assignment` raises `ValueError: cost matrix is infeasible` on non-finite inputs.
  - **Fix**: Applied `torch.nan_to_num(sdr, nan=-100.0, posinf=100.0, neginf=-100.0)` to the SDR matrix before passing to the solver.
  - **File**: `d:\Audio_Separation\src\loss.py`

- [x] 2.3 **`src/loss.py` — BCE input range safety (numerical stability bug)**
  - **Root cause**: `nn.BCELoss` raises `RuntimeError: all elements of input should be between 0 and 1` if `speaker_probs` hits exact 0.0 or 1.0 due to floating-point saturation.
  - **Fix**: Clamped `speaker_probs` to `[1e-6, 1 − 1e-6]` before the BCE call.
  - **File**: `d:\Audio_Separation\src\loss.py`

- [x] 2.4 **`src/train.py` — Mock-run resource sizing (performance)**
  - **Root cause**: `max_length_sec=4.0` (64000 samples) with `batch_size=4` was unnecessarily heavy for a mock verification pass.
  - **Fix**: Changed to `max_length_sec=2.0` (32000 samples) and `batch_size=2` for the mock run. These parameters are trivially adjustable for full training.
  - **File**: `d:\Audio_Separation\src\train.py`

---

## Task 3: Mock Execution Verification

- [x] 3.1 Run `python -m src.train` — completes in <60s on CPU, prints epoch loss.
- [x] 3.2 Verify gradient flow — all 75 parameters have non-zero `.grad` tensors after one backward pass.

---

## Walkthrough Artifact

```
PS D:\Audio_Separation> & "d:\Audio_Separation\.venv\Scripts\python.exe" -m src.train
Using device: cpu
Epoch 1 Average Loss: 29.5470
Training mock execution completed successfully.
```

**Gradient Flow Check:**
```
Parameters with valid gradients: 75
Parameters with zero gradients:  0
Parameters with no gradient:     0

Sample gradient norms:
  encoder.frontend.weight:             grad_norm=26.392467
  encoder.frontend.bias:               grad_norm=5.236300
  encoder.body.attn.in_proj_weight:    grad_norm=7.010362
  encoder.body.attn.in_proj_bias:      grad_norm=2.789743
  encoder.body.attn.out_proj.weight:   grad_norm=8.458201
  encoder.body.attn.out_proj.bias:     grad_norm=5.034201
```

All 75 trainable parameters (encoder, attractor, separator) receive non-zero gradients, confirming correct end-to-end backpropagation through the full `Encoder → Attractor → Separator → PITLoss` path.

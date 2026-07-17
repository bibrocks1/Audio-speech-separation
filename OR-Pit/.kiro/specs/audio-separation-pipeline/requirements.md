# Requirements Document

## Introduction

This document captures the requirements for the Audio Separation Pipeline — a parallelized, soft-masking speech separation system built on a dynamic curriculum dataloader and a Transformer-Decoder Attractor (TDA) network. The pipeline trains on synthetic on-the-fly mixtures of LibriSpeech speakers and DEMAND noise, progressively increasing mixture complexity through three curriculum stages, and separates an unknown number of speakers from a single-channel mixture waveform.

## Glossary

- **Pipeline**: The complete end-to-end audio separation system from waveform input to separated waveform outputs.
- **Encoder**: The `MultiScaleEncoder` module that converts a raw waveform into a dense feature representation using a convolutional front-end and a `BranchformerBlock`.
- **Attractor**: The `TransformerDecoderAttractor` module that produces one embedding vector per estimated speaker and a binary probability score indicating whether each speaker slot is active.
- **Separator**: The `SoftMaskSeparator` module that applies soft multiplicative masks derived from attractor embeddings to the encoded features, then decodes back to time-domain waveforms.
- **SeparationNetwork**: The top-level `nn.Module` (`src/model.py`) that composes `Encoder → Attractor → Separator`.
- **CurriculumManager**: The scheduler (`src/curriculum.py`) that varies the number of speakers, SNR, reverb, and gap parameters across training stages based on the current epoch.
- **DynamicMixDataset**: The `torch.utils.data.Dataset` (`src/dynamic_mixer.py`) that constructs synthetic mixture + target pairs on-the-fly according to a `CurriculumManager` configuration.
- **PITLoss**: The Permutation-Invariant Training loss (`src/loss.py`) that solves the optimal speaker assignment via the Hungarian algorithm and combines SI-SDR and BCE losses.
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio — the primary waveform quality metric used inside `PITLoss`.
- **Mixture**: A single-channel waveform produced by summing speaker signals and optional noise.
- **Target_Waveforms**: A set of clean, time-domain single-channel waveforms — one per active speaker — used as ground-truth references during training.
- **Max_Speakers**: The upper bound on simultaneous speakers the `Attractor` can represent (default: 6).
- **Stage**: One of three curriculum phases (1, 2, 3) that governs mixing complexity.

---

## Requirements

### Requirement 1: Waveform Encoding

**User Story:** As a researcher, I want the encoder to convert raw waveforms into compact multi-scale features, so that downstream modules can separate speakers in a lower-dimensional space.

#### Acceptance Criteria

1. THE `Encoder` SHALL accept a waveform tensor of shape `[Batch, 1, Time]` where `Time` is any positive integer.
2. WHEN a waveform of length `Time` is provided, THE `Encoder` SHALL produce a feature tensor of shape `[Batch, 256, ceil((Time + 2·4 - 16) / 8 + 1)]` using a strided convolutional front-end with kernel 16, stride 8, and padding 4.
3. THE `Encoder` SHALL apply one `BranchformerBlock` to the front-end output, consisting of a self-attention branch and a convolutional gated MLP branch fused with a residual connection and final `LayerNorm`.
4. THE `Encoder` SHALL produce outputs where no element is NaN or Inf for any random normal input tensor.

### Requirement 2: Speaker Attractor Estimation

**User Story:** As a researcher, I want the attractor module to estimate how many speakers are present and produce per-speaker embeddings, so that the separator can generate the correct number of output streams.

#### Acceptance Criteria

1. THE `Attractor` SHALL accept an encoded feature tensor of shape `[Batch, 256, Frames]`.
2. WHEN given encoded features, THE `Attractor` SHALL produce an attractor tensor of shape `[Batch, Max_Speakers, 256]` and a `speaker_probs` tensor of shape `[Batch, Max_Speakers, 1]`.
3. THE `Attractor` SHALL produce `speaker_probs` values in the open interval `(0, 1)` via a `Sigmoid` activation.
4. THE `Attractor` SHALL apply temporal average pooling with a stride of 160 over the `Frames` dimension before cross-attention to keep the memory sequence tractable.
5. WHEN cross-attention is applied, THE `Attractor` SHALL use a `TransformerDecoder` with 3 layers, 8 attention heads, and an `embed_dim` of 256.

### Requirement 3: Soft-Mask Separation

**User Story:** As a researcher, I want the separator to produce one time-domain waveform per speaker slot using differentiable soft masks, so that gradients can flow back through the entire network.

#### Acceptance Criteria

1. THE `Separator` SHALL accept an encoded feature tensor `Y` of shape `[Batch, 256, Frames]` and an attractor tensor of shape `[Batch, Max_Speakers, 256]`.
2. WHEN computing masks, THE `Separator` SHALL compute scores via `einsum('bsc,bcf->bscf', attractors, Y)` and apply the boundary `mask = 0.1 + 0.9 · sigmoid(score)` to ensure no speaker contribution is entirely suppressed.
3. THE `Separator` SHALL decode masked features to time-domain waveforms using a `ConvTranspose1d` with kernel 16, stride 8, in-channels 256, out-channels 1, and padding 4.
4. WHEN given an input waveform of length `Time = 32000`, THE `Separator` SHALL produce output waveforms of the same length `32000`, ensuring the encoder/decoder pair is length-symmetric.
5. THE `Separator` SHALL produce an output tensor of shape `[Batch, Max_Speakers, Time]`.

### Requirement 4: Permutation-Invariant Training Loss

**User Story:** As a researcher, I want the loss function to be permutation-invariant, so that the network is not penalized for producing correct sources in a different order than the references.

#### Acceptance Criteria

1. THE `PITLoss` SHALL accept `est_sources` of shape `[Batch, Max_Speakers, Time]`, `ref_sources` of shape `[Batch, True_Speakers, Time]`, and `speaker_probs` of shape `[Batch, Max_Speakers, 1]`.
2. WHEN computing the SI-SDR cost matrix, THE `PITLoss` SHALL replace any NaN or Inf values with a finite fallback of −100 dB before passing the matrix to the Hungarian assignment solver, so the solver always receives a feasible input.
3. THE `PITLoss` SHALL use `scipy.optimize.linear_sum_assignment` with `maximize=True` to find the optimal permutation that maximises total SI-SDR across active speaker slots.
4. THE `PITLoss` SHALL compute a binary cross-entropy halting loss between `speaker_probs` (clamped to `[1e-6, 1 − 1e-6]` to prevent numerical instability) and a target probability tensor where assigned slots are 1 and unassigned slots are 0.
5. THE `PITLoss` SHALL return a single scalar equal to `(−SI-SDR_sum / (Batch × True_Speakers)) + BCE_halting_loss`.
6. FOR ALL finite random input pairs, THE `PITLoss` SHALL return a finite scalar value (no NaN, no Inf).

### Requirement 5: Dynamic Curriculum Mixing

**User Story:** As a researcher, I want the dataset to generate progressively harder mixtures as training advances, so that the model learns easy patterns first and generalises to complex cocktail-party conditions later.

#### Acceptance Criteria

1. THE `CurriculumManager` SHALL define three stages based on epoch thresholds `stage2_epoch` and `stage3_epoch`.
2. WHILE the current epoch is less than `stage2_epoch`, THE `CurriculumManager` SHALL produce configurations with exactly 2 speakers, no noise, no reverb, and an overlap ratio uniformly sampled from `[0.0, 0.2]`.
3. WHILE the current epoch is between `stage2_epoch` and `stage3_epoch`, THE `CurriculumManager` SHALL produce configurations with 2 or 3 speakers, noise at SNR uniformly sampled from `[−5, 5]` dB, no reverb, and an overlap ratio from `[0.4, 1.0]`.
4. WHILE the current epoch is at least `stage3_epoch`, THE `CurriculumManager` SHALL produce configurations with 4 or 5 speakers, noise at SNR from `[−5, 5]` dB, reverb enabled, and an overlap ratio from `[0.5, 1.0]`.
5. THE `CurriculumManager` SHALL sample conversational gap durations from a Gamma distribution with shape 2 and a scale derived from the overlap ratio, returning `num_speakers − 1` non-negative gap values.

### Requirement 6: On-the-Fly Mixture Construction

**User Story:** As a researcher, I want the dataset to construct each mixture in memory at fetch time, so that I can train on effectively unlimited augmented data without pre-generating a fixed mixture library.

#### Acceptance Criteria

1. THE `DynamicMixDataset` SHALL load speech and noise audio files from JSON index files, resample to 16 kHz, and convert to mono.
2. WHEN the speech index contains fewer files than the configured `num_speakers`, THE `DynamicMixDataset` SHALL fall back to synthetic Gaussian noise tensors so training can proceed without a full dataset.
3. THE `DynamicMixDataset` SHALL align speaker signals by applying the cumulative gap offsets as leading zero-padding, then truncate or zero-pad all signals to a fixed `max_length_samples` window.
4. WHEN `use_reverb` is enabled, THE `DynamicMixDataset` SHALL convolve each speaker signal with a 300 ms synthetic exponential-decay RIR using `torchaudio.functional.fftconvolve`, trimming the output back to `max_length_samples`.
5. WHEN `use_noise` is enabled, THE `DynamicMixDataset` SHALL scale noise to match the configured SNR in dB relative to the summed speech power, using the formula `scale = sqrt((P_speech / P_noise) × 10^(−SNR/10))`.
6. THE `DynamicMixDataset` SHALL return each item as a dict with keys `mixed_audio` (shape `[Time]`), `target_waveforms` (shape `[num_speakers, Time]`), and `config`.
7. THE `collate_custom_mix` function SHALL zero-pad `target_waveforms` along the speaker dimension to match the maximum speaker count in the batch, and return `(mixed_audios, target_waveforms, configs)` with shapes `[Batch, 1, Time]` and `[Batch, Max_Spk_In_Batch, Time]` respectively.

### Requirement 7: End-to-End Training Loop

**User Story:** As a researcher, I want a single entry-point training script that ties together the dataloader, model, loss, and optimizer, so that I can launch training with `python -m src.train` from the project root.

#### Acceptance Criteria

1. THE `train` module SHALL be invokable as `python -m src.train` from the project root `d:\Audio_Separation` using the `.venv` interpreter without modifying `sys.path` at runtime.
2. THE `train` module SHALL use absolute package-relative imports (`from src.X import ...`) for all internal modules.
3. WHEN `data/speech_index.json` or `data/noise_index.json` do not exist, THE `train` module SHALL create them as empty JSON arrays so the dataloader can fall back to synthetic data.
4. THE `train` module SHALL use an `AdamW` optimizer with a learning rate of `1e-4` and clip gradients at `max_norm=5.0` before each parameter update.
5. WHEN executing a 1-epoch mock run with `dataset.length = 4` and `batch_size ≤ 2`, THE `train` module SHALL complete without error and print the average epoch loss to stdout.
6. AFTER one complete forward-backward pass, ALL 75 trainable parameters of the `SeparationNetwork` SHALL have non-zero gradient tensors, confirming end-to-end gradient flow.

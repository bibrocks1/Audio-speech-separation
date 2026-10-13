# Audio-speech-separation (ASR & Source Separation Repository)

Welcome to the central repository for state-of-the-art audio speech separation and automatic speech recognition (ASR) pipelines. This repository contains three distinct, cutting-edge machine learning model architectures designed for solving overlapping speech mixtures, acoustic reverberation, and diarization.

---

## 🚀 The Three Model Architectures

This repository is organized into three separate pipelines, each addressing distinct separation and transcription challenges:

### 1. [UME (Unified Mixture of Experts)](./ume/)
* **Primary Task**: Joint Speech Separation & ASR (transcription + timestamps).
* **Key Innovation**: Bypasses traditional audio-to-audio separation by directly generating text transcripts interleaved with grounded temporal speaker time anchors.
* **Architectural Features**: 
  * Sparse **Mixture-of-Experts (MoE)** upcycled from dense OWSM v3.1 weights.
  * **Dynamic Threshold Gating** for speaker-adaptive expert routing.
  * **Sortformer** with Arrival Time Sorting (ATS) and Sinusoidal Speaker Kernels (bypassing expensive PIT permutation overhead).
  * **CALM Bias Encoder** merging acoustic-linguistic log probabilities with the generative TagSpeech LLM backend.
* **Status**: Fully optimized with target causal self-attention masks and in-memory RAM caching for fast training.

### 2. [Flexlo (Recursive Lite Separation)](./Flexlo/)
* **Primary Task**: Raw audio-to-audio waveform separation.
* **Key Innovation**: A lightweight, recursive loop separation engine featuring a built-in stopping classifier to handle variable speaker mixtures dynamically.
* **Architectural Features**:
  * Residual Temporal Convolutional Network (TCN).
  * Feature-wise Linear Modulation (**FiLM**) conditioning.
  * Adaptive recurrent looping until speaker channels are clean.

### 3. [OR-Pit (Branchformer & Attractor Separation)](./OR-Pit/)
* **Primary Task**: Speaker-count-agnostic audio-to-audio waveform separation (2–5 speakers, noisy, single channel).
* **Key Innovation**: One forward pass produces six speaker slots plus a halting classifier per slot, so a single model separates every voice and estimates how many speakers are present.
* **Architectural Features**:
  * Conv1d learned filterbank + **Branchformer** encoder (self-attention in parallel with a gated convolutional MLP).
  * **Transformer-decoder attractors**: six learned speaker queries attend to a pooled mixture embedding.
  * Attractor-conditioned soft masks refined by a **FiLM-conditioned TCN**, decoded with ConvTranspose1d.
  * **Hungarian permutation-invariant training** on SI-SDR (O(N³) instead of O(N!)) with a DETR-style halting-aware matching cost.
* **Results** (single GTX 1650, 4 GB; SI-SDRi): **+4.33 dB** on 1,000 held-out noisy 2–5-speaker mixtures (+2.49 dB for the original spec); 3.11 dB on Libri2Mix and 2.69 dB on Libri3Mix (noisy test sets); a cumulative ablation and a Conv-TasNet baseline trained with the same data and GPU time. Details in the [OR-Pit README](./OR-Pit/README.md).
* **Status**: Trained and benchmarked end to end; speaker counting (36 % exact) is the main open problem.

---

## 📊 Feature Comparison Matrix

| Feature | UME (ASR + Separation) | Flexlo (Audio Separation) | OR-Pit (Audio Separation) |
|---|---|---|---|
| **Output Type** | Text Transcript + Time Anchors | Clean Audio Waveform | Clean Audio Waveform |
| **Speaker Diarization** | Built-in (interleaved timestamps) | None | None |
| **Separation Strategy** | MoE routing + Sortformer ATS | Recursive loop | Branchformer + Attractors |
| **Pre-trained Weights** | Yes (OWSM v3.1 Dense) | No | No |
| **Expert Specialization** | Dynamic Threshold Gating | Static conditioning | Attractor modulation |
| **Target Application** | Meeting/Conversation transcription | Light, fast device separation | High-fidelity speaker isolation |

---

## 📂 Repository Structure

```
Audio-speech-separation/
├── README.md               # Central Hub / Homepage (this file)
├── .gitignore              # Standard git exclusion patterns
│
├── ume/                    # Unified Mixture of Experts Pipeline
│   ├── preprocess_pipeline.py  # Unified 2-stage GPU training driver
│   ├── ume_architecture.py     # Underlying MoE, Sortformer, CALM modules
│   ├── infer_sample.py         # Autoregressive greedy inference/listening script
│   └── model_architecture.md   # Architectural documentation
│
├── Flexlo/                 # Recursive Separation Pipeline
│   ├── train.py                # Training loops
│   ├── dataset.py              # Dynamic mixer
│   └── flexio_lite.py          # TCN + FiLM model definition
│
└── OR-Pit/                 # Branchformer Attractor Pipeline
    ├── src/
    │   ├── model.py            # Encoder-separator-decoder definition
    │   └── attractor.py        # Speaker attractor generator
    └── README.md               # PIT details
```

---

## 🛠️ Quick Start

To train or run inference on any of the models, navigate to their respective subdirectories and follow the setup instructions in their READMEs:

* **For Text Transcription & Grounded Time-Anchors**: See [UME README](./ume/README.md).
* **For Recursive TCN Waveform Separation**: See [Flexlo README](./Flexlo/README.md).
* **For Branchformer Attractor Separation**: See [OR-Pit README](./OR-Pit/README.md).

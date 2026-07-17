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
* **Primary Task**: Raw audio-to-audio waveform separation.
* **Key Innovation**: Utilizes high-capacity Branchformer layers coupled with dynamic speaker attractor vectors to track, separate, and isolate multiple overlapping speakers.
* **Architectural Features**:
  * Parallel self-attention and depth-wise convolution (Branchformer block).
  * **Speaker Attractor Generator** for tracking spatial speaker locations.
  * Permutation Invariant Training (PIT) matching loss.

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

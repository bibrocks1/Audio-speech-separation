import os
import random
import torch
import torchaudio
import soundfile as sf
import numpy as np

TARGET_SR = 16000

def load_wav(path, base_dir):
    full_path = os.path.join(base_dir, path)
    try:
        data, sr = sf.read(full_path)
        wav = torch.tensor(data, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
    except Exception as e:
        wav, sr = torchaudio.load(full_path)
        
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

def sample_utterance(source_index, split, exclude_speakers=None):
    if exclude_speakers is None:
        exclude_speakers = set()
        
    # Filter by split and exclude speakers
    eligible = [s for s in source_index if s["split"] == split and s["speaker_id"] not in exclude_speakers]
    
    if not eligible:
        # Fallback if we run out of speakers
        eligible = [s for s in source_index if s["split"] == split]
        
    choice = random.choice(eligible)
    return choice

def normalize_loudness_rms(wav, target_rms_db):
    # Calculate RMS
    rms = torch.sqrt(torch.mean(wav ** 2) + 1e-8)
    # Convert target dB to linear RMS scale
    target_rms = 10 ** (target_rms_db / 20)
    # Scale signal
    scaled_wav = wav * (target_rms / rms)
    return scaled_wav

def apply_overlap_staggering(sources, overlap_ratio, chunk_len_samples):
    # Stagger sources based on overlap_ratio
    # First source starts at t = 0
    staggered_sources = []
    start_times = []
    
    # We crop each source to fit inside the chunk_len_samples (e.g. 4 seconds)
    # If the source is shorter, we keep it as-is
    cropped_sources = []
    for src in sources:
        if src.shape[-1] > chunk_len_samples:
            # Random crop
            start = random.randint(0, src.shape[-1] - chunk_len_samples)
            cropped_sources.append(src[:, start:start + chunk_len_samples])
        else:
            cropped_sources.append(src)
            
    staggered_sources.append(cropped_sources[0])
    start_times.append(0)
    
    current_end = cropped_sources[0].shape[-1]
    
    for i in range(1, len(cropped_sources)):
        src = cropped_sources[i]
        src_len = src.shape[-1]
        
        if overlap_ratio == 1.0:
            # Full overlap: start at 0
            t_start = 0
        elif overlap_ratio == 0.0:
            # Sequential: start at current end
            t_start = current_end
        else:
            # Partial overlap: start time depends on overlap ratio and previous signal end
            # We want the start time to be somewhere between 0 and current_end
            # Lower overlap -> larger start time -> closer to current_end
            # Higher overlap -> smaller start time -> closer to 0
            max_start = max(1, current_end - int(src_len * 0.1)) # leave at least 10% overlap
            t_start = int(current_end * (1.0 - overlap_ratio) * random.uniform(0.9, 1.1))
            t_start = max(0, min(max_start, t_start))
            
        start_times.append(t_start)
        current_end = max(current_end, t_start + src_len)
        
    # Now build the aligned padded sources
    aligned_sources = []
    for src, t_start in zip(cropped_sources, start_times):
        # Pad at start and end to match the total mixture length
        pad_left = t_start
        pad_right = max(0, current_end - (t_start + src.shape[-1]))
        padded = torch.nn.functional.pad(src, (pad_left, pad_right))
        aligned_sources.append(padded)
        
    # Crop the final mixture components if the staggered output is longer than chunk_len_samples
    # In speech separation, we usually crop the final mixture to chunk_len_samples for batching
    final_aligned = []
    if current_end > chunk_len_samples:
        # We take a window of size chunk_len_samples.
        # To make sure we capture active speech, let's randomly crop but ensure we cover
        # the start of the mixture or some active speaker regions.
        crop_start = random.randint(0, current_end - chunk_len_samples)
        for s in aligned_sources:
            final_aligned.append(s[:, crop_start:crop_start + chunk_len_samples])
    else:
        # Pad everyone to chunk_len_samples
        for s in aligned_sources:
            pad_right = chunk_len_samples - s.shape[-1]
            final_aligned.append(torch.nn.functional.pad(s, (0, pad_right)))
            
    return final_aligned

def convolve(signal, rir):
    # signal: [1, L], rir: [1, L_rir]
    sig_len = signal.shape[-1]
    rir_len = rir.shape[-1]
    
    # Pad signal at the end to allow for RIR tail
    padded_signal = torch.nn.functional.pad(signal, (0, rir_len - 1))
    
    # Run 1D convolution
    convolved = torch.nn.functional.conv1d(
        padded_signal.unsqueeze(0),
        rir.unsqueeze(0),
        groups=1
    ).squeeze(0)
    
    # Truncate back to the original signal length to keep shape consistent
    return convolved[:, :sig_len]

def add_noise_at_snr(mixture, noise, snr_db):
    # Crop or pad noise to match mixture length
    mix_len = mixture.shape[-1]
    noise_len = noise.shape[-1]
    
    if noise_len >= mix_len:
        # Random crop
        start = random.randint(0, noise_len - mix_len)
        noise_cropped = noise[:, start:start + mix_len]
    else:
        # Tile/repeat noise
        repeats = (mix_len // noise_len) + 1
        noise_cropped = noise.repeat(1, repeats)[:, :mix_len]
        
    # Calculate powers
    mix_power = torch.mean(mixture ** 2) + 1e-8
    noise_power = torch.mean(noise_cropped ** 2) + 1e-8
    
    # Calculate scaled noise power based on target SNR
    target_noise_power = mix_power / (10 ** (snr_db / 10))
    scale_factor = torch.sqrt(target_noise_power / noise_power)
    
    scaled_noise = noise_cropped * scale_factor
    return mixture + scaled_noise, scaled_noise

def generate_mixture(source_index, noise_files, rir_files, n_speakers, overlap_ratio, 
                     use_noise=True, use_reverb=True, chunk_len_sec=4.0, sr=16000, base_dir="", split="train-clean-100"):
    # 1. Sample N speakers and load their utterances
    chosen_speakers = set()
    sources = []
    
    # Sample speech
    for _ in range(n_speakers):
        meta = sample_utterance(source_index, split=split, exclude_speakers=chosen_speakers)
        chosen_speakers.add(meta["speaker_id"])
        wav = load_wav(os.path.join("cache", meta["file_path"]), base_dir)
        sources.append(wav)
        
    # 2. Loudness normalization (RMS between -33 and -25 dB)
    normalized_sources = []
    for src in sources:
        target_db = random.uniform(-33.0, -25.0)
        normalized_sources.append(normalize_loudness_rms(src, target_db))
        
    # 3. Apply overlap staggering
    chunk_len_samples = int(chunk_len_sec * sr)
    aligned_sources = apply_overlap_staggering(normalized_sources, overlap_ratio, chunk_len_samples)
    
    # 4. Optional RIR convolution (each speaker gets a separate RIR to simulate spatial separation)
    if use_reverb and rir_files:
        convolved_sources = []
        for src in aligned_sources:
            rir_path = random.choice(rir_files)
            rir = load_wav(rir_path, base_dir)
            
            # Normalize RIR energy to sum of squares = 1
            rir_energy = torch.sum(rir ** 2)
            if rir_energy > 0:
                rir = rir / torch.sqrt(rir_energy)
                
            convolved_sources.append(convolve(src, rir))
        aligned_sources = convolved_sources
        
    # 5. Sum sources -> raw mixture
    mixture = sum(aligned_sources)
    
    # 6. Optional noise addition
    noise_added = torch.zeros_like(mixture)
    if use_noise and noise_files:
        noise_path = random.choice(noise_files)
        noise = load_wav(noise_path, base_dir)
        snr_db = random.uniform(10.0, 25.0)
        mixture, noise_added = add_noise_at_snr(mixture, noise, snr_db)
        
    # 7. Peak normalize final mixture to -1 dBFS (amplitude ~0.89) to prevent clipping
    # Save the scale factor so we scale the sources and noise by the exact same amount
    peak = torch.max(torch.abs(mixture))
    scale_factor = 1.0
    if peak > 0:
        scale_factor = 0.89 / peak
        mixture = mixture * scale_factor
        aligned_sources = [src * scale_factor for src in aligned_sources]
        noise_added = noise_added * scale_factor
        
    # 8. Build recursion-depth ground-truth labels
    # recursion_labels[k] represents: "does the residual still contain a speaker after k extractions?"
    # If N speakers:
    # After 0 extractions: N speakers left (True)
    # After 1 extraction: N-1 speakers left (True if N-1 >= 1)
    # ...
    # After N-1 extractions: 1 speaker left (True)
    # After N extractions: 0 speakers left (False)
    # We represent the labels for each step of the recursion loop.
    # A loop of length N will be run.
    # At step 1, we extract speaker 1, residual has N-1 speakers. Label = True
    # At step N-1, we extract speaker N-1, residual has 1 speaker. Label = True
    # At step N, we extract speaker N, residual has 0 speakers. Label = False
    # So the list of labels of length N is: [True] * (N - 1) + [False]
    recursion_labels = [True] * (n_speakers - 1) + [False]
    
    return mixture, aligned_sources, noise_added, recursion_labels

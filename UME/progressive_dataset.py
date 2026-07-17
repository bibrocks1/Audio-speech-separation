import os
import urllib.request
import tarfile
import numpy as np
import scipy.io.wavfile as wav
import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE = 16000
DURATION = 3.0

def get_human_speech_files(dataset_dir="/kaggle/input"):
    """Finds audio files in mounted datasets or downloads Mini-LibriSpeech fallback."""
    audio_files = []
    if os.path.exists(dataset_dir):
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.flac', '.mp3')):
                    audio_files.append(os.path.join(root, file))
                    if len(audio_files) >= 200:
                        break
            if len(audio_files) >= 200:
                break
                
    if not audio_files:
        print("Downloading Mini-LibriSpeech fallback...")
        os.makedirs("fallback_speech", exist_ok=True)
        url = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
        tar_path = "mini_librispeech.tar.gz"
        try:
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path="fallback_speech")
            for root, dirs, files in os.walk("fallback_speech"):
                for file in files:
                    if file.lower().endswith(('.wav', '.flac')):
                        audio_files.append(os.path.join(root, file))
        except Exception as e:
            print(f"Error getting fallback files: {e}")
            
    return audio_files

def load_and_standardize_audio(filepath, target_duration=3.0, sr=16000):
    """Loads, converts to mono, resamples, and slices/pads audio to target duration."""
    try:
        waveform, orig_sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if orig_sr != sr:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=sr)
            waveform = resampler(waveform)
            
        waveform = waveform.squeeze(0)
        target_len = int(target_duration * sr)
        if len(waveform) < target_len:
            padding = target_len - len(waveform)
            waveform = torch.cat([waveform, torch.zeros(padding)])
        else:
            start = np.random.randint(0, len(waveform) - target_len + 1)
            waveform = waveform[start:start + target_len]
            
        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        return waveform.numpy()
    except Exception as e:
        print(f"Error standardizing {filepath}: {e}")
        return np.zeros(int(target_duration * sr))

def generate_synthetic_speech(pitch=150.0, duration=3.0, sr=16000):
    """Generates synthetic speech fallback wave if no human files available."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    fm = 4.0 * np.sin(2 * np.pi * 3.0 * t)
    wave = np.zeros_like(t)
    for h in range(1, 4):
        wave += (1.0 / h) * np.sin(2 * np.pi * (pitch * h) * t + fm * h)
    wave = wave * 0.5 * (1.0 + np.sin(2 * np.pi * 0.8 * t))
    return wave / (np.max(np.abs(wave)) + 1e-8)

def fast_rir_simulate(decay=0.08, delay_samples=10, sr=16000):
    """Simulates far-field RIR with direct path, early reflections and exponential decay."""
    total_len = int(sr * 0.4)
    rir = np.zeros(total_len)
    rir[delay_samples] = 1.0  # Direct path
    # Early reflections
    for d in [30, 70]:
        rir[d] = 0.25 * np.random.randn()
    # Late decay tail
    t = np.linspace(0, 0.4, total_len, endpoint=False)
    tail = np.random.randn(total_len) * np.exp(-t / decay)
    rir += 0.1 * tail
    return rir / np.sqrt(np.sum(rir**2))

def add_chime4_noise(clean_sig, snr_db=5.0, sr=16000):
    """Adds non-stationary CHiME-4 modulated noise at target SNR."""
    noise = np.random.randn(len(clean_sig))
    t = np.linspace(0, len(clean_sig)/sr, len(clean_sig), endpoint=False)
    modulation = 0.6 * (1.0 + np.sin(2 * np.pi * 2.0 * t)) + 0.15 * np.cos(2 * np.pi * 10.0 * t)
    noise = noise * modulation
    
    clean_power = np.mean(clean_sig**2)
    noise_power = np.mean(noise**2)
    factor = np.sqrt(clean_power / (noise_power * (10**(snr_db / 10.0))))
    return clean_sig + factor * noise

def generate_curriculum_datasets(num_samples=100, output_dir="curriculum_dataset"):
    """Generates Phase 1 dataset tiers representing 1, 2, and 3-speaker mixtures."""
    print("Ingesting speech data...")
    speech_files = get_human_speech_files()
    
    # Target directories
    os.makedirs(output_dir, exist_ok=True)
    tiers = ["tier1_clean", "tier2_overlap", "tier3_dense"]
    for t in tiers:
        os.makedirs(os.path.join(output_dir, t), exist_ok=True)
        os.makedirs(os.path.join(output_dir, t, "mix"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, t, "clean"), exist_ok=True)
        
    print("Generating curriculum tiers...")
    for idx in range(num_samples):
        # Speaker pools
        sig_pool = []
        if len(speech_files) >= 3:
            chosen = np.random.choice(speech_files, size=3, replace=False)
            sig_pool = [load_and_standardize_audio(f) for f in chosen]
        else:
            sig_pool = [
                generate_synthetic_speech(pitch=120.0),
                generate_synthetic_speech(pitch=180.0),
                generate_synthetic_speech(pitch=240.0)
            ]
            
        # 1. Tier 1 (1-Speaker Clean Baseline)
        clean1 = sig_pool[0]
        # Convolve RIR & add modulated noise
        rir = fast_rir_simulate(decay=np.random.uniform(0.05, 0.1))
        reverb1 = np.convolve(clean1, rir, mode='same')
        noisy_reverb1 = add_chime4_noise(reverb1, snr_db=np.random.uniform(5.0, 15.0))
        noisy_reverb1 /= (np.max(np.abs(noisy_reverb1)) + 1e-8)
        
        wav.write(os.path.join(output_dir, "tier1_clean/mix", f"sample_{idx:04d}.wav"), SAMPLE_RATE, (noisy_reverb1 * 32767).astype(np.int16))
        wav.write(os.path.join(output_dir, "tier1_clean/clean", f"sample_{idx:04d}.wav"), SAMPLE_RATE, (clean1 * 32767).astype(np.int16))
        
        # 2. Tier 2 (2-Speaker Overlapped Mixture)
        # Curriculum: overlap capped at 10% for first half, progressive for the second half
        overlap_ratio = np.random.uniform(0.01, 0.10) if idx < num_samples // 2 else np.random.uniform(0.10, 0.90)
        
        # Mix Speaker A & Speaker B with offset
        sig_a, sig_b = sig_pool[0], sig_pool[1]
        shift = int((1.0 - overlap_ratio) * len(sig_a))
        sig_b_shifted = np.zeros_like(sig_a)
        if shift < len(sig_a):
            sig_b_shifted[shift:] = sig_b[:-shift] if shift > 0 else sig_b
            
        clean2 = (sig_a + sig_b_shifted) / 2.0
        reverb2 = np.convolve(clean2, rir, mode='same')
        noisy_reverb2 = add_chime4_noise(reverb2, snr_db=np.random.uniform(0.0, 12.0))
        noisy_reverb2 /= (np.max(np.abs(noisy_reverb2)) + 1e-8)
        
        wav.write(os.path.join(output_dir, "tier2_overlap/mix", f"sample_{idx:04d}.wav"), SAMPLE_RATE, (noisy_reverb2 * 32767).astype(np.int16))
        wav.write(os.path.join(output_dir, "tier2_overlap/clean", f"sample_{idx:04d}.wav"), SAMPLE_RATE, (clean2 * 32767).astype(np.int16))
        
        # 3. Tier 3 (3-Speaker Dense Mixture)
        sig_c = sig_pool[2]
        shift_c = int(np.random.uniform(0.1, 0.5) * len(sig_a))
        sig_c_shifted = np.zeros_like(sig_a)
        sig_c_shifted[shift_c:] = sig_c[:-shift_c]
        
        clean3 = (sig_a + sig_b_shifted + sig_c_shifted) / 3.0
        reverb3 = np.convolve(clean3, rir, mode='same')
        noisy_reverb3 = add_chime4_noise(reverb3, snr_db=np.random.uniform(-3.0, 8.0))
        noisy_reverb3 /= (np.max(np.abs(noisy_reverb3)) + 1e-8)
        
        wav.write(os.path.join(output_dir, "tier3_dense/mix", f"sample_{idx:04d}.wav"), SAMPLE_RATE, (noisy_reverb3 * 32767).astype(np.int16))
        wav.write(os.path.join(output_dir, "tier3_dense/clean", f"sample_{idx:04d}.wav"), SAMPLE_RATE, (clean3 * 32767).astype(np.int16))
        
    print("Dataset curriculum generation completed.")

import os
import sys
import tarfile
import zipfile
import json
import glob
import shutil
import torch
import torchaudio
import soundfile as sf

TARGET_SR = 16000

def load_audio(file_path):
    try:
        data, samplerate = sf.read(file_path)
        tensor = torch.tensor(data, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.transpose(0, 1)
        return tensor, samplerate
    except Exception as e:
        return torchaudio.load(file_path)

def save_audio(file_path, tensor, sample_rate):
    try:
        data = tensor.squeeze(0).cpu().numpy()
        sf.write(file_path, data, sample_rate)
    except Exception as e:
        torchaudio.save(file_path, tensor, sample_rate)


def extract_archive(archive_path, extract_dir):
    filename = os.path.basename(archive_path)
    print(f"Extracting {filename} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    if filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    elif filename.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        print(f"Unsupported archive format: {filename}")
        sys.exit(1)
    print(f"Finished extracting {filename}")

def trim_silence(wav, sample_rate, threshold_db=-45, frame_length_ms=20, hop_length_ms=10):
    # Check if signal is empty/silent
    peak = torch.max(torch.abs(wav))
    if peak == 0:
        return wav
        
    frame_length = int(sample_rate * frame_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)
    
    # Absolute values
    abs_wav = torch.abs(wav[0])
    
    # Check if short waveform
    if abs_wav.shape[0] <= frame_length:
        return wav
        
    # Unfold into overlapping frames
    frames = abs_wav.unfold(0, frame_length, hop_length)
    frame_max = frames.max(dim=1)[0]
    
    # Convert to dB relative to peak amplitude
    frame_max_db = 20 * torch.log10(frame_max / peak + 1e-8)
    
    # Find frames above threshold
    active_frames = torch.where(frame_max_db > threshold_db)[0]
    if len(active_frames) == 0:
        return wav
        
    start_frame = active_frames[0].item()
    end_frame = active_frames[-1].item()
    
    # Convert back to sample indices
    start_sample = start_frame * hop_length
    end_sample = min((end_frame * hop_length) + frame_length, wav.shape[1])
    
    # Add a small cushion of 100ms at start/end
    cushion = int(sample_rate * 0.1)
    start_sample = max(0, start_sample - cushion)
    end_sample = min(wav.shape[1], end_sample + cushion)
    
    return wav[:, start_sample:end_sample]

def process_speech_files(clean_dir, cache_clean_dir, split_name):
    print(f"Processing speech files for split: {split_name}...")
    source_index = []
    speaker_ids = set()
    
    # Find all flac files in the directory
    flac_pattern = os.path.join(clean_dir, "LibriSpeech", split_name, "**", "*.flac")
    flac_files = glob.glob(flac_pattern, recursive=True)
    
    if not flac_files:
        # Check alternative nested path if LibriSpeech structure changes
        flac_pattern = os.path.join(clean_dir, "**", "*.flac")
        flac_files = glob.glob(flac_pattern, recursive=True)
        
    if not flac_files:
        print(f"Warning: No FLAC files found in {clean_dir} for split {split_name}")
        return source_index, speaker_ids
        
    total_files = len(flac_files)
    print(f"Found {total_files} FLAC files.")
    
    for i, file_path in enumerate(flac_files):
        # Extract speaker_id
        # Path format: .../split/speaker_id/chapter_id/speaker-chapter-utterance.flac
        parts = os.path.normpath(file_path).split(os.sep)
        # Find split_name index and get next element as speaker_id
        try:
            split_idx = parts.index(split_name)
            speaker_id = parts[split_idx + 1]
        except (ValueError, IndexError):
            # Fallback to parent directory names
            speaker_id = parts[-3]
            
        speaker_ids.add(speaker_id)
        
        # Load and resample
        wav, sr = load_audio(file_path)
        try:
            os.remove(file_path)
        except Exception:
            pass
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            
        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # DC offset removal
        wav = wav - wav.mean()
        
        # Trim silence
        wav = trim_silence(wav, TARGET_SR)
        
        # Skip if trimmed file is extremely short
        duration = wav.shape[-1] / TARGET_SR
        if duration < 0.5:
            continue
            
        # Save preprocessed file
        out_speaker_dir = os.path.join(cache_clean_dir, split_name, speaker_id)
        os.makedirs(out_speaker_dir, exist_ok=True)
        out_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".wav"
        out_file_path = os.path.join(out_speaker_dir, out_file_name)
        
        save_audio(out_file_path, wav, TARGET_SR)
        
        # Record metadata
        source_index.append({
            "file_path": os.path.relpath(out_file_path, os.path.dirname(cache_clean_dir)),
            "speaker_id": speaker_id,
            "duration": duration,
            "split": split_name
        })
        
        if (i + 1) % 500 == 0 or i == total_files - 1:
            print(f"Processed {i + 1}/{total_files} files...")
            
    print(f"Finished processing split {split_name}. Speakers found: {len(speaker_ids)}")
    return source_index, speaker_ids

def process_noise_and_rir(rirs_noises_dir, cache_dir):
    print("Processing SLR28 Noises and RIRs...")
    noise_bank_dir = os.path.join(cache_dir, "noise_bank")
    rir_bank_dir = os.path.join(cache_dir, "rir_bank")
    os.makedirs(noise_bank_dir, exist_ok=True)
    os.makedirs(rir_bank_dir, exist_ok=True)
    
    # 1. Process point-source noises
    noise_pattern = os.path.join(rirs_noises_dir, "**", "pointsource_noises", "**", "*.wav")
    noise_files = glob.glob(noise_pattern, recursive=True)
    
    # Fallback to search any noises inside
    if not noise_files:
        noise_pattern = os.path.join(rirs_noises_dir, "**", "noises", "**", "*.wav")
        noise_files = glob.glob(noise_pattern, recursive=True)
        
    print(f"Found {len(noise_files)} noise files.")
    for i, file_path in enumerate(noise_files):
        wav, sr = load_audio(file_path)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Peak normalize noise before caching
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak
            
        out_path = os.path.join(noise_bank_dir, f"noise_{i:04d}.wav")
        save_audio(out_path, wav, TARGET_SR)
        
    # 2. Process room impulse responses (RIRs)
    rir_pattern = os.path.join(rirs_noises_dir, "**", "*_rirs", "**", "*.wav")
    rir_files = glob.glob(rir_pattern, recursive=True)
    
    # Fallback
    if not rir_files:
        rir_pattern = os.path.join(rirs_noises_dir, "**", "simulated_rirs", "**", "*.wav")
        rir_files = glob.glob(rir_pattern, recursive=True)
        
    print(f"Found {len(rir_files)} RIR files.")
    for i, file_path in enumerate(rir_files):
        wav, sr = load_audio(file_path)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            
        # Keep channel 0 for mono baseline
        if wav.shape[0] > 1:
            wav = wav[0:1]
            
        # Normalize RIR energy (sum of squares = 1) to preserve convolved signal energy
        energy = torch.sum(wav ** 2)
        if energy > 0:
            wav = wav / torch.sqrt(energy)
            
        out_path = os.path.join(rir_bank_dir, f"rir_{i:04d}.wav")
        save_audio(out_path, wav, TARGET_SR)
        
    print("Noises and RIRs processed successfully!")

def main():
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.getcwd()
    raw_dir = os.path.join(base_dir, "raw")
    data_dir = os.path.join(base_dir, "data")
    cache_dir = os.path.join(base_dir, "cache")
    metadata_dir = os.path.join(base_dir, "metadata")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # 1. Check raw archives
    dev_archive = os.path.join(raw_dir, "dev-clean.tar.gz")
    test_archive = os.path.join(raw_dir, "test-clean.tar.gz")
    train_archive = os.path.join(raw_dir, "train-clean-100.tar.gz")
    rirs_archive = os.path.join(raw_dir, "rirs_noises.zip")
    
    missing = [f for f in [dev_archive, test_archive, train_archive, rirs_archive] if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing raw files: {missing}. Please run download_data.py first.")
        sys.exit(1)
        
    # 2. Extract and Process split-by-split to conserve disk space
    clean_dir = os.path.join(data_dir, "clean")
    rirs_noises_dir = os.path.join(data_dir, "rirs_noises")
    cache_clean_dir = os.path.join(cache_dir, "clean_sources")
    os.makedirs(cache_clean_dir, exist_ok=True)
    
    source_index = []
    splits = {}
    
    # Define splits mapping to their archives
    speech_splits = [
        ("dev-clean", dev_archive),
        ("test-clean", test_archive),
        ("train-clean-100", train_archive)
    ]
    
    for split_name, archive_path in speech_splits:
        split_extracted_path = os.path.join(clean_dir, "LibriSpeech", split_name)
        
        # Extract if not already processed/cached (or folder missing)
        if not os.path.exists(split_extracted_path):
            extract_archive(archive_path, clean_dir)
            
        # Delete raw archive immediately after extraction to save space!
        if os.path.exists(archive_path):
            print(f"Removing raw archive immediately after extraction to save space: {archive_path}")
            os.remove(archive_path)
            
        # Process files
        idx, speaker_ids = process_speech_files(clean_dir, cache_clean_dir, split_name)
        source_index.extend(idx)
        splits[split_name] = list(speaker_ids)
            
        # Delete extracted files to save space
        if os.path.exists(split_extracted_path):
            print(f"Cleaning up extracted folder to save space: {split_extracted_path}")
            shutil.rmtree(split_extracted_path)
            
    # Clean up empty parent directories under clean_dir
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)
        
    # Write metadata index
    with open(os.path.join(metadata_dir, "source_index.json"), "w") as f:
        json.dump(source_index, f, indent=4)
        
    # Write speaker splits
    with open(os.path.join(metadata_dir, "speaker_splits.json"), "w") as f:
        json.dump(splits, f, indent=4)
        
    # Process RIRs and Noises
    rir_extracted_path = os.path.join(rirs_noises_dir, "RIRS_NOISES")
    if not os.path.exists(rir_extracted_path):
        extract_archive(rirs_archive, rirs_noises_dir)
        
    # Delete RIR/Noise raw archive immediately after extraction to save space!
    if os.path.exists(rirs_archive):
        print(f"Removing raw RIR/Noise archive immediately after extraction to save space: {rirs_archive}")
        os.remove(rirs_archive)
        
    process_noise_and_rir(rirs_noises_dir, cache_dir)
    
    # Delete RIR/Noise extracted folder
    if os.path.exists(rirs_noises_dir):
        print(f"Cleaning up extracted RIR/Noise folder to save space: {rirs_noises_dir}")
        shutil.rmtree(rirs_noises_dir)
        
    # Clean up parent data_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    print("\nDataset preparation finished successfully!")

if __name__ == "__main__":
    main()

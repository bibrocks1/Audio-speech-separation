import os
import sys
import random
import json
import torch
import torchaudio
import soundfile as sf

# Append workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dynamic_mixer import generate_mixture, TARGET_SR

def save_audio(file_path, tensor, sample_rate):
    try:
        data = tensor.squeeze(0).cpu().numpy()
        sf.write(file_path, data, sample_rate)
    except Exception as e:
        torchaudio.save(file_path, tensor, sample_rate)


def build_split_eval_set(metadata_dir, cache_dir, split_name, num_mixes_per_n=100, chunk_len_sec=4.0):
    print(f"\nBuilding static evaluation set for split: {split_name}...")
    
    # Base workspace directory
    workspace_dir = os.path.dirname(metadata_dir)
    eval_base_dir = os.path.join(metadata_dir, "eval_sets", split_name)
    os.makedirs(eval_base_dir, exist_ok=True)
    
    # Load source index
    index_path = os.path.join(metadata_dir, "source_index.json")
    with open(index_path, "r") as f:
        all_sources = json.load(f)
        
    # Filter for this split
    sources = [s for s in all_sources if s["split"] == split_name]
    
    # Get noises and RIRs
    noise_dir = os.path.join(cache_dir, "noise_bank")
    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = [os.path.join("cache", "noise_bank", f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
        
    rir_dir = os.path.join(cache_dir, "rir_bank")
    rir_files = []
    if os.path.exists(rir_dir):
        rir_files = [os.path.join("cache", "rir_bank", f) for f in os.listdir(rir_dir) if f.endswith(".wav")]
        
    # Fix the seed for reproducibility!
    random.seed(1337)
    torch.manual_seed(1337)
    
    # Generate for N = 2, 3, 4, 5 speakers
    for n_speakers in [2, 3, 4, 5]:
        mix_dir = os.path.join(eval_base_dir, f"{n_speakers}mix")
        os.makedirs(mix_dir, exist_ok=True)
        
        print(f"Generating {num_mixes_per_n} mixtures for {n_speakers} speakers...")
        
        for i in range(num_mixes_per_n):
            # Sample overlap ratio uniformly across the full range
            overlap_ratio = random.uniform(0.0, 1.0)
            
            # Alternate applying reverb/noise to get a diverse evaluation split
            use_reverb = (i % 2 == 0) and len(rir_files) > 0
            use_noise = (i % 3 != 0) and len(noise_files) > 0  # 66% noisy, 33% clean
            
            mixture, aligned_sources, noise_added, recursion_labels = generate_mixture(
                source_index=sources,
                noise_files=noise_files,
                rir_files=rir_files,
                n_speakers=n_speakers,
                overlap_ratio=overlap_ratio,
                use_noise=use_noise,
                use_reverb=use_reverb,
                chunk_len_sec=chunk_len_sec,
                sr=TARGET_SR,
                base_dir=workspace_dir,
                split=split_name
            )
            
            # Save files
            mix_id = f"mix_{i:04d}"
            
            # Save mixture
            mix_path = os.path.join(mix_dir, f"{mix_id}_mixture.wav")
            save_audio(mix_path, mixture, TARGET_SR)
            
            # Save individual clean convolved sources
            for s_idx, src_wav in enumerate(aligned_sources):
                src_path = os.path.join(mix_dir, f"{mix_id}_s{s_idx + 1}.wav")
                save_audio(src_path, src_wav, TARGET_SR)
                
            # Save noise component if present
            if use_noise:
                noise_path = os.path.join(mix_dir, f"{mix_id}_noise.wav")
                save_audio(noise_path, noise_added, TARGET_SR)
                
            # Save metadata/labels JSON
            meta = {
                "mix_id": mix_id,
                "n_speakers": n_speakers,
                "overlap_ratio": overlap_ratio,
                "use_reverb": use_reverb,
                "use_noise": use_noise,
                "recursion_labels": recursion_labels
            }
            meta_path = os.path.join(mix_dir, f"{mix_id}_meta.json")
            with open(meta_path, "w") as f_meta:
                json.dump(meta, f_meta, indent=4)
                
            if (i + 1) % 20 == 0 or i == num_mixes_per_n - 1:
                print(f"  Generated {i + 1}/{num_mixes_per_n}...")
                
    print(f"Finished building evaluation set for {split_name}.")

def main():
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.getcwd()
    metadata_dir = os.path.join(base_dir, "metadata")
    cache_dir = os.path.join(base_dir, "cache")
    
    # We build evaluation sets for both validation (dev-clean) and test (test-clean) splits
    # LibriSpeech names in splits.json: dev-clean, test-clean
    build_split_eval_set(metadata_dir, cache_dir, "dev-clean", num_mixes_per_n=100)
    build_split_eval_set(metadata_dir, cache_dir, "test-clean", num_mixes_per_n=100)
    
    print("\nStatic evaluation sets generated successfully!")

if __name__ == "__main__":
    main()

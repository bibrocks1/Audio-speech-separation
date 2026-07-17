import os
import sys
import json
import torch
import numpy as np

# Append workspace root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_speaker_splits(metadata_dir):
    print("Verification Step 1: Checking Speaker Split Disjointness...")
    splits_path = os.path.join(metadata_dir, "speaker_splits.json")
    if not os.path.exists(splits_path):
        print(f"Error: {splits_path} does not exist. Run prepare_data.py first.")
        return False
        
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    train_spk = set(splits.get("train-clean-100", []))
    dev_spk = set(splits.get("dev-clean", []))
    test_spk = set(splits.get("test-clean", []))
    
    print(f"  Train Speakers: {len(train_spk)}")
    print(f"  Dev Speakers:   {len(dev_spk)}")
    print(f"  Test Speakers:  {len(test_spk)}")
    
    overlap_train_dev = train_spk.intersection(dev_spk)
    overlap_train_test = train_spk.intersection(test_spk)
    overlap_dev_test = dev_spk.intersection(test_spk)
    
    success = True
    if len(overlap_train_dev) > 0:
        print(f"  [ERROR] Train and Dev splits overlap: {overlap_train_dev}")
        success = False
    if len(overlap_train_test) > 0:
        print(f"  [ERROR] Train and Test splits overlap: {overlap_train_test}")
        success = False
    if len(overlap_dev_test) > 0:
        print(f"  [ERROR] Dev and Test splits overlap: {overlap_dev_test}")
        success = False
        
    if success:
        print("  [SUCCESS] All speaker splits are 100% disjoint!")
    return success

def verify_source_index(metadata_dir):
    print("\nVerification Step 2: Checking Source Index Metadata...")
    index_path = os.path.join(metadata_dir, "source_index.json")
    if not os.path.exists(index_path):
        print(f"Error: {index_path} does not exist.")
        return False
        
    with open(index_path, "r") as f:
        source_index = json.load(f)
        
    print(f"  Total cached utterances in index: {len(source_index)}")
    durations = [item["duration"] for item in source_index]
    print(f"  Average duration: {np.mean(durations):.2f} seconds")
    print(f"  Min duration:     {np.min(durations):.2f} seconds")
    print(f"  Max duration:     {np.max(durations):.2f} seconds")
    
    if len(source_index) > 0:
        print("  [SUCCESS] Source index is valid.")
        return True
    return False

def verify_data_loader(metadata_dir, cache_dir):
    print("\nVerification Step 3: Loading Batch & Verifying Waveforms...")
    from scripts.dataset import get_dataloader
    
    # Mock step_getter for curriculum
    global_step = 50000 # Phase 3: 3-4 speakers, noise & reverb active
    def get_step():
        return global_step
        
    try:
        dataloader = get_dataloader(
            metadata_dir=metadata_dir,
            cache_dir=cache_dir,
            split="train-clean-100",
            chunk_len_sec=4.0,
            batch_size=2,
            num_workers=0, # Use 0 workers for direct verification in main thread
            shuffle=True,
            step_getter=get_step
        )
    except Exception as e:
        print(f"  [ERROR] Failed to initialize DataLoader: {e}")
        return False
        
    # Get a batch
    batch = next(iter(dataloader))
    mixtures, sources, noises, recursion_labels = batch
    
    print(f"  Batch size: {mixtures.shape[0]}")
    print(f"  Mixture shape: {mixtures.shape}") # Expected: [B, 1, L]
    
    # Run tests per item in batch
    for b_idx in range(mixtures.shape[0]):
        mix_item = mixtures[b_idx]
        noise_item = noises[b_idx]
        sources_list = sources[b_idx]
        labels_list = recursion_labels[b_idx]
        
        n_speakers = len(sources_list)
        print(f"\n  Checking Batch Item {b_idx + 1}:")
        print(f"    Number of speakers: {n_speakers}")
        print(f"    Recursion Labels:   {labels_list}")
        
        # Check peak amplitude normalization (max abs <= 0.90 to avoid clipping)
        peak_amp = torch.max(torch.abs(mix_item)).item()
        print(f"    Peak amplitude:     {peak_amp:.4f}")
        if peak_amp > 0.90:
            print(f"    [WARNING] Peak amplitude is high: {peak_amp:.4f}. Clipping risk!")
        elif peak_amp < 0.80:
            print(f"    [WARNING] Peak amplitude is low: {peak_amp:.4f}")
        else:
            print("    [SUCCESS] Peak amplitude is perfectly normalized (~0.89 dBFS).")
            
        # Check recursion labels length match N
        if len(labels_list) != n_speakers:
            print(f"    [ERROR] Labels length ({len(labels_list)}) does not match speakers count ({n_speakers})")
            return False
            
        if labels_list != [True] * (n_speakers - 1) + [False]:
            print(f"    [ERROR] Recursion labels sequence is wrong: {labels_list}")
            return False
        print("    [SUCCESS] Recursion labels sequence is correct.")
        
        # Check sum conservation: mixture = sum(sources) + noise
        sum_sources = sum(sources_list)
        reconstruction_error = torch.max(torch.abs(mix_item - sum_sources - noise_item)).item()
        print(f"    Sum-conservation error: {reconstruction_error:.2e}")
        if reconstruction_error > 1e-4:
            print(f"    [ERROR] Mixture is NOT equal to sum of sources + noise! Error: {reconstruction_error:.2e}")
            return False
        else:
            print("    [SUCCESS] Perfect additive mixture reconstruction (mixture = sum(sources) + noise).")
            
    print("\n  [SUCCESS] All DataLoader batch checks passed!")
    return True

def main():
    if '__file__' in globals():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.getcwd()
    metadata_dir = os.path.join(base_dir, "metadata")
    cache_dir = os.path.join(base_dir, "cache")
    
    sys.path.append(base_dir)
    
    success = True
    success = success and verify_speaker_splits(metadata_dir)
    success = success and verify_source_index(metadata_dir)
    success = success and verify_data_loader(metadata_dir, cache_dir)
    
    if success:
        print("\n" + "="*50)
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("VERIFICATION FAILED! Please check the errors above.")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()

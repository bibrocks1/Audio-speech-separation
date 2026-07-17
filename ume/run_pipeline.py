import os
import scipy.io.wavfile as wav
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import custom modules
import progressive_dataset as pd
import ume_architecture as ume
import train_regimen as tr

# Define parameters for testing
VOCAB_SIZE = 100
HIDDEN_DIM = 256
NUM_TEST_SAMPLES = 20
DATASET_DIR = "curriculum_dataset"

# ----------------------------------------------------
# 1. Custom Dataset Wrapper
# ----------------------------------------------------

class CurriculumASRDataset(Dataset):
    """Loads standardized WAV mixtures, targets, and simulated text token targets."""
    def __init__(self, dataset_dir, tier, num_samples, vocab_size=100):
        self.mix_dir = os.path.join(dataset_dir, tier, "mix")
        self.clean_dir = os.path.join(dataset_dir, tier, "clean")
        self.num_samples = num_samples
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Read files
        mix_path = os.path.join(self.mix_dir, f"sample_{idx:04d}.wav")
        clean_path = os.path.join(self.clean_dir, f"sample_{idx:04d}.wav")
        
        _, mix_data = wav.read(mix_path)
        _, clean_data = wav.read(clean_path)
        
        # Convert to floats
        mix_tensor = torch.tensor(mix_data, dtype=torch.float32) / 32768.0
        clean_tensor = torch.tensor(clean_data, dtype=torch.float32) / 32768.0
        
        # Truncate waveforms to a multiple of HIDDEN_DIM (256) so they reshape perfectly
        num_frames = len(mix_tensor) // HIDDEN_DIM
        mix_tensor = mix_tensor[:num_frames * HIDDEN_DIM]
        clean_tensor = clean_tensor[:num_frames * HIDDEN_DIM]
        
        # Reshape to simulate frame sequence [Frames, Hidden]
        mix_frames = mix_tensor.view(-1, HIDDEN_DIM)
        clean_frames = clean_tensor.view(-1, HIDDEN_DIM)
        
        # Generate target tokens simulating transcripts with interleaved time anchors
        # Target tokens layout: [SOS, text_token, time_anchor, text_token, time_anchor, EOS]
        target_tokens = torch.randint(1, self.vocab_size - 5, (15,))
        # Infuse simulated anchor tokens (e.g. token_id = vocab_size - 1, vocab_size - 2)
        target_tokens[0] = 0  # SOS token
        target_tokens[-1] = self.vocab_size - 1  # EOS token
        target_tokens[5] = self.vocab_size - 2  # Simulated time anchor
        target_tokens[10] = self.vocab_size - 3  # Simulated time anchor
        
        return mix_frames, clean_frames, target_tokens

# ----------------------------------------------------
# 2. Main execution driver
# ----------------------------------------------------

def main():
    print("====================================================")
    print("Starting iV Speech Separation & ASR Pipeline Execution")
    print("====================================================")
    
    # Step 1: Generate Phase 1 curriculum dataset
    pd.generate_curriculum_datasets(num_samples=NUM_TEST_SAMPLES, output_dir=DATASET_DIR)
    
    # Step 2: Initialize dataloaders
    print("\nSetting up curriculum datasets and loaders...")
    
    tier1_dataset = CurriculumASRDataset(DATASET_DIR, "tier1_clean", NUM_TEST_SAMPLES, vocab_size=VOCAB_SIZE)
    tier2_dataset = CurriculumASRDataset(DATASET_DIR, "tier2_overlap", NUM_TEST_SAMPLES, vocab_size=VOCAB_SIZE)
    tier3_dataset = CurriculumASRDataset(DATASET_DIR, "tier3_dense", NUM_TEST_SAMPLES, vocab_size=VOCAB_SIZE)
    
    loader1 = DataLoader(tier1_dataset, batch_size=4, shuffle=True)
    loader2 = DataLoader(tier2_dataset, batch_size=4, shuffle=True)
    loader3 = DataLoader(tier3_dataset, batch_size=4, shuffle=True)
    
    # Step 3: Instantiate model
    print("\nInstantiating UME architecture...")
    model = ume.iVPipeline(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    
    # Step 4: Run Stage 1 Warmup (1-speaker clean, load pre-trained weights first)
    tr.train_stage1_warmup(model, loader1, epochs=2)
    
    # Step 5: Freeze foundational encoder weights & insert Sidecar separator
    tr.freeze_encoder_and_insert_sidecar(model)
    
    # Step 6: Run Stage 2 Escalation (progressive 2-speaker & 3-speaker mixtures)
    tr.train_stage2_escalation(model, loader2, loader3, epochs=2)
    
    print("\n====================================================")
    print("Verification completed successfully! All stages executed.")
    print("====================================================")

if __name__ == "__main__":
    main()

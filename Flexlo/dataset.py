import os
import sys
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader

if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    sys.path.append(os.getcwd())
from scripts.dynamic_mixer import generate_mixture, TARGET_SR

def resolve_kaggle_paths(metadata_dir, cache_dir):
    if not os.path.exists(metadata_dir) and os.path.exists("/kaggle/input"):
        for item in os.listdir("/kaggle/input"):
            kaggle_dataset_path = os.path.join("/kaggle/input", item)
            pot_meta = os.path.join(kaggle_dataset_path, "metadata")
            pot_cache = os.path.join(kaggle_dataset_path, "cache")
            if os.path.exists(pot_meta) and os.path.exists(pot_cache):
                print(f"Auto-detected Kaggle dataset path: {kaggle_dataset_path}")
                return pot_meta, pot_cache
    return metadata_dir, cache_dir

class DynamicMixtureDataset(Dataset):
    def __init__(self, metadata_dir, cache_dir, split="train-clean-100", 
                 chunk_len_sec=4.0, total_steps=100000, step_getter=None):
        metadata_dir, cache_dir = resolve_kaggle_paths(metadata_dir, cache_dir)
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        self.split = split
        self.chunk_len_sec = chunk_len_sec
        self.total_steps = total_steps
        self.step_getter = step_getter
        
        # Load source index
        index_path = os.path.join(metadata_dir, "source_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Source index not found at {index_path}. Run prepare_data.py first.")
            
        with open(index_path, "r") as f:
            all_sources = json.load(f)
            
        # Filter sources by split
        # LibriSpeech names: train-clean-100, dev-clean, test-clean
        self.sources = [s for s in all_sources if s["split"] == self.split]
        if not self.sources:
            raise ValueError(f"No sources found in index for split: {self.split}")
            
        # Get list of noise and RIR files
        self.noise_files = []
        noise_dir = os.path.join(cache_dir, "noise_bank")
        if os.path.exists(noise_dir):
            self.noise_files = [os.path.join("cache", "noise_bank", f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
            
        self.rir_files = []
        rir_dir = os.path.join(cache_dir, "rir_bank")
        if os.path.exists(rir_dir):
            self.rir_files = [os.path.join("cache", "rir_bank", f) for f in os.listdir(rir_dir) if f.endswith(".wav")]
            
        print(f"Dataset loaded for {split}: {len(self.sources)} clean sources, {len(self.noise_files)} noises, {len(self.rir_files)} RIRs")

    def _get_curriculum_params(self, step):
        progress = step / self.total_steps
        
        if progress < 0.25:
            # Phase 1: 2 speakers, full overlap, no noise, no reverb
            n_speakers_range = (2, 2)
            overlap_range = (1.0, 1.0)
            use_noise, use_reverb = False, False
        elif progress < 0.5:
            # Phase 2: 2-3 speakers, partial/no/full overlap, no noise, no reverb
            n_speakers_range = (2, 3)
            overlap_range = (0.0, 1.0)
            use_noise, use_reverb = False, False
        elif progress < 0.75:
            # Phase 3: 3-4 speakers, full overlap range, noise & reverb active
            n_speakers_range = (3, 4)
            overlap_range = (0.0, 1.0)
            use_noise, use_reverb = True, True
        else:
            # Phase 4: 4-6 speakers, full overlap range, full augmentations
            n_speakers_range = (4, 6)
            overlap_range = (0.0, 1.0)
            use_noise, use_reverb = True, True
            
        return n_speakers_range, overlap_range, use_noise, use_reverb

    def __len__(self):
        # Since we mix dynamically on the fly, we can define any arbitrary size for an epoch
        # Let's say 10000 examples per epoch for train, or length of raw files for validation/testing
        if "train" in self.split:
            return 10000
        else:
            return len(self.sources) // 2 # 2 speakers per mix, so roughly half the sources

    def __getitem__(self, idx):
        # Determine parameters
        if "train" in self.split and self.step_getter is not None:
            current_step = self.step_getter()
            n_speakers_range, overlap_range, use_noise, use_reverb = self._get_curriculum_params(current_step)
            n_speakers = random.randint(*n_speakers_range)
            overlap_ratio = random.uniform(*overlap_range)
        else:
            # Default evaluation/validation setup (hard test): N = 2 to 4 speakers, full overlap ranges, with noise/reverb
            n_speakers = random.randint(2, 4)
            overlap_ratio = random.uniform(0.0, 1.0)
            use_noise = len(self.noise_files) > 0
            use_reverb = len(self.rir_files) > 0
            
        # Generate mixture
        # Note: dynamic_mixer expects base_dir to be the workspace directory
        workspace_dir = os.path.dirname(self.metadata_dir)
        
        mixture, sources, noise, recursion_labels = generate_mixture(
            source_index=self.sources,
            noise_files=self.noise_files,
            rir_files=self.rir_files,
            n_speakers=n_speakers,
            overlap_ratio=overlap_ratio,
            use_noise=use_noise,
            use_reverb=use_reverb,
            chunk_len_sec=self.chunk_len_sec,
            sr=TARGET_SR,
            base_dir=workspace_dir,
            split=self.split
        )
        
        return mixture, sources, noise, recursion_labels

def collate_fn(batch):
    # batch is a list of tuples: (mixture, sources, noise, recursion_labels)
    mixtures = torch.stack([item[0] for item in batch])
    noises = torch.stack([item[2] for item in batch])
    
    # Keep sources and recursion labels as list of lists/tensors
    sources = [item[1] for item in batch]
    recursion_labels = [item[3] for item in batch]
    
    return mixtures, sources, noises, recursion_labels

def get_dataloader(metadata_dir, cache_dir, split="train-clean-100", 
                   chunk_len_sec=4.0, batch_size=4, num_workers=2, 
                   total_steps=100000, step_getter=None, shuffle=True):
    dataset = DynamicMixtureDataset(
        metadata_dir=metadata_dir,
        cache_dir=cache_dir,
        split=split,
        chunk_len_sec=chunk_len_sec,
        total_steps=total_steps,
        step_getter=step_getter
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if "train" in split else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader

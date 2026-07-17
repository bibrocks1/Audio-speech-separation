from src.dynamic_mixer import DynamicMixDataset
from src.curriculum import CurriculumManager

def test():
    print("Initializing curriculum and dataset...")
    curriculum = CurriculumManager(stage2_epoch=10, stage3_epoch=30)
    dataset = DynamicMixDataset(
        speech_index_path="data/indices/speech_index.json",
        noise_index_path="data/indices/noise_index.json",
        curriculum=curriculum,
        epoch=0
    )
    
    print("\n--- Testing Epoch 5 (Stage 1) ---")
    dataset.set_epoch(5)
    item1 = dataset[0]
    print(f"mixed_audio shape: {item1['mixed_audio'].shape}")
    print(f"target_masks shape: {item1['target_masks'].shape}")
    print(f"config: {item1['config']}")
    
    print("\n--- Testing Epoch 15 (Stage 2) ---")
    dataset.set_epoch(15)
    item2 = dataset[0]
    print(f"mixed_audio shape: {item2['mixed_audio'].shape}")
    print(f"target_masks shape: {item2['target_masks'].shape}")
    print(f"config: {item2['config']}")
    
    print("\n--- Testing Epoch 35 (Stage 3) ---")
    dataset.set_epoch(35)
    item3 = dataset[0]
    print(f"mixed_audio shape: {item3['mixed_audio'].shape}")
    print(f"target_masks shape: {item3['target_masks'].shape}")
    print(f"config: {item3['config']}")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test()

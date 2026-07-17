from src.curriculum import CurriculumManager
import pprint

def test():
    manager = CurriculumManager(stage2_epoch=10, stage3_epoch=30)
    
    epochs_to_test = [5, 15, 35]
    
    for epoch in epochs_to_test:
        print(f"\n--- Testing Epoch {epoch} ---")
        config = manager.get_batch_config(batch_size=2, current_epoch=epoch)
        pprint.pprint(config)

if __name__ == "__main__":
    test()

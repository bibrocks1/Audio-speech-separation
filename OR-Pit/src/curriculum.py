import random
import numpy as np
from typing import List, Dict, Any


class CurriculumManager:
    def __init__(self, stage2_epoch: int = 10, stage3_epoch: int = 30):
        self.stage2_epoch = stage2_epoch
        self.stage3_epoch = stage3_epoch

    def _sample_conversational_gaps(self, num_speakers: int, overlap_ratio: float) -> List[float]:
        """
        Samples realistic pause durations between speakers using a Gamma distribution.
        The sum of these gaps correlates with the non-overlapping portion of the conversation.
        """
        # We need num_speakers - 1 gaps between the speakers
        if num_speakers <= 1:
            return []
            
        # Base scale for gaps; higher overlap means smaller gaps.
        # This is a heuristic representation of pause duration in seconds.
        base_gap = 2.0 * (1.0 - overlap_ratio)
        
        # Gamma distribution parameters
        # shape (k) and scale (theta)
        shape = 2.0
        # mean = shape * scale, so scale = mean / shape
        scale = max(0.1, base_gap / shape)
        
        gaps = np.random.gamma(shape=shape, scale=scale, size=num_speakers - 1).tolist()
        return [max(0.0, float(g)) for g in gaps]

    def get_batch_config(self, batch_size: int, current_epoch: int) -> List[Dict[str, Any]]:
        """
        Generates dynamic mixing configurations for a batch based on the training epoch.
        """
        batch_configs = []
        for _ in range(batch_size):
            config = {}
            
            # Stage 1
            if current_epoch < self.stage2_epoch:
                config["stage"] = 1
                config["num_speakers"] = 2
                config["use_noise"] = False
                config["use_reverb"] = False
                config["overlap_ratio"] = random.uniform(0.0, 0.2)
                
            # Stage 2
            elif current_epoch < self.stage3_epoch:
                config["stage"] = 2
                config["num_speakers"] = random.choice([2, 3])
                config["use_noise"] = True
                config["snr_db"] = random.uniform(-5.0, 5.0)
                config["use_reverb"] = False
                config["overlap_ratio"] = random.uniform(0.4, 1.0)
                
            # Stage 3
            else:
                config["stage"] = 3
                config["num_speakers"] = random.choice([4, 5])
                config["use_noise"] = True
                config["snr_db"] = random.uniform(-5.0, 5.0)
                config["use_reverb"] = True
                config["overlap_ratio"] = random.uniform(0.5, 1.0)
                
            config["gaps"] = self._sample_conversational_gaps(
                config["num_speakers"], config["overlap_ratio"]
            )
            
            batch_configs.append(config)
            
        return batch_configs

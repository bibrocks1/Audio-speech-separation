"""Curriculum for on-the-fly mixing (training-data policy; not part of ARCHITECTURE.md).

Default stages reproduce the original training recipe (3 stages):
  Stage 1 (epoch < 10):  2 speakers, overlap ratio U[0.0, 0.2], no noise, no reverb
  Stage 2 (epoch < 30):  2–3 speakers, overlap U[0.4, 1.0], noise SNR U[-5, 5] dB, no reverb
  Stage 3:               4–5 speakers, overlap U[0.5, 1.0], noise SNR U[-5, 5] dB, reverb
Stage boundaries and ranges are configurable under `curriculum.stages` in the YAML config.

Gaps between successive speaker onsets are Gamma(shape=2, scale=max(0.1, 2(1-overlap)/2))
seconds, as in the original implementation (higher overlap → shorter gaps).
"""

from typing import Any

import numpy as np

DEFAULT_STAGES = [
    {"until_epoch": 10, "num_speakers": [2], "overlap": [0.0, 0.2], "noise": False, "reverb": False},
    {
        "until_epoch": 30,
        "num_speakers": [2, 3],
        "overlap": [0.4, 1.0],
        "noise": True,
        "snr_db": [-5.0, 5.0],
        "reverb": False,
    },
    {
        "until_epoch": None,
        "num_speakers": [4, 5],
        "overlap": [0.5, 1.0],
        "noise": True,
        "snr_db": [-5.0, 5.0],
        "reverb": True,
    },
]


def sample_gaps(rng: np.random.Generator, num_speakers: int, overlap_ratio: float) -> list[float]:
    if num_speakers <= 1:
        return []
    base_gap = 2.0 * (1.0 - overlap_ratio)
    shape = 2.0
    scale = max(0.1, base_gap / shape)
    return [max(0.0, float(g)) for g in rng.gamma(shape=shape, scale=scale, size=num_speakers - 1)]


def sample_mix_config(rng: np.random.Generator, stage: dict[str, Any], stage_id: int = 0) -> dict[str, Any]:
    """Draw one mixture configuration from a stage description."""
    n = int(rng.choice(stage["num_speakers"]))
    overlap = float(rng.uniform(*stage["overlap"]))
    cfg = {
        "stage": stage_id,
        "num_speakers": n,
        "overlap_ratio": overlap,
        "use_noise": bool(stage.get("noise", False)),
        "use_reverb": bool(stage.get("reverb", False)),
        "gaps": sample_gaps(rng, n, overlap),
    }
    if cfg["use_noise"]:
        cfg["snr_db"] = float(rng.uniform(*stage.get("snr_db", [-5.0, 5.0])))
    return cfg


class CurriculumManager:
    def __init__(self, stages: list[dict[str, Any]] | None = None, reverb_enabled: bool = True):
        self.stages = stages or DEFAULT_STAGES
        # Global switch: reverb can be disabled everywhere (configurable flag).
        self.reverb_enabled = reverb_enabled

    def stage_index(self, epoch: int) -> int:
        for i, st in enumerate(self.stages):
            if st.get("until_epoch") is None or epoch < st["until_epoch"]:
                return i
        return len(self.stages) - 1

    def get_config(self, epoch: int, rng: np.random.Generator) -> dict[str, Any]:
        i = self.stage_index(epoch)
        stage = dict(self.stages[i])
        if not self.reverb_enabled:
            stage["reverb"] = False
        return sample_mix_config(rng, stage, stage_id=i + 1)

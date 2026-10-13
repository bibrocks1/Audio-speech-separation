"""Audio I/O via soundfile + scipy (no torchaudio dependency)."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path, target_sr: int = 16000, start: int = 0, frames: int = -1) -> np.ndarray:
    """Load mono float32 audio at target_sr. start/frames are in the file's native rate."""
    data, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    wav = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    return wav


def save_audio(path, wav: np.ndarray, sr: int = 16000, subtype: str = "PCM_16") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0), sr, subtype=subtype)

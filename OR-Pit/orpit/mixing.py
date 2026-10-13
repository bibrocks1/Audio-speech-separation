"""Mixture recipes and rendering (LibriSpeech speakers + WHAM! noise + optional reverb).

A *recipe* is a small JSON-serialisable dict that fully determines one mixture
(utterances, crop points, onsets, gains, noise file/offset/SNR, room index). Recipes are
used two ways:
  * static splits: scripts/make_mixtures.py writes data/recipes/<split>.json with a fixed
    seed (and can optionally materialise the wavs), and RecipeDataset renders them;
  * on-the-fly: OnTheFlyDataset draws a fresh recipe per (seed, epoch, index).

Rendering steps
  1. each speaker's utterance is cropped and placed at its onset inside a T-sample window
  2. optional reverb: convolve with an RIR from a precomputed room bank (pyroomacoustics,
     scripts/make_rir_bank.py) or a synthetic exponential-decay RIR as fallback
  3. each source is scaled to -25 dBFS RMS (over its active region) + U[-2.5, 2.5] dB
  4. noise is scaled to the requested SNR relative to the summed speech
  5. if the mixture peak exceeds 0.9 everything is scaled down together (no clipping)
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from orpit.audio_io import load_audio

REF_DBFS = -25.0
PEAK_LIMIT = 0.9


# ── recipe sampling ──────────────────────────────────────────────────────────


def group_by_speaker(index: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for e in index:
        by_spk.setdefault(str(e["speaker"]), []).append(e)
    return by_spk


def sample_recipe(
    rng: np.random.Generator,
    mix_cfg: dict[str, Any],
    by_speaker: dict[str, list],
    noise_index: list | None,
    seg_len: int,
    sr: int = 16000,
    rir_bank_size: int = 0,
    min_active_sec: float = 1.0,
    gain_db_range: float = 2.5,
) -> dict[str, Any]:
    n = mix_cfg["num_speakers"]
    speakers = sorted(by_speaker)
    if len(speakers) < n:
        raise ValueError(f"need {n} distinct speakers, index has {len(speakers)}")
    chosen = rng.choice(len(speakers), size=n, replace=False)
    min_active = min(int(min_active_sec * sr), seg_len)
    onsets = np.concatenate([[0.0], np.cumsum(mix_cfg.get("gaps", [0.0] * (n - 1)))])[:n]

    sources = []
    for k, si in enumerate(chosen):
        spk = speakers[si]
        utt = by_speaker[spk][rng.integers(len(by_speaker[spk]))]
        onset = int(min(round(onsets[k] * sr), seg_len - min_active))
        avail = seg_len - onset
        utt_len = int(utt["num_frames"] * sr / utt.get("sample_rate", sr))
        start = int(rng.integers(0, max(1, utt_len - avail + 1)))
        sources.append(
            {
                "path": utt["path"],
                "speaker": spk,
                "start": start,
                "onset": onset,
                "length": min(avail, utt_len - start),
                "gain_db": float(rng.uniform(-gain_db_range, gain_db_range)),
            }
        )

    recipe = {
        "num_speakers": n,
        "seg_len": seg_len,
        "sr": sr,
        "sources": sources,
        "overlap_ratio": mix_cfg.get("overlap_ratio"),
        "stage": mix_cfg.get("stage"),
        "noise": None,
        "reverb": None,
    }
    if mix_cfg.get("use_noise"):
        if noise_index:
            ne = noise_index[rng.integers(len(noise_index))]
            n_len = int(ne["num_frames"] * sr / ne.get("sample_rate", sr))
            recipe["noise"] = {
                "path": ne["path"],
                "start": int(rng.integers(0, max(1, n_len - seg_len + 1))),
                "snr_db": float(mix_cfg["snr_db"]),
            }
        else:  # documented fallback: white Gaussian noise (labelled in the recipe)
            recipe["noise"] = {
                "path": None,
                "synthetic": "white",
                "seed": int(rng.integers(2**31)),
                "snr_db": float(mix_cfg["snr_db"]),
            }
    if mix_cfg.get("use_reverb"):
        if rir_bank_size > 0:
            recipe["reverb"] = {
                "room": int(rng.integers(rir_bank_size)),
                "slots": [int(x) for x in rng.permutation(6)[:n]],
            }
        else:
            recipe["reverb"] = {"synthetic_seeds": [int(x) for x in rng.integers(2**31, size=n)]}
    return recipe


# ── RIRs ──────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=4)
def load_rir_bank(path: str) -> np.ndarray:
    """[rooms, 6, L] float16 RIR bank produced by scripts/make_rir_bank.py."""
    return np.load(path)["rirs"]


def synthetic_rir(seed: int, sr: int = 16000) -> np.ndarray:
    """Original fallback: 300 ms Gaussian noise with exponential decay, plus a direct-path tap."""
    r = np.random.default_rng(seed)
    n = int(0.3 * sr)
    rir = r.standard_normal(n) * np.exp(-np.linspace(0, 10, n))
    rir[0] = np.abs(rir).max() * 3.0
    return (rir / np.abs(rir).max()).astype(np.float32)


def early_part(rir: np.ndarray, sr: int, early_ms: float = 50.0) -> np.ndarray:
    d = int(np.argmax(np.abs(rir)))
    out = np.zeros_like(rir)
    end = min(len(rir), d + int(early_ms * sr / 1000))
    out[:end] = rir[:end]
    return out


# ── rendering ─────────────────────────────────────────────────────────────────


def _active_rms(x: np.ndarray, lo: int, hi: int) -> float:
    seg = x[lo:hi] if hi > lo else x
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))


def _load_active_crop(path, sr, start, length, min_rms=1e-3):
    """Load the recipe crop; if it is (near-)silent, fall back deterministically to other
    crop points of the same utterance so no target is ever all-silent."""
    wav = load_audio(path, sr, start=start, frames=length)
    if _active_rms(wav, 0, len(wav)) >= min_rms:
        return wav
    full = load_audio(path, sr)
    best = wav
    for st in np.linspace(0, max(0, len(full) - length), 9).astype(int):
        cand = full[st : st + length]
        if _active_rms(cand, 0, len(cand)) > _active_rms(best, 0, len(best)):
            best = cand
        if _active_rms(best, 0, len(best)) >= min_rms:
            break
    return best


def render_recipe(
    recipe: dict[str, Any],
    audio_root: str,
    rir_bank_path: str | None = None,
    reverb_target: str = "reverberant",
) -> dict[str, np.ndarray]:
    """Return {"mixture": [T], "sources": [n, T], "noise": [T]} float32."""
    sr, T = recipe["sr"], recipe["seg_len"]
    root = Path(audio_root)
    bank = (
        load_rir_bank(str(rir_bank_path)) if (recipe.get("reverb") and "room" in recipe["reverb"]) else None
    )

    srcs = []
    for k, s in enumerate(recipe["sources"]):
        wav = _load_active_crop(root / s["path"], sr, s["start"], s["length"])
        dry = np.zeros(T, dtype=np.float32)
        L = min(len(wav), T - s["onset"])
        dry[s["onset"] : s["onset"] + L] = wav[:L]
        lo, hi = s["onset"], s["onset"] + L
        rv = recipe.get("reverb")
        if rv:
            rir = (
                bank[rv["room"], rv["slots"][k]].astype(np.float32)
                if bank is not None
                else synthetic_rir(rv["synthetic_seeds"][k], sr)
            )
            wet = fftconvolve(dry, rir)[:T]
            target = wet if reverb_target == "reverberant" else fftconvolve(dry, early_part(rir, sr))[:T]
            # scale both by the same factor so mixture/target stay consistent
            g = 10 ** ((REF_DBFS + s["gain_db"]) / 20) / _active_rms(wet, lo, hi)
            srcs.append((wet * g, target * g))
        else:
            g = 10 ** ((REF_DBFS + s["gain_db"]) / 20) / _active_rms(dry, lo, hi)
            srcs.append((dry * g, dry * g))

    mix_part = np.stack([a for a, _ in srcs]).astype(np.float32)
    targets = np.stack([b for _, b in srcs]).astype(np.float32)
    speech = mix_part.sum(0)

    noise = np.zeros(T, dtype=np.float32)
    nz = recipe.get("noise")
    if nz:
        if nz.get("path"):
            raw = load_audio(root / nz["path"], sr, start=nz["start"], frames=T)
            if len(raw) < T:
                raw = np.tile(raw, int(np.ceil(T / max(1, len(raw)))))[:T]
        else:
            raw = np.random.default_rng(nz["seed"]).standard_normal(T).astype(np.float32)
        p_s = np.mean(speech.astype(np.float64) ** 2)
        p_n = np.mean(raw.astype(np.float64) ** 2)
        if p_n > 0 and p_s > 0:
            noise = (raw * np.sqrt(p_s / p_n * 10 ** (-nz["snr_db"] / 10))).astype(np.float32)

    mixture = speech + noise
    peak = float(np.abs(mixture).max())
    if peak > PEAK_LIMIT:
        s = PEAK_LIMIT / peak
        mixture, targets, noise = mixture * s, targets * s, noise * s
    return {
        "mixture": mixture.astype(np.float32),
        "sources": targets.astype(np.float32),
        "noise": noise.astype(np.float32),
    }

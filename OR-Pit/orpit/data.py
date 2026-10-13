"""Datasets, collate and resumable sampler.

* OnTheFlyDataset – fresh curriculum-driven recipe per (seed, epoch, index); no disk use.
* RecipeDataset   – renders a fixed list of recipes (static val/test/train splits).
Both return dicts: mixture [T], sources [n, T], num_speakers, recipe.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from orpit.curriculum import CurriculumManager
from orpit.mixing import group_by_speaker, render_recipe, sample_recipe


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class OnTheFlyDataset(Dataset):
    def __init__(
        self,
        speech_index: list[dict],
        noise_index: list[dict] | None,
        curriculum: CurriculumManager,
        audio_root: str,
        length: int,
        seg_len: int,
        sr: int = 16000,
        seed: int = 0,
        rir_bank: str | None = None,
        reverb_target: str = "reverberant",
        min_active_sec: float = 1.0,
    ):
        self.by_speaker = group_by_speaker(speech_index)
        self.noise_index = noise_index or []
        self.curriculum = curriculum
        self.audio_root = audio_root
        self.length, self.seg_len, self.sr, self.seed = length, seg_len, sr, seed
        self.rir_bank = rir_bank if rir_bank and Path(rir_bank).exists() else None
        self.rir_bank_size = 0
        if self.rir_bank:
            from orpit.mixing import load_rir_bank

            self.rir_bank_size = int(load_rir_bank(self.rir_bank).shape[0])
        self.reverb_target = reverb_target
        self.min_active_sec = min_active_sec
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def make_recipe(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        mix_cfg = self.curriculum.get_config(self.epoch, rng)
        return sample_recipe(
            rng,
            mix_cfg,
            self.by_speaker,
            self.noise_index,
            self.seg_len,
            self.sr,
            rir_bank_size=self.rir_bank_size,
            min_active_sec=self.min_active_sec,
        )

    def __getitem__(self, idx: int):
        recipe = self.make_recipe(idx)
        out = render_recipe(recipe, self.audio_root, self.rir_bank, self.reverb_target)
        return {
            "mixture": torch.from_numpy(out["mixture"]),
            "sources": torch.from_numpy(out["sources"]),
            "num_speakers": recipe["num_speakers"],
            "recipe": recipe,
        }


class RecipeDataset(Dataset):
    def __init__(
        self,
        recipes: list[dict],
        audio_root: str,
        rir_bank: str | None = None,
        reverb_target: str = "reverberant",
    ):
        self.recipes = recipes
        self.audio_root = audio_root
        self.rir_bank = rir_bank
        self.reverb_target = reverb_target

    def __len__(self):
        return len(self.recipes)

    def __getitem__(self, idx: int):
        r = self.recipes[idx]
        out = render_recipe(r, self.audio_root, self.rir_bank, self.reverb_target)
        return {
            "mixture": torch.from_numpy(out["mixture"]),
            "sources": torch.from_numpy(out["sources"]),
            "num_speakers": r["num_speakers"],
            "recipe": r,
        }


class FolderMixtureDataset(Dataset):
    """LibriMix-style folder: <root>/<mix_dir>/<id>.wav and <root>/s1/<id>.wav, s2/, ... (any length)."""

    def __init__(self, root, mix_dir="mix_both", limit=None, sr=16000):
        import soundfile as sf

        self.sf, self.root, self.mix_dir, self.sr = sf, Path(root), mix_dir, sr
        self.ids = sorted(p.stem for p in (self.root / mix_dir).glob("*.wav"))[: limit or None]
        self.srcs = sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and d.name[:1] == "s" and d.name[1:].isdigit()
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        i = self.ids[idx]
        mix, sr = self.sf.read(str(self.root / self.mix_dir / f"{i}.wav"), dtype="float32")
        assert sr == self.sr, f"{i}: {sr} Hz"
        srcs = np.stack(
            [self.sf.read(str(self.root / s / f"{i}.wav"), dtype="float32")[0] for s in self.srcs]
        )
        return {
            "mixture": torch.from_numpy(mix),
            "sources": torch.from_numpy(srcs),
            "num_speakers": len(self.srcs),
            "recipe": {"id": i},
        }


def collate_mixtures(batch):
    """→ mixture [B, 1, T], sources [B, max_n, T] (zero-padded), num_speakers LongTensor [B], recipes."""
    max_n = max(item["sources"].shape[0] for item in batch)
    T = batch[0]["mixture"].shape[-1]
    sources = torch.zeros(len(batch), max_n, T)
    for i, item in enumerate(batch):
        sources[i, : item["sources"].shape[0]] = item["sources"]
    mixture = torch.stack([item["mixture"] for item in batch]).unsqueeze(1)
    n = torch.tensor([item["num_speakers"] for item in batch], dtype=torch.long)
    return mixture, sources, n, [item["recipe"] for item in batch]


class ResumableShuffleSampler(Sampler):
    """Permutation seeded by (seed, epoch); can start part-way through for mid-epoch resume."""

    def __init__(self, length: int, seed: int = 0, shuffle: bool = True):
        self.length, self.seed, self.shuffle = length, seed, shuffle
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch, 7]).permutation(self.length)
        else:
            order = np.arange(self.length)
        return iter(order[self.start :].tolist())

    def __len__(self):
        return self.length - self.start

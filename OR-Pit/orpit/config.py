"""YAML config loading with `extends:` inheritance and `key.sub=value` CLI overrides."""

import copy
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_ROOT = "data/audio"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_abs(p: str) -> bool:
    # treat Windows drive paths as absolute on every OS (configs are shared)
    return bool(p) and (Path(p).is_absolute() or (len(p) > 2 and p[1] == ":" and p[2] in "/\\"))


def load_config(path, overrides=None) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return apply_env(cfg)


def apply_env(cfg: dict) -> dict:
    """Resolve ``paths.audio_root``.

    ``ORPIT_AUDIO_ROOT`` replaces an unset or absolute (machine-specific) audio root, so the
    same config, or a config stored inside a checkpoint, keeps working after the data moves.
    Project-relative roots such as the synthetic smoke data are kept as written. Without the
    variable, an unset root falls back to ``data/audio`` inside the project.
    """
    paths = cfg.setdefault("paths", {})
    env, root = os.environ.get("ORPIT_AUDIO_ROOT"), paths.get("audio_root")
    if env and (not root or _is_abs(root)):
        paths["audio_root"] = env
    elif not root:
        paths["audio_root"] = DEFAULT_AUDIO_ROOT
    return cfg


def resolve(path_str: str) -> Path:
    """Resolve a config path relative to the OR-Pit project root."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p

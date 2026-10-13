"""Fetch LibriMix metadata (MIT licence, https://github.com/JorisCos/LibriMix) into data/librimix/.

    python scripts/get_librimix_metadata.py

Downloads the Libri2Mix / Libri3Mix test-clean mixture lists (used to build the standard
benchmark test sets with scripts/make_librimix_test.py) and the LibriSpeech utterance-length
tables (used by scripts/build_index.py --librimix_lengths to index train-clean-360 without
opening 100k audio files).
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import resolve  # noqa: E402

BASE = "https://raw.githubusercontent.com/JorisCos/LibriMix/master/"
FILES = [
    "LICENSE",
    "metadata/Libri2Mix/libri2mix_test-clean.csv",
    "metadata/Libri3Mix/libri3mix_test-clean.csv",
    "metadata/LibriSpeech/train-clean-100.csv",
    "metadata/LibriSpeech/train-clean-360.csv",
    "metadata/LibriSpeech/dev-clean.csv",
    "metadata/LibriSpeech/test-clean.csv",
]


def main():
    out = resolve("data/librimix")
    for f in FILES:
        dest = out / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {f}")
            continue
        with urllib.request.urlopen(BASE + f, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"[ok] {f} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

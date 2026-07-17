"""
build_kaggle_indices.py
───────────────────────
Scans Kaggle input directories for audio files and writes JSON index files
that DynamicMixDataset can consume directly.

Usage (from the notebook working directory, e.g. /kaggle/working):
    python -m src.build_kaggle_indices \
        --speech_dir /kaggle/input/librispeech-train-clean-360 \
        --noise_dir  /kaggle/input/whamr-noise

Outputs:
    ./data/indices/speech_index.json
    ./data/indices/noise_index.json

Each entry in the JSON arrays has the shape:
    {"path": "<abs_path>", "sample_rate": <int>, "num_frames": <int>}

soundfile.info() is used for metadata (no audio decode), so indexing
120,000 files takes only a few seconds.
"""

import os
import glob
import json
import argparse
import soundfile as sf


# ── Supported extensions ──────────────────────────────────────────────────────
SPEECH_EXTS = ("*.flac", "*.wav")
NOISE_EXTS  = ("*.wav",  "*.flac")


def scan_audio_files(root_dir: str, extensions: tuple) -> list:
    """
    Recursively find all audio files under root_dir matching the given
    glob extensions.  Returns a sorted list of absolute paths.
    """
    found = []
    for ext in extensions:
        pattern = os.path.join(root_dir, "**", ext)
        found.extend(glob.glob(pattern, recursive=True))

    # Deduplicate (a .flac might match both *.flac and if we ever overlap exts)
    # and sort for a deterministic, reproducible index order.
    return sorted(set(found))


def build_index(paths: list) -> list:
    """
    For each audio path, read sample_rate and num_frames via soundfile.info()
    (metadata only — no PCM decode) and return a list of record dicts.
    Paths that cannot be read are skipped with a warning.
    """
    records = []
    skipped = 0

    for i, path in enumerate(paths):
        try:
            info = sf.info(path)
            records.append(
                {
                    "path":        os.path.abspath(path),
                    "sample_rate": info.samplerate,
                    "num_frames":  info.frames,
                }
            )
        except Exception as exc:
            print(f"  [WARN] Skipping {path}: {exc}")
            skipped += 1

        if (i + 1) % 5000 == 0:
            print(f"  Indexed {i + 1:,} / {len(paths):,} files…")

    if skipped:
        print(f"  [WARN] {skipped} file(s) skipped due to read errors.")

    return records


def save_index(records: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved {len(records):,} entries → {output_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build DynamicMixDataset-compatible JSON index files from "
                    "Kaggle input directories."
    )
    parser.add_argument(
        "--speech_dir",
        required=True,
        help="Root directory to scan for speech files (.flac, .wav). "
             "Example: /kaggle/input/librispeech-train-clean-360",
    )
    parser.add_argument(
        "--noise_dir",
        required=True,
        help="Root directory to scan for noise files (.wav, .flac). "
             "Example: /kaggle/input/whamr-noise",
    )
    parser.add_argument(
        "--speech_out",
        default="data/indices/speech_index.json",
        help="Output path for the speech index JSON (default: data/indices/speech_index.json).",
    )
    parser.add_argument(
        "--noise_out",
        default="data/indices/noise_index.json",
        help="Output path for the noise index JSON (default: data/indices/noise_index.json).",
    )
    args = parser.parse_args()

    # ── Speech index ──────────────────────────────────────────────────────────
    print(f"\nScanning speech files in: {args.speech_dir}")
    speech_paths = scan_audio_files(args.speech_dir, SPEECH_EXTS)
    print(f"  Found {len(speech_paths):,} speech files. Building index…")
    speech_records = build_index(speech_paths)
    save_index(speech_records, args.speech_out)

    # ── Noise index ───────────────────────────────────────────────────────────
    print(f"\nScanning noise files in: {args.noise_dir}")
    noise_paths = scan_audio_files(args.noise_dir, NOISE_EXTS)
    print(f"  Found {len(noise_paths):,} noise files. Building index…")
    noise_records = build_index(noise_paths)
    save_index(noise_records, args.noise_out)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\n✓ Index build complete."
        f"\n  Speech entries : {len(speech_records):,}"
        f"\n  Noise entries  : {len(noise_records):,}"
        f"\n  Ready to run   : python -m src.train"
    )


if __name__ == "__main__":
    main()

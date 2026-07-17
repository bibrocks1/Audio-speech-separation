import argparse
import json
from pathlib import Path
import soundfile as sf


def index_audio(target_dir: str, output_json: str) -> None:
    """Recursively scan *target_dir* for audio files and write metadata to *output_json*.

    The function looks for files with ``.wav`` or ``.flac`` extensions. For each file it
    extracts:
    - absolute file path
    - sample rate
    - total number of frames (length)

    The information is saved as a JSON list where each entry is a dictionary with the
    keys ``"path"``, ``"sample_rate"`` and ``"num_frames"``.
    """
    target_path = Path(target_dir)
    if not target_path.is_dir():
        raise ValueError(f"Target directory does not exist: {target_dir}")

    audio_files = list(target_path.rglob("*.wav")) + list(target_path.rglob("*.flac"))
    metadata = []
    for audio_file in audio_files:
        try:
            info = sf.info(str(audio_file))
            metadata.append({
                "path": str(audio_file.resolve()),
                "sample_rate": info.samplerate,
                "num_frames": info.frames,
            })
        except Exception as e:
            # Skip files that torchaudio cannot read and report the issue.
            print(f"Warning: could not read {audio_file}: {e}")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata for {len(metadata)} files written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate an audio metadata index.")
    parser.add_argument("target_dir", type=str, help="Directory to scan for .wav/.flac files")
    parser.add_argument("output_json", type=str, help="Path to write the JSON index")
    args = parser.parse_args()
    index_audio(args.target_dir, args.output_json)


if __name__ == "__main__":
    main()

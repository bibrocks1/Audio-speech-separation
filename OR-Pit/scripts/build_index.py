"""Build JSON indices (paths relative to audio_root, forward slashes).

    python scripts/build_index.py --audio_root data/audio [--out data/indices]

LibriSpeech/train-clean-100 → train.json, dev-clean → val.json, test-clean → test.json
wham_noise_subset/{tr,cv,tt} → noise_tr.json, noise_cv.json, noise_tt.json
Speaker id = first field of the LibriSpeech filename. These LibriSpeech subsets have
disjoint speakers by construction (verified by scripts/sanity_check_data.py).
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import resolve  # noqa: E402

SPEECH = {"train": "train-clean-100", "val": "dev-clean", "test": "test-clean"}
NOISE = {"noise_tr": "tr", "noise_cv": "cv", "noise_tt": "tt"}


def scan(root: Path, sub: Path, speech: bool):
    entries = []
    files = sorted(list((root / sub).rglob("*.flac")) + list((root / sub).rglob("*.wav")))
    for p in [f for f in files if not f.name.endswith(".tmp.flac")]:
        info = sf.info(str(p))
        e = {
            "path": p.relative_to(root).as_posix(),
            "sample_rate": info.samplerate,
            "num_frames": info.frames,
        }
        if speech:
            e["speaker"] = p.name.split("-")[0]
        entries.append(e)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out", default="data/indices")
    ap.add_argument("--wham_dir", default="wham_noise_subset")
    ap.add_argument("--only", choices=["all", "speech", "noise"], default="all")
    ap.add_argument(
        "--librimix_lengths",
        nargs="*",
        default=None,
        help="LibriSpeech subsets to index from LibriMix's length tables instead of opening "
        "every file (fast), e.g. train-clean-100 train-clean-360 → train_<names>.json",
    )
    args = ap.parse_args()
    root, out = Path(args.audio_root), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.librimix_lengths:
        import csv

        entries, missing = [], 0
        for sub in args.librimix_lengths:
            for r in csv.DictReader(open(resolve("data/librimix/metadata/LibriSpeech") / f"{sub}.csv")):
                rel = "LibriSpeech/" + r["origin_path"]
                if not (root / rel).parent.exists():
                    missing += 1
                    continue
                entries.append(
                    {
                        "path": rel,
                        "sample_rate": 16000,
                        "num_frames": int(r["length"]),
                        "speaker": r["speaker_ID"],
                        "sex": r["sex"],
                    }
                )
        name = "train_" + "_".join(s.replace("train-clean-", "") for s in args.librimix_lengths)
        json.dump(entries, open(out / f"{name}.json", "w"), indent=0)
        print(
            f"[{name}] {len(entries)} utterances, {len({e['speaker'] for e in entries})} speakers, "
            f"{sum(e['num_frames'] for e in entries) / 16000 / 3600:.1f} h (skipped {missing} missing)"
        )
        return
    for name, sub in SPEECH.items() if args.only in ("all", "speech") else []:
        d = Path("LibriSpeech") / sub
        if not (root / d).exists():
            print(f"[skip] {root / d} missing")
            continue
        e = scan(root, d, True)
        json.dump(e, open(out / f"{name}.json", "w"), indent=0)
        hours = sum(x["num_frames"] / x["sample_rate"] for x in e) / 3600
        print(f"[{name}] {len(e)} utterances, {len({x['speaker'] for x in e})} speakers, {hours:.1f} h")
    for name, sub in NOISE.items() if args.only in ("all", "noise") else []:
        d = Path(args.wham_dir) / sub
        if not (root / d).exists():
            print(f"[skip] {root / d} missing")
            continue
        e = scan(root, d, False)
        json.dump(e, open(out / f"{name}.json", "w"), indent=0)
        print(
            f"[{name}] {len(e)} noise files, {sum(x['num_frames'] / x['sample_rate'] for x in e) / 3600:.1f} h"
        )


if __name__ == "__main__":
    main()

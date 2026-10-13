"""Fetch exactly the WHAM! noise files that the LibriMix test sets reference, unmodified
(channel 0, float32, 16 kHz, original file names), so Libri2Mix / Libri3Mix test mixtures can
be rebuilt bit-for-bit like the official create_librimix_from_metadata.py.
WHAM! noise is CC BY-NC 4.0 (http://wham.whisper.ai/).

    python scripts/download_wham_librimix.py --root data/audio [--workers 8]
Output: <root>/wham_noise/tt/<name>.wav
"""

import argparse
import csv
import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orpit.config import resolve  # noqa: E402

URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from remotezip import RemoteZip

    meta = resolve("data/librimix/metadata")
    needed = set()
    for f in ("Libri2Mix/libri2mix_test-clean.csv", "Libri3Mix/libri3mix_test-clean.csv"):
        needed |= {r["noise_path"] for r in csv.DictReader(open(meta / f))}
    out = Path(args.root) / "wham_noise"
    todo = sorted(n for n in needed if not (out / n).exists())
    print(f"[wham-librimix] {len(needed)} files referenced, {len(todo)} to fetch")
    local, lock, done, t0 = threading.local(), threading.Lock(), [len(needed) - len(todo)], time.time()

    def fetch(rel):
        for attempt in range(8):
            try:
                if getattr(local, "zip", None) is None:
                    local.zip = RemoteZip(URL, timeout=120)
                data, sr = sf.read(
                    io.BytesIO(local.zip.read("wham_noise/" + rel)), dtype="float32", always_2d=True
                )
                break
            except Exception as e:
                print(f"  retry {attempt + 1} {rel}: {str(e)[:100]}", flush=True)
                local.zip = None
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(rel)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.stem + ".tmp.wav")
        sf.write(str(tmp), data[:, 0], sr, subtype="FLOAT")
        tmp.replace(dest)
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"  {done[0]}/{len(needed)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(fetch, todo))
    print(f"[wham-librimix] done → {out}")


if __name__ == "__main__":
    main()

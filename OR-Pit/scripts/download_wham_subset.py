"""Fetch a deterministic subset of WHAM! noise without downloading the whole 17 GB zip.

WHAM! noise: CC BY-NC 4.0 (non-commercial), http://wham.whisper.ai/ . Official zip:
  https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
(17 GB zipped / 35 GB unzipped; 16 kHz, 2-channel float32; splits tr=20,000 cv=5,000 tt=3,000).

HTTP range requests (remotezip) read only the chosen members. Each file is down-mixed to
mono, peak-limited to 0.9 if needed, and stored as 16-bit FLAC:
  <root>/wham_noise_subset/{tr,cv,tt}/<name>.flac   + MANIFEST.json + LICENSE_NOTICE.txt

    python scripts/download_wham_subset.py --root data/audio --n_tr 3000 --n_cv 500 --n_tt 500 [--workers 8]
Re-running resumes (existing files are skipped).
"""

import argparse
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n_tr", type=int, default=3000)
    ap.add_argument("--n_cv", type=int, default=500)
    ap.add_argument("--n_tt", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--workers", type=int, default=8, help="parallel range-request connections")
    args = ap.parse_args()
    from remotezip import RemoteZip

    out = Path(args.root) / "wham_noise_subset"
    out.mkdir(parents=True, exist_ok=True)
    (out / "LICENSE_NOTICE.txt").write_text(
        "Subset of the WHAM! noise dataset (http://wham.whisper.ai/), licensed CC BY-NC 4.0.\n"
        "Non-commercial use only. Converted to mono 16-bit FLAC for OR-PiT experiments.\n"
    )
    t0 = time.time()
    local = threading.local()
    with RemoteZip(args.url, timeout=120) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".wav")]
        print(f"[wham] {len(names)} wav members in zip ({time.time() - t0:.0f}s to read directory)")
        rng = np.random.default_rng(args.seed)
        manifest = {"url": args.url, "seed": args.seed, "files": {}}
        for split, n in (("tr", args.n_tr), ("cv", args.n_cv), ("tt", args.n_tt)):
            members = sorted(m for m in names if f"/{split}/" in m)
            pick = sorted(rng.choice(len(members), size=min(n, len(members)), replace=False))
            chosen = [members[i] for i in pick]
            manifest["files"][split] = chosen
            (out / split).mkdir(exist_ok=True)

            def have(m, split=split):
                f = out / split / (Path(m).stem + ".flac")
                try:
                    return f.exists() and sf.info(str(f)).frames > 0
                except Exception:  # partial file from an interrupted run → refetch
                    return False

            todo = [m for m in chosen if not have(m)]
            done = [len(chosen) - len(todo)]
            lock = threading.Lock()

            def fetch(m, split=split, done=done, lock=lock, n_total=len(chosen)):  # noqa: B008
                for attempt in range(8):
                    try:
                        if getattr(local, "zip", None) is None:
                            local.zip = RemoteZip(args.url, timeout=120)
                        data, sr = sf.read(io.BytesIO(local.zip.read(m)), dtype="float32", always_2d=True)
                        break
                    except Exception as e:  # connection resets: back off and reconnect
                        print(f"  retry {attempt + 1} {m}: {str(e)[:120]}", flush=True)
                        local.zip = None
                        time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError(f"failed to fetch {m}")
                mono = data.mean(axis=1)
                peak = float(np.abs(mono).max())
                if peak > 0.9:
                    mono = mono * (0.9 / peak)
                dest = out / split / (Path(m).stem + ".flac")
                tmp = dest.with_suffix(".tmp.flac")
                sf.write(str(tmp), mono, sr, subtype="PCM_16", format="FLAC")
                tmp.replace(dest)
                with lock:
                    done[0] += 1
                    if done[0] % 100 == 0:
                        print(
                            f"  [{split}] {done[0]}/{n_total}  elapsed {(time.time() - t0) / 60:.1f} min",
                            flush=True,
                        )

            with ThreadPoolExecutor(args.workers) as ex:
                list(ex.map(fetch, todo))
            print(f"[wham] {split}: {len(chosen)} files")
        with open(out / "MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"[wham] done in {(time.time() - t0) / 60:.1f} min → {out}")


if __name__ == "__main__":
    main()

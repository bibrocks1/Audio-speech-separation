"""Download + md5-verify + extract LibriSpeech subsets (CC BY 4.0, https://www.openslr.org/12/).

    python scripts/download_librispeech.py --root data/audio [--subsets dev-clean test-clean train-clean-100]

Sizes (compressed): dev-clean 337 MB, test-clean 346 MB, train-clean-100 6.3 GB.
Downloads resume after interruption (HTTP Range). Archives are kept in <root>/downloads
(delete them yourself after a successful extract to free ~7 GB).
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

MD5 = {  # from https://openslr.trmal.net/resources/12/md5sum.txt
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
}
MIRRORS = [
    "https://www.openslr.org/resources/12",
    "https://openslr.elda.org/resources/12",
    "https://openslr.trmal.net/resources/12",
]


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest: Path):
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "orpit-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored Range → restart
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        mode = "ab" if have else "wb"
        t0, done, last = time.time(), have, 0.0
        with open(dest, mode) as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if time.time() - last > 15:
                    last = time.time()
                    rate = (done - have) / max(1e-6, last - t0) / 2**20
                    print(
                        f"    {dest.name}: {done / 2**20:,.0f}/{total / 2**20:,.0f} MB ({rate:.1f} MB/s)",
                        flush=True,
                    )
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subsets", nargs="+", default=["dev-clean", "test-clean", "train-clean-100"])
    args = ap.parse_args()
    root = Path(args.root)
    dl = root / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    for sub in args.subsets:
        marker = root / "LibriSpeech" / f".{sub}.done"
        if marker.exists():
            print(f"[skip] {sub} already extracted")
            continue
        free_gb = shutil.disk_usage(root).free / 2**30
        print(f"[{sub}] free disk {free_gb:.1f} GB")
        arc = dl / f"{sub}.tar.gz"
        ok = arc.exists() and md5sum(arc) == MD5[sub]
        for mirror in MIRRORS:
            if ok:
                break
            try:
                print(f"[{sub}] downloading from {mirror}")
                download(f"{mirror}/{sub}.tar.gz", arc)
                got = md5sum(arc)
                ok = got == MD5[sub]
                print(f"[{sub}] md5 {got} {'OK' if ok else 'MISMATCH (expected ' + MD5[sub] + ')'}")
                if not ok:
                    arc.rename(arc.with_suffix(".bad"))
            except Exception as e:  # try next mirror, keeping partial file for resume
                print(f"[{sub}] {mirror} failed: {e}")
        if not ok:
            print(f"[{sub}] FAILED to obtain a verified archive")
            sys.exit(1)
        print(f"[{sub}] extracting to {root}")
        with tarfile.open(arc, "r:gz") as t:
            t.extractall(root, filter="data") if sys.version_info >= (3, 12) else t.extractall(root)
        marker.write_text("ok\n")
        print(f"[{sub}] done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download the SPIHIM experimental single-pixel data used by ASPIR.

SPIHIM is the open experimental collection of the OpenSpyrit ecosystem:
Beneti Martins et al., "OpenSpyrit: an ecosystem for open single-pixel
hyperspectral imaging", Opt. Express 31(10), 15599 (2023),
DOI 10.1364/OE.483937. Licence CC-BY 4.0 — cite it when you use the data.

The data are not part of the ASPIR repository. This script fetches them into
``datasets/spihim/`` (git-ignored), resumes interrupted downloads, checks
SHA-256 checksums and unpacks each archive next to it.

    python scripts/download_spihim.py             # pattern bank + Jun + Nov (~1 GB)
    python scripts/download_spihim.py --with-jul  # also Jul (+900 MB, STL-10 prints)
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://www.creatis.insa-lyon.fr/~ducros/Spihim/"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO / "datasets" / "spihim"

# name -> (size in bytes, SHA-256). Checksums computed on the first download
# (2026-09-27); a mismatch means the remote file changed or the copy is corrupt.
FILES = {
    "Hadamard_64x64.zip": (2843575, "b80000351cef90aef6a13bda957d9164b28fa366faebe9497952d8f3cba1c1e4"),
    "spihim_2020-Jun-11.zip": (None, "b6c151fc0db6c3f9f38be1214693b1c92003939434f32b4f88ecc1af68c56791"),
    "spihim_2020-Nov-18.zip": (None, "7353cde70713c0fb09a1f5e980a3e1e6021e388798b043e61c6e5949acbd5a56"),
    "spihim_2020-Jul-01.zip": (None, None),
}
DEFAULT_SET = ["Hadamard_64x64.zip", "spihim_2020-Jun-11.zip", "spihim_2020-Nov-18.zip"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(name, dest):
    """Fetch ``name`` into ``dest``, resuming a partial ``.part`` file."""
    target = dest / name
    if target.exists():
        return target
    part = dest / (name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(BASE_URL + name)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        resumed = resp.status == 206
        total = resp.headers.get("Content-Length")
        total = int(total) + (have if resumed else 0) if total else None
        mode = "ab" if resumed else "wb"
        done = have if resumed else 0
        with open(part, mode) as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total:
                    print(f"\r  {name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.rename(target)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--with-jul", action="store_true", help="also fetch the Jul session (STL-10 prints)")
    ap.add_argument("--keep-zip", action="store_true", help="keep the archives after unpacking")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    names = DEFAULT_SET + (["spihim_2020-Jul-01.zip"] if args.with_jul else [])
    sums_file = args.dest / "SHA256SUMS"
    recorded = {}
    if sums_file.exists():
        for line in sums_file.read_text().splitlines():
            digest, fname = line.split(maxsplit=1)
            recorded[fname.strip()] = digest

    ok = True
    for name in names:
        unpacked = args.dest / name[:-4]
        if unpacked.exists() and name in recorded:
            print(f"  {name}: already unpacked")
            continue
        path = download(name, args.dest)
        digest = sha256(path)
        expected = FILES[name][1] or recorded.get(name)
        if expected and digest != expected:
            print(f"  {name}: CHECKSUM MISMATCH {digest} != {expected}", file=sys.stderr)
            ok = False
            continue
        recorded[name] = digest
        print(f"  {name}: sha256 {digest}")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(unpacked)
        if not args.keep_zip:
            path.unlink()

    sums_file.write_text("".join(f"{d}  {n}\n" for n, d in sorted(recorded.items())))
    print(f"Checksums in {sums_file}")
    print("SPIHIM: Beneti Martins et al., Opt. Express 31(10), 15599 (2023), "
          "DOI 10.1364/OE.483937 — CC-BY 4.0.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

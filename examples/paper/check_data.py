#!/usr/bin/env python
"""Check that a batch run scored the paper's test images.

    python examples/paper/check_data.py <batch results folder> [...]

A batch exported with the *All data* level writes ``data/<test>/test_images.npz``,
whose ``originals`` are the test split's reference images. For the paper's
datasets they must be bit-identical to the original submission's: the SHA-256 of
the array (float32, C order) is compared with the values below, computed from the
stored R0 runs (UC2: 300 CelebA images, 64 x 64; UC1: 200 IR beams, 32 x 32). A
different hash means a different dataset or split, and the comparisons with
earlier results (SPyRiT, SPIHIM, R0) would no longer use the same images.

Exit status 0 when every test matches.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

EXPECTED = {
    (300, 64, 64): ("UC2 CelebA", "efecd81c3917637dc0faf6162bec16920c42bbf8930a4d8ea7e0e2cd085b487c"),
    (200, 32, 32): ("UC1 IR beams", "deafb51c1d82175f23419ee99a5190fca5e90fea39b50c94943d9913c1120858"),
}


def sha256(images):
    return hashlib.sha256(np.ascontiguousarray(images, dtype=np.float32).tobytes()).hexdigest()


def check(folder: Path) -> bool:
    files = sorted(folder.glob("data/*/test_images.npz"))
    if not files:
        print(f"[FAIL] {folder}: no data/*/test_images.npz (export with the 'All data' level)")
        return False
    ok = True
    for f in files:
        with np.load(f) as z:
            originals = z["originals"]
        name, expected = EXPECTED.get(originals.shape, (None, None))
        got = sha256(originals)
        match = got == expected
        ok = ok and match
        label = name or f"unknown shape {originals.shape}"
        print(f"[{'PASS' if match else 'FAIL'}] {f.parent.name}: {label}, sha256 {got[:16]}…")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = [check(Path(p)) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)

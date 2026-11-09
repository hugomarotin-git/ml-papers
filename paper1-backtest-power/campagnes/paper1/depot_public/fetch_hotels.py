"""
Download the public "Hotel booking demand" dataset (Antonio, de Almeida & Nunes, 2019,
Data in Brief 22) used in Section 6 of the paper, and verify its SHA-256.

Source: the TidyTuesday mirror of 2020-02-11, a byte-for-byte copy of the file published
with the article. The hash below is the one recorded when the campaigns were run.

Usage:
    python fetch_hotels.py [--out data/hors_finance/hotels.csv]

The script refuses to write a file whose hash does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[4] / "data" / "hors_finance" / "hotels.csv"
)  # ../data next to the repo


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write hotels.csv")
    ap.add_argument("--check-only", action="store_true", help="only verify an existing file")
    a = ap.parse_args()

    if a.check_only or a.out.exists():
        if not a.out.exists():
            print(f"missing: {a.out}", file=sys.stderr)
            return 1
        got = sha256_of(a.out.read_bytes())
        ok = got == SHA256
        print(f"{a.out}: sha256 {got} -> {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        data = resp.read()
    got = sha256_of(data)
    if got != SHA256:
        print(f"sha256 mismatch: expected {SHA256}, got {got}; nothing written", file=sys.stderr)
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    print(f"written {a.out} ({len(data):,} bytes), sha256 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

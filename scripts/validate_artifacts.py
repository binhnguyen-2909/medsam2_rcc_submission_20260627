"""Validate SHA256 checksums exported by scripts/export_checksums.py."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    failures = []
    for line in (ROOT / args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"missing {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"sha mismatch {rel}: expected {expected}, got {actual}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("all artifacts validated")


if __name__ == "__main__":
    main()

"""Export SHA256 checksums for reproducibility-critical files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = [
    "configs",
    "src",
    "checkpoints",
    "models",
    "data/manifests",
    "results",
    "legacy_artifacts",
]
DEFAULT_FILES = [
    ".gitattributes",
    ".gitignore",
    "README.md",
    "environment.yml",
    "requirements.txt",
    "pyproject.toml",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/manifests/artifact_checksums.sha256")
    parser.add_argument("--include-large-data", action="store_true")
    args = parser.parse_args()

    dirs = list(DEFAULT_DIRS)
    if args.include_large_data:
        dirs += ["data/raw", "data/ground_truth", "data/processed", "data/deliverable_dataset", "data/interim"]

    out = ROOT / args.out
    rows = []
    for rel_file in DEFAULT_FILES:
        path = ROOT / rel_file
        if path.is_file() and path.resolve() != out.resolve():
            rows.append(f"{sha256(path)}  {path.relative_to(ROOT)}")
    for rel_dir in dirs:
        base = ROOT / rel_dir
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.resolve() == out.resolve():
                continue
            rows.append(f"{sha256(path)}  {path.relative_to(ROOT)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} checksums -> {out}")


if __name__ == "__main__":
    main()

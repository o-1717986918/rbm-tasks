#!/usr/bin/env python3
"""Materialize the digit manifest into a portable split/class folder tree."""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_out = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["image"])
            if not source.is_absolute():
                source = args.source_root / source
            suffix = source.suffix.lower() or ".jpg"
            digest = hashlib.sha1(str(source).encode()).hexdigest()[:12]
            name = f"{row['source']}_{source.stem}_{digest}{suffix}"
            relative = Path("images") / row["split"] / row["class_name"] / name
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            rows_out.append({**row, "image": relative.as_posix()})
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "label", "class_name", "source", "group", "split"])
        writer.writeheader(); writer.writerows(rows_out)
    print(f"materialized {len(rows_out)} images at {args.output}")


if __name__ == "__main__":
    main()

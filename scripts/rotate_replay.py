"""Replace older replay snapshots with newer ones under size/file limits."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rotate replay snapshots to keep storage bounded.")
    p.add_argument("--source_dir", default="data/replay/raw/dqn/FrozenLake-v1/fresh")
    p.add_argument("--ext", default=".npz")
    p.add_argument("--max_files", type=int, default=25)
    p.add_argument("--max_total_mb", type=int, default=2048)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir)
    source.mkdir(parents=True, exist_ok=True)

    files = sorted(source.glob(f"*{args.ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    keep: list[Path] = []
    total_bytes = 0
    max_bytes = int(args.max_total_mb) * 1024 * 1024

    for file in files:
        size = file.stat().st_size
        if len(keep) < args.max_files and total_bytes + size <= max_bytes:
            keep.append(file)
            total_bytes += size

    remove = [f for f in files if f not in keep]

    print(f"[rotate_replay] source={source}")
    print(f"[rotate_replay] files_found={len(files)} files_kept={len(keep)} files_removed={len(remove)}")
    print(f"[rotate_replay] kept_total_mb={total_bytes / (1024 * 1024):.2f}")

    for f in remove:
        print(f"[rotate_replay] remove {f.name}")
        if not args.dry_run:
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

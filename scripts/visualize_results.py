"""Generate consolidated visualizations from result CSV files."""

from __future__ import annotations

import argparse

from rl_course.utils.results import update_overview_visualizations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build overview charts from result/csv logs.")
    p.add_argument("--result_dir", default="result")
    p.add_argument("--visualization_dir", default="visualization")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    update_overview_visualizations(
        result_dir=args.result_dir,
        visualization_dir=args.visualization_dir,
    )
    print(f"[visualize_results] updated visualizations in {args.visualization_dir}")


if __name__ == "__main__":
    main()

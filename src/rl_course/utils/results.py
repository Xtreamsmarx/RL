"""Result and visualization helpers for training runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_output_dirs(result_dir: str = "result", visualization_dir: str = "visualization") -> tuple[Path, Path, Path, Path]:
    result_root = Path(result_dir)
    csv_dir = result_root / "csv"
    fig_dir = result_root / "figures"
    vis_dir = Path(visualization_dir)

    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    return result_root, csv_dir, fig_dir, vis_dir


def append_csv_row(csv_path: Path, fieldnames: list[str], row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_series_csv(csv_path: Path, header: list[str], rows: Iterable[tuple]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_line_plot(path: Path, xs: list[float], ys: list[float], title: str, xlabel: str, ylabel: str) -> None:
    if not xs or not ys:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def update_overview_visualizations(result_dir: str = "result", visualization_dir: str = "visualization") -> None:
    _, csv_dir, _, vis_dir = ensure_output_dirs(result_dir=result_dir, visualization_dir=visualization_dir)

    classical_csv = csv_dir / "classical_runs.csv"
    if classical_csv.exists():
        algorithms: list[str] = []
        eval_means: list[float] = []
        with open(classical_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("eval_mean") not in (None, ""):
                    algorithms.append(str(row.get("algorithm", "unknown")))
                    eval_means.append(float(row["eval_mean"]))

        if algorithms and eval_means:
            plt.figure(figsize=(10, 4.8))
            plt.bar(range(len(eval_means)), eval_means)
            plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha="right")
            plt.title("Classical RL Eval Mean by Run")
            plt.ylabel("Eval mean return")
            plt.tight_layout()
            plt.savefig(vis_dir / "classical_eval_overview.png", dpi=150)
            plt.close()

    dqn_csv = csv_dir / "dqn_runs.csv"
    if dqn_csv.exists():
        run_idx: list[int] = []
        mean_returns: list[float] = []
        mean_losses: list[float] = []
        with open(dqn_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                run_idx.append(idx)
                mean_returns.append(float(row.get("mean_return", 0.0)))
                mean_losses.append(float(row.get("mean_loss", 0.0)))

        if run_idx:
            save_line_plot(
                path=vis_dir / "dqn_mean_return_over_runs.png",
                xs=run_idx,
                ys=mean_returns,
                title="DQN Mean Return Over Runs",
                xlabel="Run index",
                ylabel="Mean return",
            )
            save_line_plot(
                path=vis_dir / "dqn_mean_loss_over_runs.png",
                xs=run_idx,
                ys=mean_losses,
                title="DQN Mean Loss Over Runs",
                xlabel="Run index",
                ylabel="Mean loss",
            )

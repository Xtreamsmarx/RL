"""Rasterization helper for Q-value tables on 4x4 FrozenLake."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARROWS = ["←", "↓", "→", "↑"]


def save_q_policy_raster(q_values: np.ndarray, path: str | Path, title: str = "Greedy Policy Raster") -> None:
    if q_values.shape[0] != 16:
        raise ValueError("FrozenLake 4x4 rasterizer expects 16 states.")
    if q_values.shape[1] != 4:
        raise ValueError("FrozenLake action size should be 4.")

    greedy = np.argmax(q_values, axis=1).reshape(4, 4)
    value = np.max(q_values, axis=1).reshape(4, 4)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(value, cmap="viridis")
    for r in range(4):
        for c in range(4):
            ax.text(c, r, ARROWS[int(greedy[r, c])], ha="center", va="center", color="white", fontsize=15)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

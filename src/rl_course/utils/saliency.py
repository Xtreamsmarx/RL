"""Saliency utilities for DQN-style discrete-state models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_course.networks.mlp import DiscreteQNetwork


def dqn_saliency_for_state(model: DiscreteQNetwork, state_index: int, n_states: int) -> np.ndarray:
    model.eval()
    x = torch.zeros((1, n_states), dtype=torch.float32, requires_grad=True)
    x[0, state_index] = 1.0

    q = model(x)
    action = int(torch.argmax(q, dim=1).item())
    q[0, action].backward()

    grads = x.grad.detach().cpu().numpy()[0]
    return np.abs(grads)


def save_saliency_plot(values: np.ndarray, path: str | Path, title: str = "DQN State Saliency") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 2.8))
    plt.bar(np.arange(values.shape[0]), values)
    plt.title(title)
    plt.xlabel("Input state index")
    plt.ylabel("|gradient|")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


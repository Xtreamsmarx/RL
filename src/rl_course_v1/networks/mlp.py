"""Neural network architectures for value-based deep RL."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiscreteQNetwork(nn.Module):
    """Small MLP for discrete-action Q-value approximation."""

    def __init__(self, input_dim: int, n_actions: int, hidden_dims: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, n_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

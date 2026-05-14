"""Replay buffer with bounded storage and reproducible persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Numpy replay buffer with hard capacity assertions."""

    def __init__(self, capacity: int, state_shape: tuple[int, ...], max_assert_size: Optional[int] = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_assert_size is not None:
            assert capacity <= max_assert_size, (
                f"Requested capacity {capacity} exceeds max_assert_size {max_assert_size}."
            )

        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty replay buffer.")
        idx = rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
        )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            next_states=self.next_states[: self.size],
            dones=self.dones[: self.size],
        )

    @classmethod
    def load_npz(
        cls,
        path: str | Path,
        capacity: Optional[int] = None,
        max_assert_size: Optional[int] = None,
    ) -> "ReplayBuffer":
        data = np.load(path)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_states = data["next_states"]
        dones = data["dones"]

        n = int(states.shape[0])
        cap = int(capacity) if capacity is not None else max(n, 1)
        buffer = cls(capacity=cap, state_shape=tuple(states.shape[1:]), max_assert_size=max_assert_size)

        n_copy = min(n, buffer.capacity)
        buffer.states[:n_copy] = states[:n_copy]
        buffer.actions[:n_copy] = actions[:n_copy]
        buffer.rewards[:n_copy] = rewards[:n_copy]
        buffer.next_states[:n_copy] = next_states[:n_copy]
        buffer.dones[:n_copy] = dones[:n_copy]
        buffer.size = n_copy
        buffer.ptr = n_copy % buffer.capacity
        return buffer

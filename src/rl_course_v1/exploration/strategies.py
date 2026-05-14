"""
Exploration injection strategies.

Provides:
  - EpsilonGreedy     : ε-greedy (decaying or fixed)
  - UCB1              : Upper Confidence Bound (tabular)
  - BoltzmannExplorer : softmax / Boltzmann exploration
  - CountBonus        : intrinsic curiosity via visit count bonus
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class EpsilonGreedy:
    """
    ε-greedy exploration with optional linear or exponential decay.

    Parameters
    ----------
    epsilon_start : initial ε
    epsilon_end   : minimum ε after decay
    decay_steps   : number of steps over which to anneal ε linearly
                    (set to 0 for constant ε)
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.01,
        decay_steps:   int   = 50_000,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.decay_steps   = decay_steps
        self._step         = 0

    @property
    def epsilon(self) -> float:
        if self.decay_steps == 0:
            return self.epsilon_start
        frac = min(self._step / self.decay_steps, 1.0)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1.0 - frac)

    def select(
        self,
        q_values: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        self._step += 1
        if rng.random() < self.epsilon:
            return int(rng.integers(len(q_values)))
        return int(np.argmax(q_values))

    def reset(self):
        self._step = 0


class UCB1:
    """
    UCB1 exploration for tabular state-action pairs.

    A_t = argmax_a [Q(s,a) + c √(ln N(s) / (N(s,a) + 1))]
    """

    def __init__(self, n_states: int, n_actions: int, c: float = 2.0):
        self.c = c
        self.N_sa = np.zeros((n_states, n_actions), dtype=np.float64)
        self.N_s  = np.zeros(n_states, dtype=np.float64)

    def select(self, s: int, q_values: np.ndarray) -> int:
        self.N_s[s] += 1
        bonus  = self.c * np.sqrt(np.log(self.N_s[s]) / (self.N_sa[s] + 1))
        a      = int(np.argmax(q_values + bonus))
        self.N_sa[s, a] += 1
        return a

    def reset(self):
        self.N_sa[:] = 0
        self.N_s[:]  = 0


class BoltzmannExplorer:
    """
    Softmax / Boltzmann exploration.

    π(a|s) ∝ exp(Q(s,a) / τ)

    Parameters
    ----------
    tau : temperature; high → uniform exploration, low → greedy
    """

    def __init__(self, tau: float = 1.0):
        self.tau = tau

    def select(self, q_values: np.ndarray, rng: np.random.Generator) -> int:
        shifted = q_values - np.max(q_values)          # numerical stability
        probs   = np.exp(shifted / self.tau)
        probs  /= probs.sum()
        return int(rng.choice(len(q_values), p=probs))


class CountBonus:
    """
    Intrinsic curiosity via visit-count exploration bonus.

    r_intrinsic(s,a) = β / √N(s,a)

    Parameters
    ----------
    beta : bonus scale factor
    """

    def __init__(self, n_states: int, n_actions: int, beta: float = 1.0):
        self.beta = beta
        self.N    = np.zeros((n_states, n_actions), dtype=np.float64)

    def bonus(self, s: int, a: int) -> float:
        return self.beta / np.sqrt(self.N[s, a] + 1.0)

    def update(self, s: int, a: int):
        self.N[s, a] += 1

    def augmented_reward(self, s: int, a: int, r: float) -> float:
        b = self.bonus(s, a)
        self.update(s, a)
        return r + b

    def reset(self):
        self.N[:] = 0

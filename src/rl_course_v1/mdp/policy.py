"""
Policy representations for tabular RL.

Provides:
  - Policy          : stochastic policy π(a|s) stored as a (S,A) probability matrix
  - DeterministicPolicy : deterministic π(s) → a stored as an integer array
  - QValueFunction  : Q(s,a) table
  - ValueFunction   : V(s) array
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Value functions
# ---------------------------------------------------------------------------
class ValueFunction:
    """State-value function V(s) as a flat float array."""

    def __init__(self, n_states: int, init: float = 0.0):
        self.n_states = n_states
        self.values   = np.full(n_states, init, dtype=np.float64)

    def __getitem__(self, s: int) -> float:
        return float(self.values[s])

    def __setitem__(self, s: int, v: float):
        self.values[s] = v

    def copy(self) -> "ValueFunction":
        vf = ValueFunction(self.n_states)
        vf.values = self.values.copy()
        return vf

    def max_diff(self, other: "ValueFunction") -> float:
        return float(np.max(np.abs(self.values - other.values)))

    def __repr__(self) -> str:
        return f"ValueFunction(n_states={self.n_states})"


class QValueFunction:
    """Action-value function Q(s,a) as a (S,A) float array."""

    def __init__(self, n_states: int, n_actions: int, init: float = 0.0):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.values    = np.full((n_states, n_actions), init, dtype=np.float64)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, val):
        self.values[key] = val

    def to_value_function(self, policy: "Policy") -> ValueFunction:
        """V(s) = Σ_a π(a|s) Q(s,a)."""
        vf = ValueFunction(self.n_states)
        vf.values = np.sum(policy.probs * self.values, axis=1)
        return vf

    def copy(self) -> "QValueFunction":
        q = QValueFunction(self.n_states, self.n_actions)
        q.values = self.values.copy()
        return q

    def max_diff(self, other: "QValueFunction") -> float:
        return float(np.max(np.abs(self.values - other.values)))

    def __repr__(self) -> str:
        return f"QValueFunction(n_states={self.n_states}, n_actions={self.n_actions})"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class Policy:
    """
    Stochastic tabular policy π(a|s) stored as a (S,A) probability matrix.
    
    Can represent both stochastic and deterministic policies.
    A deterministic policy has a single 1.0 entry per row.
    """

    def __init__(self, n_states: int, n_actions: int):
        self.n_states  = n_states
        self.n_actions = n_actions
        # Initialise as uniform random policy
        self.probs = np.ones((n_states, n_actions), dtype=np.float64) / n_actions

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------
    @classmethod
    def uniform(cls, n_states: int, n_actions: int) -> "Policy":
        return cls(n_states, n_actions)

    @classmethod
    def greedy_from_q(cls, q: QValueFunction) -> "Policy":
        """Deterministic greedy policy derived from Q(s,a)."""
        pi = cls(q.n_states, q.n_actions)
        pi.probs[:] = 0.0
        best_actions = np.argmax(q.values, axis=1)
        pi.probs[np.arange(q.n_states), best_actions] = 1.0
        return pi

    @classmethod
    def greedy_from_v(cls, v: ValueFunction, mdp) -> "Policy":
        """
        Deterministic greedy policy derived from V(s) using the MDP model.
        π(s) = argmax_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
        """
        from rl_course_v1.mdp.base import TabularMDP
        assert isinstance(mdp, TabularMDP)
        pi = cls(mdp.n_states, mdp.n_actions)
        pi.probs[:] = 0.0
        q_values = (
            mdp.R_matrix
            + mdp.gamma * np.einsum("ijk,k->ij", mdp.P_matrix, v.values)
        )
        best_actions = np.argmax(q_values, axis=1)
        pi.probs[np.arange(mdp.n_states), best_actions] = 1.0
        return pi

    @classmethod
    def epsilon_greedy_from_q(
        cls, q: QValueFunction, epsilon: float
    ) -> "Policy":
        """ε-greedy policy from Q(s,a)."""
        pi = cls(q.n_states, q.n_actions)
        pi.probs[:] = epsilon / q.n_actions
        best_actions = np.argmax(q.values, axis=1)
        pi.probs[np.arange(q.n_states), best_actions] += 1.0 - epsilon
        return pi

    # ------------------------------------------------------------------
    def act(self, s: int, rng: Optional[np.random.Generator] = None) -> int:
        """Sample an action from π(·|s)."""
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(self.n_actions, p=self.probs[s]))

    def deterministic_action(self, s: int) -> int:
        """Return argmax_a π(a|s); only meaningful for deterministic policies."""
        return int(np.argmax(self.probs[s]))

    def copy(self) -> "Policy":
        pi = Policy(self.n_states, self.n_actions)
        pi.probs = self.probs.copy()
        return pi

    def is_deterministic(self) -> bool:
        return bool(np.all(np.max(self.probs, axis=1) == 1.0))

    def __repr__(self) -> str:
        det = "deterministic" if self.is_deterministic() else "stochastic"
        return f"Policy({det}, n_states={self.n_states}, n_actions={self.n_actions})"


class DeterministicPolicy:
    """
    Compact deterministic policy stored as an integer array.
    Convenience wrapper when you only need π(s) → a.
    """

    def __init__(self, actions: np.ndarray):
        self.actions = actions.astype(np.int64)

    @classmethod
    def from_policy(cls, pi: Policy) -> "DeterministicPolicy":
        return cls(np.argmax(pi.probs, axis=1))

    def act(self, s: int) -> int:
        return int(self.actions[s])

    def to_policy(self, n_actions: int) -> Policy:
        n_states = len(self.actions)
        pi = Policy(n_states, n_actions)
        pi.probs[:] = 0.0
        pi.probs[np.arange(n_states), self.actions] = 1.0
        return pi

    def __repr__(self) -> str:
        return f"DeterministicPolicy(n_states={len(self.actions)})"

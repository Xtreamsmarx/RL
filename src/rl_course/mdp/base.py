"""
MDP base abstractions.

An MDP is a 5-tuple (S, A, P, R, γ) where
  S  – finite state space
  A  – finite action space
  P  – transition kernel  P(s'|s,a)
  R  – expected reward    R(s,a,s')
  γ  – discount factor ∈ [0,1)

This module provides:
  - TabularMDP  : wraps a Gymnasium discrete env into the (S,A,P,R,γ) tuple
  - Subtask     : a named slice of the MDP (sub-state space + sub-goal)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core type aliases
# ---------------------------------------------------------------------------
State  = int
Action = int
# Transition entry: (probability, next_state, reward, done)
TransitionEntry = Tuple[float, State, float, bool]


# ---------------------------------------------------------------------------
# TabularMDP
# ---------------------------------------------------------------------------
class TabularMDP:
    """
    Wraps a Gymnasium env that exposes `env.P` (the standard Toy-Text dict)
    into explicit matrices for use by DP / model-based algorithms.

    Attributes
    ----------
    n_states  : int
    n_actions : int
    P_matrix  : ndarray shape (n_states, n_actions, n_states)  – transition probs
    R_matrix  : ndarray shape (n_states, n_actions)            – expected rewards
    gamma     : float
    P_raw     : dict  – original Gymnasium transition dict P[s][a]
    """

    def __init__(self, env, gamma: float = 0.99):
        # Support both raw envs and wrapped envs (e.g. TimeLimit)
        raw = env.unwrapped if hasattr(env, "unwrapped") else env
        assert hasattr(raw, "P"), (
            "Environment must expose a transition dict `env.unwrapped.P` "
            "(e.g. FrozenLake-v1, Taxi-v3, CliffWalking-v0)."
        )
        self.env       = env
        self.gamma     = gamma
        self.n_states  = env.observation_space.n
        self.n_actions = env.action_space.n
        self.P_raw     = raw.P  # P[s][a] = [(prob, s', r, done), ...]

        self.P_matrix, self.R_matrix = self._build_matrices()

    # ------------------------------------------------------------------
    def _build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build dense (S,A,S) transition and (S,A) reward matrices."""
        S, A = self.n_states, self.n_actions
        P = np.zeros((S, A, S), dtype=np.float64)
        R = np.zeros((S, A),    dtype=np.float64)

        for s in range(S):
            for a in range(A):
                for prob, s_next, reward, done in self.P_raw[s][a]:
                    P[s, a, s_next] += prob
                    R[s, a]         += prob * reward

        return P, R

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def transitions(self, s: State, a: Action) -> List[TransitionEntry]:
        """Raw transition list from the Gymnasium dict."""
        return self.P_raw[s][a]

    def expected_reward(self, s: State, a: Action) -> float:
        return float(self.R_matrix[s, a])

    def state_space(self) -> np.ndarray:
        return np.arange(self.n_states)

    def action_space_array(self) -> np.ndarray:
        return np.arange(self.n_actions)

    def __repr__(self) -> str:
        return (
            f"TabularMDP(n_states={self.n_states}, "
            f"n_actions={self.n_actions}, gamma={self.gamma})"
        )


# ---------------------------------------------------------------------------
# Subtask
# ---------------------------------------------------------------------------
@dataclass
class Subtask:
    """
    A named sub-problem of the full MDP.

    Parameters
    ----------
    name        : human-readable identifier
    state_ids   : list of state indices that belong to this subtask
    goal_states : subset of state_ids considered terminal / rewarding
    mdp         : reference to the parent TabularMDP
    """
    name        : str
    state_ids   : List[State]
    goal_states : List[State]
    mdp         : TabularMDP

    # Derived
    _state_set  : set = field(init=False, repr=False)

    def __post_init__(self):
        self._state_set = set(self.state_ids)

    def contains(self, s: State) -> bool:
        return s in self._state_set

    def is_goal(self, s: State) -> bool:
        return s in self.goal_states

    def local_transitions(self, s: State, a: Action) -> List[TransitionEntry]:
        """Transitions clipped to stay within this subtask's state space."""
        raw = self.mdp.transitions(s, a)
        return [
            (p, s_next, r, done or s_next not in self._state_set)
            for p, s_next, r, done in raw
        ]

    def __repr__(self) -> str:
        return (
            f"Subtask(name={self.name!r}, "
            f"|states|={len(self.state_ids)}, "
            f"goals={self.goal_states})"
        )

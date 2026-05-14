"""
Monte Carlo methods for tabular RL (model-free, episodic).

Implements:
  - mc_prediction_first_visit  : first-visit MC policy evaluation â†’ V^Ï€
  - mc_prediction_every_visit  : every-visit MC policy evaluation â†’ V^Ï€
  - mc_control_es              : MC Exploring Starts â†’ Ï€*
  - mc_control_epsilon_greedy  : on-policy MC control with Îµ-greedy â†’ Ï€*
  - mc_control_off_policy_is   : off-policy MC control (importance sampling)

Reference:
  Sutton & Barto Ch. 5 (2018).
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Callable, Optional

from rl_course.mdp.policy import Policy, ValueFunction, QValueFunction


Episode = list[tuple[int, int, float]]   # [(s, a, r), ...]


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------
def generate_episode(
    env,
    pi: Policy,
    rng: np.random.Generator,
    max_steps: int = 1_000,
) -> Episode:
    """Roll out one episode following Ï€."""
    episode: Episode = []
    s, _ = env.reset()
    for _ in range(max_steps):
        a = pi.act(int(s), rng)
        s_next, r, terminated, truncated, _ = env.step(a)
        episode.append((int(s), int(a), float(r)))
        s = s_next
        if terminated or truncated:
            break
    return episode


# ---------------------------------------------------------------------------
# MC Prediction
# ---------------------------------------------------------------------------
def mc_prediction_first_visit(
    env,
    pi: Policy,
    n_episodes: int = 5_000,
    gamma: float = 0.99,
    rng: Optional[np.random.Generator] = None,
) -> ValueFunction:
    """First-visit MC policy evaluation."""
    rng  = rng or np.random.default_rng()
    n_s  = env.observation_space.n
    V    = ValueFunction(n_s)
    returns: dict[int, list[float]] = defaultdict(list)

    for _ in range(n_episodes):
        ep     = generate_episode(env, pi, rng)
        G      = 0.0
        visited: set[int] = set()

        for s, a, r in reversed(ep):
            G = r + gamma * G
            if s not in visited:
                visited.add(s)
                returns[s].append(G)
                V[s] = float(np.mean(returns[s]))

    return V


def mc_prediction_every_visit(
    env,
    pi: Policy,
    n_episodes: int = 5_000,
    gamma: float = 0.99,
    rng: Optional[np.random.Generator] = None,
) -> ValueFunction:
    """Every-visit MC policy evaluation."""
    rng   = rng or np.random.default_rng()
    n_s   = env.observation_space.n
    V     = ValueFunction(n_s)
    N     = np.zeros(n_s, dtype=np.float64)
    total = np.zeros(n_s, dtype=np.float64)

    for _ in range(n_episodes):
        ep = generate_episode(env, pi, rng)
        G  = 0.0

        for s, a, r in reversed(ep):
            G         = r + gamma * G
            N[s]     += 1
            total[s] += G
            V[s]      = total[s] / N[s]

    return V


# ---------------------------------------------------------------------------
# MC Control
# ---------------------------------------------------------------------------
def mc_control_epsilon_greedy(
    env,
    n_episodes: int = 50_000,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, QValueFunction]:
    """
    On-policy first-visit MC control with Îµ-greedy exploration.
    Returns (Ï€*, Q*).
    """
    rng   = rng or np.random.default_rng()
    n_s   = env.observation_space.n
    n_a   = env.action_space.n
    Q     = QValueFunction(n_s, n_a, init=0.0)
    N     = np.zeros((n_s, n_a), dtype=np.float64)
    pi    = Policy.epsilon_greedy_from_q(Q, epsilon)

    for _ in range(n_episodes):
        ep = generate_episode(env, pi, rng)
        G  = 0.0
        visited: set[tuple[int, int]] = set()

        for s, a, r in reversed(ep):
            G        = r + gamma * G
            pair     = (s, a)
            if pair not in visited:
                visited.add(pair)
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

        # Improve policy after each episode
        pi = Policy.epsilon_greedy_from_q(Q, epsilon)

    return pi, Q


def mc_control_exploring_starts(
    env,
    n_episodes: int = 50_000,
    gamma: float    = 0.99,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, QValueFunction]:
    """
    MC Exploring Starts (ES) â€” assumes env can be reset to any (s,a).
    Approximated here by random first action.
    Returns (Ï€*, Q*).
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q   = QValueFunction(n_s, n_a, init=0.0)
    N   = np.zeros((n_s, n_a), dtype=np.float64)
    pi  = Policy.greedy_from_q(Q)

    for _ in range(n_episodes):
        # Exploring start: pick random first action
        s0, _ = env.reset()
        a0    = int(rng.integers(n_a))
        ep: Episode = [(int(s0), a0, 0.0)]

        s_next, r, terminated, truncated, _ = env.step(a0)
        ep[-1] = (int(s0), a0, float(r))
        s = s_next

        if not (terminated or truncated):
            rest = generate_episode(env, pi, rng)
            ep.extend(rest)

        G       = 0.0
        visited: set[tuple[int, int]] = set()

        for s, a, r in reversed(ep):
            G    = r + gamma * G
            pair = (s, a)
            if pair not in visited:
                visited.add(pair)
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

        pi = Policy.greedy_from_q(Q)

    return pi, Q


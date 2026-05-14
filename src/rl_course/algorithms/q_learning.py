"""
Q-learning (off-policy TD control).

Implements:
  - q_learning  : one-step Q-learning â†’ Q*

Q(S_t,A_t) â† Q(S_t,A_t) + Î± [R_{t+1} + Î³ max_{a'} Q(S_{t+1},a') âˆ’ Q(S_t,A_t)]

Reference:
  Watkins & Dayan (1992); Sutton & Barto Ch. 6.5 (2018).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from rl_course.mdp.policy import Policy, QValueFunction


def q_learning(
    env,
    n_episodes: int = 10_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, QValueFunction]:
    """
    Tabular Q-learning (off-policy, one-step TD).

    Behaviour policy : Îµ-greedy w.r.t. current Q
    Target policy    : greedy (max_a Q)

    Returns
    -------
    (Ï€*, Q*)
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q   = QValueFunction(n_s, n_a, init=0.0)

    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False

        while not done:
            # Îµ-greedy behaviour policy
            if rng.random() < epsilon:
                a = int(rng.integers(n_a))
            else:
                a = int(np.argmax(Q[int(s)]))

            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # Off-policy target: greedy max
            max_q_next = 0.0 if done else float(np.max(Q[int(s_next)]))
            td_target  = r + gamma * max_q_next
            Q[int(s), a] += alpha * (td_target - Q[int(s), a])
            s = s_next

    pi = Policy.epsilon_greedy_from_q(Q, epsilon)
    return pi, Q


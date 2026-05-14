"""
Sarsa algorithms — on-policy TD control operating on Q-values.

Implements:
  - sarsa_zero          : one-step Sarsa → Q^π
  - n_step_sarsa        : forward-view Sarsa(n) → Q*
  - sarsa_lambda        : backward-view Sarsa(λ) via eligibility traces → Q*

All methods use ε-greedy exploration and improve Q(s,a) directly.

Reference:
  Sutton & Barto Ch. 6.4, 7.2, 12.7 (2018).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from rl_course_v1.mdp.policy import Policy, QValueFunction


# ---------------------------------------------------------------------------
# One-step Sarsa
# ---------------------------------------------------------------------------
def sarsa_zero(
    env,
    n_episodes: int = 10_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, QValueFunction]:
    """
    One-step on-policy Sarsa.

    Q(S_t,A_t) ← Q(S_t,A_t) + α [R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)]
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q   = QValueFunction(n_s, n_a, init=0.0)

    def eps_greedy(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_a))
        return int(np.argmax(Q[s]))

    for _ in range(n_episodes):
        s, _  = env.reset()
        a     = eps_greedy(int(s))
        done  = False

        while not done:
            s_next, r, terminated, truncated, _ = env.step(a)
            done   = terminated or truncated
            a_next = eps_greedy(int(s_next))
            td_target = r + gamma * (0.0 if done else Q[int(s_next), a_next])
            Q[int(s), a] += alpha * (td_target - Q[int(s), a])
            s, a = s_next, a_next

    pi = Policy.epsilon_greedy_from_q(Q, epsilon)
    return pi, Q


# ---------------------------------------------------------------------------
# n-step Sarsa  (forward view)
# ---------------------------------------------------------------------------
def n_step_sarsa(
    env,
    n: int          = 4,
    n_episodes: int = 10_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, QValueFunction]:
    """
    n-step on-policy Sarsa (forward view / bootstrapped Q-values).

    G_{t:t+n} = R_{t+1} + … + γ^{n-1} R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})
    Q(S_t, A_t) ← Q(S_t, A_t) + α [G_{t:t+n} − Q(S_t, A_t)]
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q   = QValueFunction(n_s, n_a, init=0.0)

    def eps_greedy(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_a))
        return int(np.argmax(Q[s]))

    for _ in range(n_episodes):
        s, _    = env.reset()
        states  = [int(s)]
        actions = [eps_greedy(int(s))]
        rewards = [0.0]        # rewards[0] unused (padding)
        done    = False
        T       = float("inf")
        t       = 0

        while True:
            if t < T:
                s_next, r, terminated, truncated, _ = env.step(actions[t])
                rewards.append(float(r))
                states.append(int(s_next))
                if terminated or truncated:
                    T = t + 1
                else:
                    actions.append(eps_greedy(int(s_next)))

            tau = t - n + 1
            if tau >= 0:
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, min(tau + n, int(T)) + 1)
                )
                if tau + n < T:
                    G += gamma ** n * Q[states[tau + n], actions[tau + n]]
                s_tau = states[tau]
                a_tau = actions[tau]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])

            if tau == T - 1:
                break
            t += 1

    pi = Policy.epsilon_greedy_from_q(Q, epsilon)
    return pi, Q


# ---------------------------------------------------------------------------
# Sarsa(λ)  — backward view (eligibility traces)
# ---------------------------------------------------------------------------
def sarsa_lambda(
    env,
    lam: float      = 0.9,
    n_episodes: int = 10_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    replacing_traces: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, QValueFunction]:
    """
    Backward-view Sarsa(λ) with eligibility traces.

    δ_t = R_{t+1} + γ Q(S_{t+1},A_{t+1}) − Q(S_t,A_t)
    e_t(s,a) = γλ e_{t-1}(s,a) + 1[S_t=s, A_t=a]   (accumulating)
             = γλ e_{t-1}(s,a) (if already visited) | 1 (current)  (replacing)
    Q(s,a) ← Q(s,a) + α δ_t e_t(s,a)  ∀s,a

    Parameters
    ----------
    replacing_traces : bool
        True  → replacing traces (clips eligibility at 1, reduces variance).
        False → accumulating traces (standard).
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    n_a = env.action_space.n
    Q   = QValueFunction(n_s, n_a, init=0.0)

    def eps_greedy(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_a))
        return int(np.argmax(Q[s]))

    for _ in range(n_episodes):
        s, _ = env.reset()
        a    = eps_greedy(int(s))
        e    = np.zeros((n_s, n_a), dtype=np.float64)
        done = False

        while not done:
            s_next, r, terminated, truncated, _ = env.step(a)
            done   = terminated or truncated
            a_next = eps_greedy(int(s_next))

            delta = r + gamma * (0.0 if done else Q[int(s_next), a_next]) - Q[int(s), a]

            if replacing_traces:
                e[int(s), a] = 1.0
            else:
                e[int(s), a] += 1.0

            Q.values += alpha * delta * e
            e        *= gamma * lam

            s, a = s_next, a_next

    pi = Policy.epsilon_greedy_from_q(Q, epsilon)
    return pi, Q

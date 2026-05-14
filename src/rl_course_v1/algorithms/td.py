"""
Temporal Difference learning algorithms.

Implements:
  - td0_prediction           : TD(0) policy evaluation → V^π
  - n_step_td_prediction     : forward-view TD(n) → V^π
  - td_lambda_prediction     : backward-view TD(λ) via eligibility traces → V^π
  - n_step_td_control        : forward-view TD(n) + greedy improvement → π*
  - td_lambda_control        : backward-view TD(λ) + greedy improvement → π*

All methods are on-policy and operate on V(s) (state values).
For Q-value variants see sarsa.py.

Reference:
  Sutton & Barto Ch. 6–7, 12 (2018).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from rl_course_v1.mdp.policy import Policy, ValueFunction


# ---------------------------------------------------------------------------
# TD(0) — one-step prediction
# ---------------------------------------------------------------------------
def td0_prediction(
    env,
    pi: Policy,
    n_episodes: int = 5_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    rng: Optional[np.random.Generator] = None,
) -> ValueFunction:
    """
    TD(0) policy evaluation.
    V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]
    """
    rng  = rng or np.random.default_rng()
    n_s  = env.observation_space.n
    V    = ValueFunction(n_s)

    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a             = pi.act(int(s), rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            done          = terminated or truncated
            target        = r + gamma * (0.0 if done else V[int(s_next)])
            V[int(s)]    += alpha * (target - V[int(s)])
            s             = s_next

    return V


# ---------------------------------------------------------------------------
# n-step TD prediction  (forward view)
# ---------------------------------------------------------------------------
def n_step_td_prediction(
    env,
    pi: Policy,
    n: int          = 4,
    n_episodes: int = 5_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    rng: Optional[np.random.Generator] = None,
) -> ValueFunction:
    """
    n-step TD policy evaluation (forward view / bootstrapped returns).

    G_{t:t+n} = R_{t+1} + γ R_{t+2} + … + γ^{n-1} R_{t+n} + γ^n V(S_{t+n})
    V(S_t) ← V(S_t) + α [G_{t:t+n} - V(S_t)]
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    V   = ValueFunction(n_s)

    for _ in range(n_episodes):
        states  = []
        rewards = []
        s, _    = env.reset()
        states.append(int(s))
        done    = False
        T       = float("inf")
        t       = 0

        while True:
            if t < T:
                a = pi.act(states[-1], rng)
                s_next, r, terminated, truncated, _ = env.step(a)
                rewards.append(float(r))
                states.append(int(s_next))
                if terminated or truncated:
                    T = t + 1

            tau = t - n + 1          # state being updated
            if tau >= 0:
                # Build n-step return
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, min(tau + n, int(T)) + 1)
                )
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])

            if tau == T - 1:
                break
            t += 1

    return V


# ---------------------------------------------------------------------------
# TD(λ) prediction — backward view (eligibility traces)
# ---------------------------------------------------------------------------
def td_lambda_prediction(
    env,
    pi: Policy,
    lam: float      = 0.9,
    n_episodes: int = 5_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    rng: Optional[np.random.Generator] = None,
) -> ValueFunction:
    """
    Backward-view TD(λ) with accumulating eligibility traces.

    δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
    e_t(s) = γλ e_{t-1}(s) + 1[S_t = s]
    V(s) ← V(s) + α δ_t e_t(s)  ∀s
    """
    rng  = rng or np.random.default_rng()
    n_s  = env.observation_space.n
    V    = ValueFunction(n_s)

    for _ in range(n_episodes):
        s, _ = env.reset()
        e    = np.zeros(n_s, dtype=np.float64)
        done = False

        while not done:
            a = pi.act(int(s), rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            done   = terminated or truncated
            delta  = r + gamma * (0.0 if done else V[int(s_next)]) - V[int(s)]
            e     *= gamma * lam
            e[int(s)] += 1.0            # accumulating trace
            V.values  += alpha * delta * e
            s          = s_next

    return V


# ---------------------------------------------------------------------------
# n-step TD control  (forward view + greedy improvement)
# ---------------------------------------------------------------------------
def n_step_td_control(
    env,
    n: int          = 4,
    n_episodes: int = 10_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, ValueFunction]:
    """
    Forward-view TD(n) + ε-greedy policy improvement on expected state values.
    Uses a tabular V(s); greedy actions are chosen by one-step lookahead
    (requires the environment to expose env.P like Toy-Text envs).
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    n_a = env.action_space.n
    V   = ValueFunction(n_s)

    # Greedy policy via one-step Bellman lookahead using env.P
    P_dict = env.unwrapped.P if hasattr(env, 'unwrapped') else env.P

    def greedy_action(s: int) -> int:
        q_vals = np.zeros(n_a)
        for a in range(n_a):
            for prob, s_next, r, done in P_dict[s][a]:
                q_vals[a] += prob * (r + gamma * (0.0 if done else V[s_next]))
        return int(np.argmax(q_vals))

    pi = Policy.uniform(n_s, n_a)

    for _ in range(n_episodes):
        states  = []
        rewards = []
        s, _    = env.reset()
        states.append(int(s))
        done    = False
        T       = float("inf")
        t       = 0

        while True:
            if t < T:
                # ε-greedy action
                if rng.random() < epsilon:
                    a = int(rng.integers(n_a))
                else:
                    a = greedy_action(states[-1])
                s_next, r, terminated, truncated, _ = env.step(a)
                rewards.append(float(r))
                states.append(int(s_next))
                if terminated or truncated:
                    T = t + 1

            tau = t - n + 1
            if tau >= 0:
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, min(tau + n, int(T)) + 1)
                )
                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])

            if tau == T - 1:
                break
            t += 1

    # Derive final policy
    for s in range(n_s):
        a_star = greedy_action(s)
        pi.probs[s, :] = epsilon / n_a
        pi.probs[s, a_star] += 1.0 - epsilon

    return pi, V


# ---------------------------------------------------------------------------
# TD(λ) control — backward view + greedy improvement
# ---------------------------------------------------------------------------
def td_lambda_control(
    env,
    lam: float      = 0.9,
    n_episodes: int = 10_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, ValueFunction]:
    """
    Backward-view TD(λ) + ε-greedy improvement on expected state values.
    Requires env.P for one-step greedy action selection.
    """
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    n_a = env.action_space.n
    V   = ValueFunction(n_s)

    P_dict = env.unwrapped.P if hasattr(env, 'unwrapped') else env.P

    def greedy_action(s: int) -> int:
        q_vals = np.zeros(n_a)
        for a in range(n_a):
            for prob, s_next, r, done in P_dict[s][a]:
                q_vals[a] += prob * (r + gamma * (0.0 if done else V[s_next]))
        return int(np.argmax(q_vals))

    pi = Policy.uniform(n_s, n_a)

    for _ in range(n_episodes):
        s, _ = env.reset()
        e    = np.zeros(n_s, dtype=np.float64)
        done = False

        while not done:
            if rng.random() < epsilon:
                a = int(rng.integers(n_a))
            else:
                a = greedy_action(int(s))

            s_next, r, terminated, truncated, _ = env.step(a)
            done  = terminated or truncated
            delta = r + gamma * (0.0 if done else V[int(s_next)]) - V[int(s)]
            e    *= gamma * lam
            e[int(s)] += 1.0
            V.values  += alpha * delta * e
            s          = s_next

    # Derive final policy
    for s in range(n_s):
        a_star = greedy_action(s)
        pi.probs[s, :] = epsilon / n_a
        pi.probs[s, a_star] += 1.0 - epsilon

    return pi, V

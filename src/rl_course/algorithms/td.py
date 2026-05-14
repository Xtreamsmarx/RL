"""
Temporal Difference learning algorithms.

Implements:
  - td0_prediction           : TD(0) policy evaluation â†’ V^Ï€
  - n_step_td_prediction     : forward-view TD(n) â†’ V^Ï€
    - td_lambda_prediction_forward : forward-view TD(Î») â†’ V^Ï€
  - td_lambda_prediction     : backward-view TD(Î») via eligibility traces â†’ V^Ï€
  - n_step_td_control        : forward-view TD(n) + greedy improvement â†’ Ï€*
  - td_lambda_control        : backward-view TD(Î») + greedy improvement â†’ Ï€*

All methods are on-policy and operate on V(s) (state values).
For Q-value variants see sarsa.py.

Reference:
  Sutton & Barto Ch. 6â€“7, 12 (2018).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from rl_course.mdp.policy import Policy, ValueFunction


def _lambda_return_from_t(
    rewards: list[float],
    states: list[int],
    V: ValueFunction,
    t: int,
    T: int,
    lam: float,
    gamma: float,
) -> float:
    """Compute the forward-view lambda-return for one timestep t."""
    g_lambda = 0.0
    weight = 1.0
    horizon = T - t

    for n in range(1, horizon):
        g_n = 0.0
        for k in range(1, n + 1):
            g_n += (gamma ** (k - 1)) * rewards[t + k]
        g_n += (gamma ** n) * V[states[t + n]]
        g_lambda += (1.0 - lam) * weight * g_n
        weight *= lam

    g_mc = 0.0
    for k in range(1, horizon + 1):
        g_mc += (gamma ** (k - 1)) * rewards[t + k]
    g_lambda += weight * g_mc
    return g_lambda


# ---------------------------------------------------------------------------
# TD(0) â€” one-step prediction
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
    V(S_t) â† V(S_t) + Î± [R_{t+1} + Î³ V(S_{t+1}) - V(S_t)]
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

    G_{t:t+n} = R_{t+1} + Î³ R_{t+2} + â€¦ + Î³^{n-1} R_{t+n} + Î³^n V(S_{t+n})
    V(S_t) â† V(S_t) + Î± [G_{t:t+n} - V(S_t)]
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
# TD(Î») prediction â€” backward view (eligibility traces)
# ---------------------------------------------------------------------------
def td_lambda_prediction(
    env,
    pi: Policy,
    lam: float      = 0.9,
    n_episodes: int = 5_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    trace_cutoff: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> ValueFunction:
    """
    Backward-view TD(Î») with accumulating eligibility traces.

    Î´_t = R_{t+1} + Î³ V(S_{t+1}) - V(S_t)
    e_t(s) = Î³Î» e_{t-1}(s) + 1[S_t = s]
    V(s) â† V(s) + Î± Î´_t e_t(s)  âˆ€s

    Parameters
    ----------
    trace_cutoff : Optional[int]
        Optional practical n-cutoff variant. If set, traces with tiny
        weights below (gamma * lam)^trace_cutoff are truncated to zero.
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
            if trace_cutoff is not None:
                threshold = (gamma * lam) ** max(int(trace_cutoff), 1)
                e[e < threshold] = 0.0
            V.values  += alpha * delta * e
            s          = s_next

    return V


def td_lambda_prediction_forward(
    env,
    pi: Policy,
    lam: float      = 0.9,
    n_episodes: int = 5_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    rng: Optional[np.random.Generator] = None,
) -> ValueFunction:
    """Forward-view TD(lambda) policy evaluation via lambda-returns."""
    rng = rng or np.random.default_rng()
    n_s = env.observation_space.n
    V = ValueFunction(n_s)

    for _ in range(n_episodes):
        states: list[int] = []
        rewards: list[float] = [0.0]
        s, _ = env.reset()
        states.append(int(s))

        done = False
        while not done:
            a = pi.act(states[-1], rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            rewards.append(float(r))
            states.append(int(s_next))
            done = terminated or truncated

        T = len(states) - 1
        for t in range(T):
            g_lam = _lambda_return_from_t(
                rewards=rewards,
                states=states,
                V=V,
                t=t,
                T=T,
                lam=lam,
                gamma=gamma,
            )
            s_t = states[t]
            V[s_t] += alpha * (g_lam - V[s_t])

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
    Forward-view TD(n) + Îµ-greedy policy improvement on expected state values.
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
                # Îµ-greedy action
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
                # Avoid overflow when T is infinity and bounds-check array access
                if T == float("inf"):
                    end_idx = min(tau + n, len(rewards) - 1)
                else:
                    end_idx = min(tau + n, int(T), len(rewards) - 1)
                G = sum(
                    gamma ** (i - tau - 1) * rewards[i]
                    for i in range(tau + 1, end_idx + 1)
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
# TD(Î») control â€” backward view + greedy improvement
# ---------------------------------------------------------------------------
def td_lambda_control(
    env,
    lam: float      = 0.9,
    n_episodes: int = 10_000,
    alpha: float    = 0.1,
    gamma: float    = 0.99,
    epsilon: float  = 0.1,
    trace_cutoff: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Policy, ValueFunction]:
    """
    Backward-view TD(Î») + Îµ-greedy improvement on expected state values.
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
            
            # Optional trace cutoff for numerical stability
            if trace_cutoff is not None:
                threshold = (gamma * lam) ** max(int(trace_cutoff), 1)
                e[e < threshold] = 0.0
            
            V.values  += alpha * delta * e
            s          = s_next

    # Derive final policy
    for s in range(n_s):
        a_star = greedy_action(s)
        pi.probs[s, :] = epsilon / n_a
        pi.probs[s, a_star] += 1.0 - epsilon

    return pi, V


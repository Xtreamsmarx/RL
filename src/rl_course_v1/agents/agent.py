"""
Agent framework.

AgentConfig  : dataclass of all hyperparameters
TabularAgent : wraps an algorithm + policy + value function.
               Supports train() and evaluate() given a subtask.
"""

from __future__ import annotations

import time
import cloudpickle
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from rl_course_v1.mdp.base   import TabularMDP, Subtask
from rl_course_v1.mdp.policy import Policy, ValueFunction, QValueFunction


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """All tunable hyperparameters for a TabularAgent run."""

    # Algorithm selection
    algorithm: str = "q_learning"   # one of the keys in ALGORITHM_REGISTRY

    # Shared
    gamma:      float = 0.99
    alpha:      float = 0.1
    epsilon:    float = 0.1
    n_episodes: int   = 10_000
    seed:       int   = 42

    # TD(n) / Sarsa(n)
    n_steps: int = 4

    # TD(λ) / Sarsa(λ)
    lam:              float = 0.9
    replacing_traces: bool  = True

    # Checkpoint
    checkpoint_dir: str = "models"


# ---------------------------------------------------------------------------
# Algorithm registry  (algorithm name → callable)
# ---------------------------------------------------------------------------
def _build_registry() -> dict[str, Callable]:
    from rl_course_v1.algorithms import (
        policy_iteration, q_policy_iteration, value_iteration,
        mc_control_epsilon_greedy, mc_control_exploring_starts,
        td0_prediction, n_step_td_control, td_lambda_control,
        sarsa_zero, n_step_sarsa, sarsa_lambda,
        q_learning,
    )
    return {
        "policy_iteration":        policy_iteration,
        "q_policy_iteration":      q_policy_iteration,
        "value_iteration":         value_iteration,
        "mc_epsilon_greedy":       mc_control_epsilon_greedy,
        "mc_exploring_starts":     mc_control_exploring_starts,
        "td0":                     td0_prediction,
        "n_step_td":               n_step_td_control,
        "td_lambda":               td_lambda_control,
        "sarsa":                   sarsa_zero,
        "n_step_sarsa":            n_step_sarsa,
        "sarsa_lambda":            sarsa_lambda,
        "q_learning":              q_learning,
    }


# ---------------------------------------------------------------------------
# TabularAgent
# ---------------------------------------------------------------------------
class TabularAgent:
    """
    Wraps an RL algorithm so that train() / evaluate() can be called
    uniformly given a subtask, an environment, and an AgentConfig.

    Usage
    -----
    >>> agent = TabularAgent(config)
    >>> agent.train(env, subtask)
    >>> mean_ret = agent.evaluate(env, n_episodes=200)
    >>> agent.save("models/best.pkl")
    """

    def __init__(self, config: AgentConfig):
        self.config    = config
        self.policy:   Optional[Policy]         = None
        self.V:        Optional[ValueFunction]  = None
        self.Q:        Optional[QValueFunction] = None
        self._registry = _build_registry()
        self._rng      = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, env, subtask: Optional[Subtask] = None) -> "TabularAgent":
        """
        Train the agent.

        Parameters
        ----------
        env     : Gymnasium environment (must expose env.P for DP methods)
        subtask : optional Subtask to restrict training state space
                  (currently used for logging / curriculum purposes)
        """
        alg  = self.config.algorithm
        cfg  = self.config
        fn   = self._registry.get(alg)
        if fn is None:
            raise ValueError(
                f"Unknown algorithm '{alg}'. "
                f"Available: {sorted(self._registry)}"
            )

        print(f"[TabularAgent] Training with algorithm='{alg}' "
              f"for {cfg.n_episodes} episodes …")
        t0 = time.perf_counter()

        # DP methods need the TabularMDP wrapper, not the raw env
        if alg in ("policy_iteration", "q_policy_iteration", "value_iteration"):
            mdp = TabularMDP(env, gamma=cfg.gamma)
            result = fn(mdp)
        elif alg in ("td0",):
            from rl_course_v1.mdp.policy import Policy
            pi  = Policy.uniform(env.observation_space.n, env.action_space.n)
            result = (fn(env, pi,
                         n_episodes=cfg.n_episodes,
                         alpha=cfg.alpha,
                         gamma=cfg.gamma,
                         rng=self._rng),)
        elif alg in ("n_step_td",):
            result = fn(env,
                        n=cfg.n_steps,
                        n_episodes=cfg.n_episodes,
                        alpha=cfg.alpha,
                        gamma=cfg.gamma,
                        epsilon=cfg.epsilon,
                        rng=self._rng)
        elif alg in ("td_lambda",):
            result = fn(env,
                        lam=cfg.lam,
                        n_episodes=cfg.n_episodes,
                        alpha=cfg.alpha,
                        gamma=cfg.gamma,
                        epsilon=cfg.epsilon,
                        rng=self._rng)
        elif alg in ("sarsa",):
            result = fn(env,
                        n_episodes=cfg.n_episodes,
                        alpha=cfg.alpha,
                        gamma=cfg.gamma,
                        epsilon=cfg.epsilon,
                        rng=self._rng)
        elif alg in ("n_step_sarsa",):
            result = fn(env,
                        n=cfg.n_steps,
                        n_episodes=cfg.n_episodes,
                        alpha=cfg.alpha,
                        gamma=cfg.gamma,
                        epsilon=cfg.epsilon,
                        rng=self._rng)
        elif alg in ("sarsa_lambda",):
            result = fn(env,
                        lam=cfg.lam,
                        n_episodes=cfg.n_episodes,
                        alpha=cfg.alpha,
                        gamma=cfg.gamma,
                        epsilon=cfg.epsilon,
                        replacing_traces=cfg.replacing_traces,
                        rng=self._rng)
        elif alg in ("mc_epsilon_greedy",):
            result = fn(env,
                        n_episodes=cfg.n_episodes,
                        gamma=cfg.gamma,
                        epsilon=cfg.epsilon,
                        rng=self._rng)
        elif alg in ("mc_exploring_starts",):
            result = fn(env,
                        n_episodes=cfg.n_episodes,
                        gamma=cfg.gamma,
                        rng=self._rng)
        else:
            result = fn(env,
                        n_episodes=cfg.n_episodes,
                        alpha=cfg.alpha,
                        gamma=cfg.gamma,
                        epsilon=cfg.epsilon,
                        rng=self._rng)

        elapsed = time.perf_counter() - t0

        # Unpack (policy, value) or just policy
        if isinstance(result, tuple):
            if len(result) == 2:
                self.policy, vq = result
                if isinstance(vq, ValueFunction):
                    self.V = vq
                elif isinstance(vq, QValueFunction):
                    self.Q = vq
            else:
                self.policy = result[0]
        else:
            # Prediction-only path returns a value function
            self.V = result

        print(f"[TabularAgent] Done in {elapsed:.2f}s.")
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        env,
        n_episodes: int = 200,
        max_steps:  int = 200,
        render:     bool = False,
    ) -> float:
        """
        Evaluate the current policy greedily.

        Returns
        -------
        mean_return : float  (undiscounted episode return)
        """
        if self.policy is None:
            raise RuntimeError("Agent has not been trained yet.")

        returns = []
        for _ in range(n_episodes):
            s, _ = env.reset()
            G    = 0.0
            for _ in range(max_steps):
                a = self.policy.deterministic_action(int(s))
                s, r, terminated, truncated, _ = env.step(a)
                G += r
                if render:
                    env.render()
                if terminated or truncated:
                    break
            returns.append(G)

        mean_ret = float(np.mean(returns))
        std_ret  = float(np.std(returns))
        print(f"[TabularAgent] Eval over {n_episodes} eps: "
              f"mean={mean_ret:.4f}  std={std_ret:.4f}")
        return mean_ret

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        """Persist policy + value function to disk via cloudpickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "policy": self.policy,
            "V":      self.V,
            "Q":      self.Q,
        }
        with open(path, "wb") as f:
            cloudpickle.dump(payload, f)
        print(f"[TabularAgent] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TabularAgent":
        """Load a previously saved agent."""
        with open(path, "rb") as f:
            payload = cloudpickle.load(f)
        agent        = cls(payload["config"])
        agent.policy = payload["policy"]
        agent.V      = payload["V"]
        agent.Q      = payload["Q"]
        print(f"[TabularAgent] Loaded from {path}")
        return agent

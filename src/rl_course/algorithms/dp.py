"""
Dynamic Programming algorithms for tabular MDPs.

Implements (all model-based, assuming full knowledge of P and R):
  - policy_evaluation       : iterative V^Ï€ via Bellman expectation
  - policy_improvement      : one-step greedy improvement on V or Q
  - policy_iteration        : PE + PI loop â†’ Ï€*
  - value_iteration         : Bellman optimality operator â†’ V*
  - q_policy_evaluation     : Q^Ï€ via Bellman expectation
  - q_policy_iteration      : Q-based PI loop â†’ Ï€*

Reference:
  Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed.,
  Chapters 4.1â€“4.4 (MIT Press, 2018).  http://incompleteideas.net/book/the-book.html
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from rl_course.mdp.base   import TabularMDP
from rl_course.mdp.policy import Policy, ValueFunction, QValueFunction


# ---------------------------------------------------------------------------
# Policy Evaluation  (Bellman expectation, iterative)
# ---------------------------------------------------------------------------
def policy_evaluation(
    mdp: TabularMDP,
    pi: Policy,
    theta: float = 1e-8,
    max_iter: int = 10_000,
) -> ValueFunction:
    """
    Compute V^Ï€ via iterative Bellman expectation backup.

    V(s) â† Î£_a Ï€(a|s) [R(s,a) + Î³ Î£_{s'} P(s'|s,a) V(s')]

    Parameters
    ----------
    mdp     : TabularMDP
    pi      : Policy  (stochastic or deterministic)
    theta   : convergence threshold on max |Î”V|
    max_iter: safety cap on iterations

    Returns
    -------
    V : ValueFunction
    """
    V = ValueFunction(mdp.n_states, init=0.0)

    # Vectorised: R_Ï€(s) = Î£_a Ï€(a|s) R(s,a)   shape (S,)
    R_pi = np.einsum("sa,sa->s", pi.probs, mdp.R_matrix)
    # P_Ï€(s,s') = Î£_a Ï€(a|s) P(s'|s,a)          shape (S,S)
    P_pi = np.einsum("sa,saj->sj", pi.probs, mdp.P_matrix)

    for _ in range(max_iter):
        V_new = R_pi + mdp.gamma * P_pi @ V.values
        delta = np.max(np.abs(V_new - V.values))
        V.values = V_new
        if delta < theta:
            break

    return V


# ---------------------------------------------------------------------------
# Q-Policy Evaluation  (Bellman expectation on Q)
# ---------------------------------------------------------------------------
def q_policy_evaluation(
    mdp: TabularMDP,
    pi: Policy,
    theta: float = 1e-8,
    max_iter: int = 10_000,
) -> QValueFunction:
    """
    Compute Q^Ï€ via iterative Bellman expectation backup.

    Q(s,a) â† R(s,a) + Î³ Î£_{s'} P(s'|s,a) Î£_{a'} Ï€(a'|s') Q(s',a')

    Returns
    -------
    Q : QValueFunction
    """
    Q = QValueFunction(mdp.n_states, mdp.n_actions, init=0.0)

    for _ in range(max_iter):
        # V_Ï€(s') = Î£_{a'} Ï€(a'|s') Q(s',a')   shape (S,)
        V_pi = np.einsum("sa,sa->s", pi.probs, Q.values)
        # Q_new(s,a) = R(s,a) + Î³ Î£_{s'} P(s'|s,a) V_Ï€(s')
        Q_new = mdp.R_matrix + mdp.gamma * np.einsum("saj,j->sa", mdp.P_matrix, V_pi)
        delta = np.max(np.abs(Q_new - Q.values))
        Q.values = Q_new
        if delta < theta:
            break

    return Q


# ---------------------------------------------------------------------------
# Policy Improvement
# ---------------------------------------------------------------------------
def policy_improvement_v(mdp: TabularMDP, V: ValueFunction) -> Policy:
    """
    One-step greedy policy improvement from V(s).

    Ï€'(s) = argmax_a [R(s,a) + Î³ Î£_{s'} P(s'|s,a) V(s')]
    """
    return Policy.greedy_from_v(V, mdp)


def policy_improvement_q(Q: QValueFunction) -> Policy:
    """
    One-step greedy policy improvement from Q(s,a).

    Ï€'(s) = argmax_a Q(s,a)
    """
    return Policy.greedy_from_q(Q)


# ---------------------------------------------------------------------------
# Policy Iteration  (V-based)
# ---------------------------------------------------------------------------
def policy_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_sweeps: int = 1_000,
) -> Tuple[Policy, ValueFunction]:
    """
    Policy Iteration: alternate Policy Evaluation and Policy Improvement
    until the policy is stable.

    Returns
    -------
    (Ï€*, V^{Ï€*})
    """
    pi = Policy.uniform(mdp.n_states, mdp.n_actions)

    for _ in range(max_sweeps):
        V  = policy_evaluation(mdp, pi, theta=theta)
        pi_new = policy_improvement_v(mdp, V)

        # Check stability: policies equal iff greedy actions agree everywhere
        if np.array_equal(
            np.argmax(pi.probs, axis=1),
            np.argmax(pi_new.probs, axis=1),
        ):
            break
        pi = pi_new

    return pi, V


# ---------------------------------------------------------------------------
# Q-Policy Iteration  (Q-based)
# ---------------------------------------------------------------------------
def q_policy_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_sweeps: int = 1_000,
) -> Tuple[Policy, QValueFunction]:
    """
    Policy Iteration operating entirely on Q-values.
    Ï€'(s) = argmax_a Q^Ï€(s,a)

    Returns
    -------
    (Ï€*, Q^{Ï€*})
    """
    pi = Policy.uniform(mdp.n_states, mdp.n_actions)

    for _ in range(max_sweeps):
        Q = q_policy_evaluation(mdp, pi, theta=theta)
        pi_new = policy_improvement_q(Q)

        if np.array_equal(
            np.argmax(pi.probs, axis=1),
            np.argmax(pi_new.probs, axis=1),
        ):
            break
        pi = pi_new

    return pi, Q


# ---------------------------------------------------------------------------
# Value Iteration  (Bellman optimality operator)
# ---------------------------------------------------------------------------
def value_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_iter: int = 10_000,
) -> Tuple[Policy, ValueFunction]:
    """
    Value Iteration: apply Bellman optimality backup until convergence.

    V(s) â† max_a [R(s,a) + Î³ Î£_{s'} P(s'|s,a) V(s')]

    Returns
    -------
    (Ï€*, V*)
    """
    V = ValueFunction(mdp.n_states, init=0.0)

    for _ in range(max_iter):
        # Q(s,a) = R(s,a) + Î³ Î£_{s'} P(s'|s,a) V(s')   shape (S,A)
        Q = mdp.R_matrix + mdp.gamma * np.einsum("saj,j->sa", mdp.P_matrix, V.values)
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V_new - V.values))
        V.values = V_new
        if delta < theta:
            break

    # Extract greedy policy from converged Q
    Q_star = QValueFunction(mdp.n_states, mdp.n_actions)
    Q_star.values = mdp.R_matrix + mdp.gamma * np.einsum(
        "saj,j->sa", mdp.P_matrix, V.values
    )
    pi_star = policy_improvement_q(Q_star)
    return pi_star, V


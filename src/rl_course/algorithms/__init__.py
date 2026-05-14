"""rl_course.algorithms package."""
from rl_course.algorithms.dp          import (
    policy_evaluation, q_policy_evaluation,
    policy_improvement_v, policy_improvement_q,
    policy_iteration, q_policy_iteration, value_iteration,
)
from rl_course.algorithms.monte_carlo import (
    mc_prediction_first_visit, mc_prediction_every_visit,
    mc_control_epsilon_greedy, mc_control_exploring_starts,
)
from rl_course.algorithms.td          import (
    td0_prediction, n_step_td_prediction, td_lambda_prediction,
    td_lambda_prediction_forward,
    n_step_td_control, td_lambda_control,
)
from rl_course.algorithms.sarsa       import (
    sarsa_zero,
    n_step_sarsa,
    sarsa_lambda,
    sarsa_lambda_forward,
)
from rl_course.algorithms.q_learning  import q_learning

__all__ = [
    # DP
    "policy_evaluation", "q_policy_evaluation",
    "policy_improvement_v", "policy_improvement_q",
    "policy_iteration", "q_policy_iteration", "value_iteration",
    # MC
    "mc_prediction_first_visit", "mc_prediction_every_visit",
    "mc_control_epsilon_greedy", "mc_control_exploring_starts",
    # TD
    "td0_prediction", "n_step_td_prediction", "td_lambda_prediction",
    "td_lambda_prediction_forward",
    "n_step_td_control", "td_lambda_control",
    # Sarsa
    "sarsa_zero", "n_step_sarsa", "sarsa_lambda", "sarsa_lambda_forward",
    # Q-learning
    "q_learning",
]


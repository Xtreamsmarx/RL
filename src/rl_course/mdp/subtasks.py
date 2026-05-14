"""
Subtask factory for standard Gymnasium environments.

Subtasks partition the MDP state space into named sub-problems,
each with its own goal set and local transition structure.
"""

from __future__ import annotations

from rl_course.mdp.base import TabularMDP, Subtask


def make_frozenlake_subtasks(mdp: TabularMDP) -> list[Subtask]:
    """
    4Ã—4 FrozenLake-v1 subtasks.

    Layout (row-major, 0-indexed):
        S F F F
        F H F H
        F F F H
        H F F G

    Subtasks defined by spatial quadrants:
      - top_left  : states  0-7   goal = [5] (avoid holes, reach mid)
      - bottom_right : states 8-15 goal = [15] (reach G)
    """
    assert mdp.n_states == 16, "Expected 4Ã—4 FrozenLake (16 states)."

    top_left = Subtask(
        name="top_left",
        state_ids=list(range(8)),
        goal_states=[5],          # arbitrary sub-goal: state 5 (F, reachable)
        mdp=mdp,
    )

    bottom_right = Subtask(
        name="bottom_right",
        state_ids=list(range(8, 16)),
        goal_states=[15],         # actual goal
        mdp=mdp,
    )

    return [top_left, bottom_right]


def make_full_task(mdp: TabularMDP) -> Subtask:
    """Single subtask covering the entire state space (full task)."""
    goal = [s for s in range(mdp.n_states)
            if any(r > 0 for a in range(mdp.n_actions)
                   for _, _, r, _ in mdp.P_raw[s][a])]
    return Subtask(
        name="full_task",
        state_ids=list(range(mdp.n_states)),
        goal_states=goal,
        mdp=mdp,
    )


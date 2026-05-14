"""
train.py — CLI entry point for training and evaluating tabular RL agents.

Usage examples
--------------
# Train Q-learning and save best policy:
  python train.py --algorithm q_learning --n_episodes 20000 --save

# Evaluate a saved policy:
  python train.py --load models/best.pkl --eval_only

# Policy iteration (model-based DP):
  python train.py --algorithm policy_iteration --save

# Sarsa(λ) with custom hyperparameters:
  python train.py --algorithm sarsa_lambda --lam 0.8 --alpha 0.05 --epsilon 0.1 --n_episodes 30000

Run `python train.py --help` for the full argument list.
"""

import argparse
import gymnasium as gym

from rl_course_v1.agents.agent import TabularAgent, AgentConfig


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train or evaluate a tabular RL agent on FrozenLake-v1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Algorithm & mode
    p.add_argument(
        "--algorithm", "-a",
        default="q_learning",
        choices=[
            "policy_iteration", "q_policy_iteration", "value_iteration",
            "mc_epsilon_greedy", "mc_exploring_starts",
            "td0", "n_step_td", "td_lambda",
            "sarsa", "n_step_sarsa", "sarsa_lambda",
            "q_learning",
        ],
        help="RL algorithm to use.",
    )
    p.add_argument(
        "--env", default="FrozenLake-v1",
        help="Gymnasium environment id.",
    )
    p.add_argument(
        "--is_slippery", action="store_true", default=True,
        help="Use stochastic (slippery) FrozenLake.",
    )

    # Hyperparameters
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--alpha",      type=float, default=0.1)
    p.add_argument("--epsilon",    type=float, default=0.1)
    p.add_argument("--n_episodes", type=int,   default=20_000)
    p.add_argument("--n_steps",    type=int,   default=4,
                   help="n for n-step methods.")
    p.add_argument("--lam",        type=float, default=0.9,
                   help="λ for TD(λ)/Sarsa(λ).")
    p.add_argument("--seed",       type=int,   default=42)

    # Eval
    p.add_argument("--eval_episodes", type=int, default=500)
    p.add_argument("--eval_only",     action="store_true",
                   help="Skip training; load and evaluate only.")

    # Save / load
    p.add_argument("--save",  action="store_true",
                   help="Save the trained agent to --checkpoint.")
    p.add_argument("--checkpoint", default="models/best.pkl",
                   help="Path to save/load the agent.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    env_kwargs = {}
    if args.env == "FrozenLake-v1":
        env_kwargs["is_slippery"] = args.is_slippery

    env = gym.make(args.env, **env_kwargs)

    if args.eval_only:
        agent = TabularAgent.load(args.checkpoint)
    else:
        config = AgentConfig(
            algorithm    = args.algorithm,
            gamma        = args.gamma,
            alpha        = args.alpha,
            epsilon      = args.epsilon,
            n_episodes   = args.n_episodes,
            n_steps      = args.n_steps,
            lam          = args.lam,
            seed         = args.seed,
            checkpoint_dir = "models",
        )
        agent = TabularAgent(config)
        agent.train(env)

        if args.save:
            agent.save(args.checkpoint)

    agent.evaluate(env, n_episodes=args.eval_episodes)
    env.close()


if __name__ == "__main__":
    main()

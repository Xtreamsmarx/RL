"""
train.py â€” CLI entry point for training and evaluating tabular RL agents.

Usage examples
--------------
# Train Q-learning and save best policy:
  python train.py --algorithm q_learning --n_episodes 20000 --save

# Evaluate a saved policy:
  python train.py --load models/best.pkl --eval_only

# Policy iteration (model-based DP):
  python train.py --algorithm policy_iteration --save

# Sarsa(Î») with custom hyperparameters:
  python train.py --algorithm sarsa_lambda --lam 0.8 --alpha 0.05 --epsilon 0.1 --n_episodes 30000

Run `python train.py --help` for the full argument list.
"""

import argparse
import gymnasium as gym

from rl_course.agents.agent import TabularAgent, AgentConfig
from rl_course.utils.results import (
    append_csv_row,
    ensure_output_dirs,
    save_line_plot,
    utc_timestamp,
    update_overview_visualizations,
)


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
            "td0", "td_lambda_forward", "n_step_td", "td_lambda",
            "sarsa", "n_step_sarsa", "sarsa_lambda_forward", "sarsa_lambda",
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
                   help="Î» for TD(Î»)/Sarsa(Î»).")
    p.add_argument("--trace_cutoff", type=int, default=None,
                   help="Optional practical n-cutoff for backward-view traces.")
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
    run_ts = utc_timestamp()

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
            trace_cutoff = args.trace_cutoff,
            seed         = args.seed,
            checkpoint_dir = "models",
        )
        agent = TabularAgent(config)
        agent.train(env)

        if args.save:
            agent.save(args.checkpoint)

    eval_mean = None
    if agent.policy is not None:
        eval_mean = agent.evaluate(env, n_episodes=args.eval_episodes)
    else:
        print("[train.py] This algorithm produced a value function only (no policy to evaluate).")
    env.close()

    _, csv_dir, fig_dir, _ = ensure_output_dirs(result_dir="result", visualization_dir="visualization")
    fieldnames = [
        "timestamp",
        "algorithm",
        "env",
        "n_episodes",
        "eval_episodes",
        "eval_mean",
        "gamma",
        "alpha",
        "epsilon",
        "n_steps",
        "lam",
        "trace_cutoff",
        "seed",
        "checkpoint",
        "mode",
    ]
    row = {
        "timestamp": run_ts,
        "algorithm": args.algorithm,
        "env": args.env,
        "n_episodes": args.n_episodes,
        "eval_episodes": args.eval_episodes,
        "eval_mean": "" if eval_mean is None else f"{eval_mean:.6f}",
        "gamma": args.gamma,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "n_steps": args.n_steps,
        "lam": args.lam,
        "trace_cutoff": args.trace_cutoff,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "mode": "eval_only" if args.eval_only else "train_eval",
    }
    append_csv_row(csv_dir / "classical_runs.csv", fieldnames, row)
    append_csv_row(csv_dir / f"classical_run_{run_ts}.csv", fieldnames, row)

    if eval_mean is not None:
        save_line_plot(
            path=fig_dir / f"classical_eval_{args.algorithm}_{run_ts}.png",
            xs=[1.0],
            ys=[float(eval_mean)],
            title=f"Classical Eval Mean | {args.algorithm}",
            xlabel="Run",
            ylabel="Eval mean return",
        )

    update_overview_visualizations(result_dir="result", visualization_dir="visualization")


if __name__ == "__main__":
    main()


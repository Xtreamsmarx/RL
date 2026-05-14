"""CLI entry point for deep RL training (v2)."""

from __future__ import annotations

import argparse

from rl_course_v1.agents.deep_agent import DeepAgent, DeepAgentConfig
from rl_course_v1.deep import DQNConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train deep RL agents (v2).")
    p.add_argument("--algorithm", default="dqn", choices=["dqn"])
    p.add_argument("--env", default="FrozenLake-v1")
    p.add_argument("--is_slippery", action="store_true", default=True)
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--replay_capacity", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", default="checkpoints/dqn/FrozenLake-v1/default")
    p.add_argument("--replay_dir", default="data/replay/raw/dqn/FrozenLake-v1/fresh")
    return p


def main() -> None:
    args = build_parser().parse_args()

    dqn_cfg = DQNConfig(
        env_id=args.env,
        is_slippery=args.is_slippery,
        n_episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        replay_dir=args.replay_dir,
    )
    cfg = DeepAgentConfig(algorithm=args.algorithm, dqn=dqn_cfg)
    agent = DeepAgent(cfg)
    _, metrics = agent.train()
    print(f"[DeepAgent] training complete: {metrics}")


if __name__ == "__main__":
    main()

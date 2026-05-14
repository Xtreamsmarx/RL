"""Generate saliency plot for a saved DQN checkpoint."""

from __future__ import annotations

import argparse

import torch

from rl_course.networks.mlp import DiscreteQNetwork
from rl_course.utils.saliency import dqn_saliency_for_state, save_saliency_plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create DQN saliency plot for one state.")
    p.add_argument("--checkpoint", default="checkpoints/dqn/FrozenLake-v1/default/online.pt")
    p.add_argument("--n_states", type=int, default=16)
    p.add_argument("--n_actions", type=int, default=4)
    p.add_argument("--state", type=int, default=0)
    p.add_argument("--out", default="reports/figures/dqn_saliency_state0.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = DiscreteQNetwork(input_dim=args.n_states, n_actions=args.n_actions)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    saliency = dqn_saliency_for_state(model, state_index=args.state, n_states=args.n_states)
    save_saliency_plot(
        saliency,
        path=args.out,
        title=f"DQN Saliency | state={args.state}",
    )
    print(f"[saliency_dqn] saved {args.out}")


if __name__ == "__main__":
    main()


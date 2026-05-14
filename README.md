# RL Course - V2

Version 2 extends the repository from tabular RL into a mixed classical + deep RL codebase, while keeping reproducible structure for replay data, checkpoints, and diagnostics.

## Setup

```bash
pip install -e .
```

or

```bash
conda env create -f environment.yml
conda activate rl_course
pip install -e .
```

## V2 Repository Additions

```
checkpoints/
   dqn/
      FrozenLake-v1/
         default/                # online.pt, target.pt, config.json, metrics.json

data/
   replay/
      raw/
         dqn/
            FrozenLake-v1/
               fresh/              # replay snapshots (step_*.npz, latest.npz)

docs/
   technical-challenges.md     # bugs, surprises, and implementation notes

scripts/
   rotate_replay.py            # replace old replay files with newer snapshots
   saliency_dqn.py             # generate state-saliency plots from DQN checkpoint

src/rl_course/
   deep/
      dqn.py                    # DQN implementation
   networks/
      mlp.py                    # Q-network architectures
   replay/
      buffer.py                 # replay buffer with size assertions and .npz persistence
   utils/
      saliency.py               # gradient saliency tools
      rasterize_q_values.py     # tabular policy/value raster utility
```

## Classical RL Coverage

Implemented algorithms include:

1. Monte Carlo: first-visit prediction, every-visit prediction, epsilon-greedy control, exploring starts control.
2. TD(n): forward-view prediction and forward-view control.
3. TD(lambda):
    1. forward-view prediction via lambda-returns
    2. backward-view prediction with eligibility traces
    3. backward-view control with eligibility traces
4. SARSA(n): forward-view n-step SARSA.
5. SARSA(lambda):
    1. forward-view control via lambda-returns
    2. backward-view control with eligibility traces
6. Q-learning: one-step off-policy control.

Backward-view lambda methods support practical n-cutoff style trace truncation through an optional trace_cutoff parameter.

## Deep RL in V2

Currently implemented deep RL algorithm:

1. DQN for discrete-action environments (FrozenLake-v1 baseline).

DQN features included:

1. online and target networks with periodic sync.
2. epsilon-greedy exploration schedule.
3. replay buffer with hard capacity assertions.
4. replay snapshot persistence for reproducibility.
5. JSON config and metric logging per run.

### Train DQN

```bash
python train_drl.py --algorithm dqn --env FrozenLake-v1 --episodes 1500 --checkpoint_dir checkpoints/dqn/FrozenLake-v1/default --replay_dir data/replay/raw/dqn/FrozenLake-v1/fresh
```

## Replay Buffer Organization and Rotation

Replay data is organized by algorithm, task, and freshness under:

data/replay/raw/<algorithm>/<task>/<freshness>

To replace older replay with newer snapshots while respecting storage limits:

```bash
python scripts/rotate_replay.py --source_dir data/replay/raw/dqn/FrozenLake-v1/fresh --max_files 25 --max_total_mb 2048
```

## Saliency and Visualization Utilities

1. Rasterization utility for tabular Q/policy visualization: src/rl_course/utils/rasterize_q_values.py
2. DQN saliency utility: src/rl_course/utils/saliency.py
3. Script to generate saliency figures from a checkpoint:

```bash
python scripts/saliency_dqn.py --checkpoint checkpoints/dqn/FrozenLake-v1/default/online.pt --state 0 --out reports/figures/dqn_saliency_state0.png
```

## Justification for Deep RL Choice (DQN)

Environment fit is the main reason for prioritizing DQN in V2.

1. FrozenLake is discrete in state and action spaces. DQN naturally supports discrete action value learning, making it the most direct deep extension from tabular Q-learning.
2. REINFORCE and vanilla actor-critic are valid, but both introduce higher policy-gradient variance without clear advantage for this low-dimensional benchmark.
3. PPO and TRPO are robust policy-gradient families, but they are heavier optimization stacks than needed for the current environment scale.
4. DDPG, TD3, and SAC are designed primarily for continuous action control, so they are comparatively mismatched to FrozenLake.
5. DQN keeps implementation complexity moderate while still adding core deep RL engineering requirements: replay buffer, target network, checkpoint tracking, and training logs.

This makes DQN the strongest first DRL choice for this environment, while leaving room to add actor-critic style methods in future versions.

## Difficulties and Surprises

Technical obstacles and surprises are tracked in docs/technical-challenges.md.

## Core Commands

Classical training:

```bash
python train.py --algorithm q_learning --n_episodes 20000 --save --checkpoint models/best_full.pkl
```

Classical evaluation:

```bash
python train.py --eval_only --checkpoint models/best_full.pkl
```

## License

MIT


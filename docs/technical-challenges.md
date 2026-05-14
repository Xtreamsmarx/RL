# Technical Challenges

1. Packaging/import mismatch: running train.py directly initially failed when src was not on PYTHONPATH. Resolved by editable install in venv.
2. Classical lambda variants: backward-view methods existed first; forward-view lambda-return implementations were added for TD(lambda) and SARSA(lambda).
3. Replay storage growth: replay snapshots can grow quickly in iterative experiments. Added hard capacity assertions and a rotation script for bounded storage.
4. DQN stability on sparse rewards: FrozenLake rewards are sparse, so early learning is noisy. Target network updates and replay warmup were required to stabilize updates.
5. Reproducibility details: model checkpoints alone were insufficient to reproduce behavior. Added config.json and metrics.json outputs next to checkpoints.
6. Visualization usefulness: raw reward curves were not enough to interpret model behavior. Added saliency utilities and tabular rasterization helper for diagnostics.

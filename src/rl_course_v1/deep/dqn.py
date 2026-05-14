"""DQN training implementation for discrete-action environments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from rl_course_v1.networks.mlp import DiscreteQNetwork
from rl_course_v1.replay.buffer import ReplayBuffer


@dataclass
class DQNConfig:
    env_id: str = "FrozenLake-v1"
    is_slippery: bool = True
    seed: int = 42

    n_episodes: int = 1500
    max_steps: int = 200
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    warmup_steps: int = 500

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000

    replay_capacity: int = 50_000
    replay_capacity_assert: int = 250_000
    target_update_interval: int = 200
    train_every: int = 4
    hidden_dims: tuple[int, ...] = (128, 128)

    replay_save_interval: int = 10_000
    replay_dir: str = "data/replay/raw/dqn/FrozenLake-v1/fresh"
    checkpoint_dir: str = "checkpoints/dqn/FrozenLake-v1/default"


class DQNTrainer:
    def __init__(self, config: DQNConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _obs_to_vec(obs, n_states: int) -> np.ndarray:
        obs_arr = np.asarray(obs)
        if np.isscalar(obs_arr) or obs_arr.shape == ():
            vec = np.zeros((n_states,), dtype=np.float32)
            vec[int(obs_arr)] = 1.0
            return vec
        return obs_arr.astype(np.float32).reshape(-1)

    def train(self, env):
        n_actions = env.action_space.n
        n_states = env.observation_space.n
        input_dim = n_states

        online_net = DiscreteQNetwork(input_dim=input_dim, n_actions=n_actions, hidden_dims=self.cfg.hidden_dims).to(
            self.device
        )
        target_net = DiscreteQNetwork(input_dim=input_dim, n_actions=n_actions, hidden_dims=self.cfg.hidden_dims).to(
            self.device
        )
        target_net.load_state_dict(online_net.state_dict())
        optimizer = optim.Adam(online_net.parameters(), lr=self.cfg.lr)

        replay = ReplayBuffer(
            capacity=self.cfg.replay_capacity,
            state_shape=(input_dim,),
            max_assert_size=self.cfg.replay_capacity_assert,
        )

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        replay_dir = Path(self.cfg.replay_dir)
        replay_dir.mkdir(parents=True, exist_ok=True)

        epsilon = self.cfg.eps_start
        total_steps = 0
        episode_returns: list[float] = []
        losses: list[float] = []

        for episode in range(self.cfg.n_episodes):
            obs, _ = env.reset(seed=int(self.cfg.seed + episode))
            state = self._obs_to_vec(obs, n_states)
            ep_return = 0.0

            for _ in range(self.cfg.max_steps):
                total_steps += 1
                frac = min(1.0, total_steps / max(self.cfg.eps_decay_steps, 1))
                epsilon = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

                if self.rng.random() < epsilon:
                    action = int(self.rng.integers(n_actions))
                else:
                    with torch.no_grad():
                        q = online_net(torch.from_numpy(state).float().to(self.device).unsqueeze(0))
                        action = int(torch.argmax(q, dim=1).item())

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                next_state = self._obs_to_vec(next_obs, n_states)
                replay.add(state, action, reward, next_state, done)

                state = next_state
                ep_return += float(reward)

                if len(replay) >= self.cfg.warmup_steps and total_steps % self.cfg.train_every == 0:
                    batch = replay.sample(self.cfg.batch_size, self.rng)
                    s = torch.from_numpy(batch.states).float().to(self.device)
                    a = torch.from_numpy(batch.actions).long().to(self.device)
                    r = torch.from_numpy(batch.rewards).float().to(self.device)
                    ns = torch.from_numpy(batch.next_states).float().to(self.device)
                    d = torch.from_numpy(batch.dones).float().to(self.device)

                    q_sa = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q_max = target_net(ns).max(dim=1).values
                        y = r + self.cfg.gamma * (1.0 - d) * next_q_max

                    loss = F.mse_loss(q_sa, y)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                    optimizer.step()
                    losses.append(float(loss.item()))

                if total_steps % self.cfg.target_update_interval == 0:
                    target_net.load_state_dict(online_net.state_dict())

                if total_steps % self.cfg.replay_save_interval == 0 and len(replay) > 0:
                    replay_file = replay_dir / f"step_{total_steps}.npz"
                    replay.save_npz(replay_file)

                if done:
                    break

            episode_returns.append(ep_return)

            if (episode + 1) % 100 == 0:
                recent = float(np.mean(episode_returns[-100:]))
                print(
                    f"[DQN] ep={episode + 1}/{self.cfg.n_episodes} "
                    f"recent_mean_return={recent:.4f} epsilon={epsilon:.3f}"
                )

        torch.save(online_net.state_dict(), ckpt_dir / "online.pt")
        torch.save(target_net.state_dict(), ckpt_dir / "target.pt")

        metrics = {
            "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "std_return": float(np.std(episode_returns)) if episode_returns else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "episodes": self.cfg.n_episodes,
            "steps": total_steps,
            "device": str(self.device),
        }

        with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        with open(ckpt_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if len(replay) > 0:
            replay.save_npz(replay_dir / "latest.npz")

        return online_net, metrics

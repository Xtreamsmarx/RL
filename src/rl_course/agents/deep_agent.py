"""Deep RL agent wrapper for model-free DRL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym

from rl_course.deep import DQNConfig, DQNTrainer


@dataclass
class DeepAgentConfig:
    algorithm: str = "dqn"
    dqn: DQNConfig = field(default_factory=DQNConfig)


class DeepAgent:
    def __init__(self, config: DeepAgentConfig):
        self.config = config

    def train(self):
        if self.config.algorithm != "dqn":
            raise ValueError("Only algorithm='dqn' is currently supported in v2.")

        env = gym.make(self.config.dqn.env_id, is_slippery=self.config.dqn.is_slippery)
        trainer = DQNTrainer(self.config.dqn)
        model, metrics = trainer.train(env)
        env.close()
        return model, metrics


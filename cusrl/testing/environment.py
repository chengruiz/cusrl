import numpy as np
import torch

from cusrl.template.environment import Environment

__all__ = [
    "DummyNumpyEnvironment",
    "DummyTorchEnvironment",
]


class DummyNumpyEnvironment(Environment):
    def reset(self, *, indices=None, randomize_episode_progress=False):
        num_instances = self.num_instances
        if indices is not None:
            num_instances = np.zeros(num_instances)[indices].size
        return (
            np.random.randn(num_instances, self.observation_dim).astype(np.float32),
            None if self.state_dim is None else np.random.randn(num_instances, self.state_dim).astype(np.float32),
            {},
        )

    def step(self, action):
        assert isinstance(action, np.ndarray)
        return (
            np.random.randn(self.num_instances, self.observation_dim).astype(np.float32),
            None if self.state_dim is None else np.random.randn(self.num_instances, self.state_dim).astype(np.float32),
            np.random.randn(self.num_instances, self.spec.reward_dim).astype(np.float32),
            np.random.rand(self.num_instances, 1) > 0.9,
            np.zeros((self.num_instances, 1), dtype=bool),
            {},
        )


class DummyTorchEnvironment(Environment):
    def reset(self, *, indices=None, randomize_episode_progress=False):
        num_instances = self.num_instances
        if indices is not None:
            num_instances = torch.zeros(num_instances)[indices].numel()
        return (
            torch.randn(num_instances, self.observation_dim),
            None if self.state_dim is None else torch.randn(num_instances, self.state_dim),
            {},
        )

    def step(self, action):
        assert isinstance(action, torch.Tensor)
        return (
            torch.randn(self.num_instances, self.observation_dim),
            None if self.state_dim is None else torch.randn(self.num_instances, self.state_dim),
            torch.randn(self.num_instances, self.spec.reward_dim),
            torch.rand(self.num_instances, 1) > 0.9,
            torch.rand(self.num_instances, 1) > 0.9,
            {},
        )

import random

import numpy as np
import torch

import cusrl
from cusrl.template.environment import get_done_indices

__all__ = [
    "create_dummy_env",
    "run_environment_evaluation_loop",
    "test_module_consistency",
]


class DummyEnvironment(cusrl.Environment):
    def reset(self, *, indices=None):
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
            torch.zeros(self.num_instances, 1, dtype=torch.bool),
            {},
        )


class DummyNumpyEnvironment(cusrl.Environment):
    def reset(self, *, indices=None):
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


def create_random_symmetry_def(dim: int) -> cusrl.hook.symmetry.SymmetryDef:
    shuffled_indices = list(range(dim))
    random.shuffle(shuffled_indices)
    symmetry_destinations = [-1] * dim
    symmetry_flipped = [False] * dim
    for i in range(dim // 2):
        a, b = shuffled_indices[i * 2], shuffled_indices[i * 2 + 1]
        symmetry_destinations[a], symmetry_destinations[b] = b, a
        symmetry_flipped[a] = symmetry_flipped[b] = random.random() < 0.5
    if dim % 2 == 1:
        symmetry_destinations[shuffled_indices[-1]] = shuffled_indices[-1]
        symmetry_flipped[shuffled_indices[-1]] = random.random() < 0.5
    symmetry_def = cusrl.hook.symmetry.SymmetryDef(symmetry_destinations, symmetry_flipped)
    dummy_input = torch.arange(dim, dtype=torch.float32)
    mirrored_input = symmetry_def(dummy_input)
    mirrored_mirrored_input = symmetry_def(mirrored_input)
    assert torch.allclose(mirrored_mirrored_input, dummy_input)
    return symmetry_def


def create_dummy_env(
    *,
    with_state: bool = False,
    reward_dim: int = 1,
    numpy: bool = False,
    symmetric: bool = False,
):
    env = (DummyNumpyEnvironment if numpy else DummyEnvironment)(
        num_instances=8,
        observation_dim=16,
        action_dim=8,
        state_dim=24 if with_state else None,
        spec={"reward_dim": reward_dim},
    )
    if symmetric:
        env.spec.mirror_observation = create_random_symmetry_def(env.observation_dim)
        env.spec.mirror_action = create_random_symmetry_def(env.action_dim)
        if with_state:
            env.spec.mirror_state = create_random_symmetry_def(env.state_dim)
    return env


def run_environment_evaluation_loop(environment: cusrl.Environment, agent: cusrl.Agent, num_iterations=5):
    observation, state, _ = environment.reset()
    for iteration in range(num_iterations):
        while True:
            action = agent.act(observation, state)
            observation, state, reward, terminated, truncated, _ = environment.step(action)
            ready_to_update = agent.step(observation, reward, terminated, truncated, state)
            environment.reset(indices=get_done_indices(terminated, truncated))
            if ready_to_update:
                break
        agent.update()


def test_module_consistency(backbone_factory=None, is_recurrent=False, atol=1e-4):
    class ConsistencyHook(cusrl.Hook):
        def objective(self, batch):
            if batch["epoch_index"] == 0 and batch["mini_batch_index"] == 0:
                max_error = (batch["curr_action_mean"] - batch["action_mean"]).abs().max()
                assert max_error < atol, f"Max error {max_error} exceeds tolerance {atol}"

    if is_recurrent:
        agent_factory = cusrl.preset.ppo.RecurrentAgentFactory(sampler_mini_batches=1, sampler_epochs=1)
    else:
        agent_factory = cusrl.preset.ppo.AgentFactory(sampler_mini_batches=1, sampler_epochs=1)
    if backbone_factory is not None:
        agent_factory.actor_factory = cusrl.Actor.Factory(
            backbone_factory=backbone_factory,
            distribution_factory=cusrl.NormalDist.Factory(),
        )
    agent_factory.register_hook(ConsistencyHook())
    trainer = cusrl.Trainer(create_dummy_env, agent_factory, num_iterations=5)
    trainer.run_training_loop()

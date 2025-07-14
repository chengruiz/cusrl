import random

import gymnasium as gym
import pytest
import torch

import cusrl
from cusrl_test import create_dummy_env


def train_agent_with_observation_normalization(env, window_size=None, num_iterations=5):
    agent_factory = cusrl.preset.ppo.AgentFactory()
    agent_factory.register_hook(cusrl.hook.ObservationNormalization(window_size), after="ModuleInitialization")
    trainer = cusrl.Trainer(env, agent_factory, num_iterations=num_iterations)
    trainer.run_training_loop()
    return trainer.agent


@pytest.mark.parametrize(
    "environment",
    [
        create_dummy_env(),
        create_dummy_env(with_state=True),
        create_dummy_env(numpy=True),
        create_dummy_env(numpy=True, with_state=True),
        cusrl.make_gym_vec(
            "MountainCarContinuous-v0",
            num_envs=4,
            vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.DISABLED},
        ),
    ],
)
@pytest.mark.parametrize("window_size", [None, 100])
def test_observation_normalization(environment, window_size):
    agent = train_agent_with_observation_normalization(environment, window_size)
    agent.load_state_dict(agent.state_dict())


def test_observation_normalization_with_symmetry():
    env = create_dummy_env(with_state=True, symmetric=True)
    agent = train_agent_with_observation_normalization(env)
    hook = agent.hook["ObservationNormalization"]

    mean = hook.observation_rms.mean
    mirrored_mean = env.spec.mirror_observation(mean)
    var = hook.observation_rms.var
    mirrored_var = abs(env.spec.mirror_observation(var))
    assert torch.allclose(mean, mirrored_mean)
    assert torch.allclose(var, mirrored_var)

    mean = hook.state_rms.mean
    mirrored_mean = env.spec.mirror_state(mean)
    var = hook.state_rms.var
    mirrored_var = abs(env.spec.mirror_state(var))
    assert torch.allclose(mean, mirrored_mean)
    assert torch.allclose(var, mirrored_var)


@pytest.mark.parametrize(
    "env",
    [
        create_dummy_env(with_state=True),
        create_dummy_env(with_state=True, symmetric=True),
    ],
)
def test_observation_normalization_with_observation_is_subset_of_state(env):
    random_indices = list(range(env.state_dim))
    random.shuffle(random_indices)

    for subset in (slice(0, env.observation_dim), random_indices[: env.observation_dim]):
        env.spec.observation_is_subset_of_state = subset
        agent = train_agent_with_observation_normalization(env)
        hook = agent.hook["ObservationNormalization"]

        mean = hook.observation_rms.mean
        sliced_mean = hook.state_rms.mean[subset]
        var = hook.observation_rms.var
        sliced_var = hook.state_rms.var[subset]
        assert torch.allclose(mean, sliced_mean)
        assert torch.allclose(var, sliced_var)


def test_observation_normalization_with_stat_group():
    env = create_dummy_env(with_state=True)
    env.spec.observation_stat_groups = ((20, 30),)
    env.spec.state_stat_groups = ((0, 10), (30, 40))
    agent = train_agent_with_observation_normalization(env)
    hook = agent.hook["ObservationNormalization"]

    for observation_stat_group in env.spec.observation_stat_groups:
        subset = slice(*observation_stat_group)
        mean = hook.observation_rms.mean[subset]
        var = hook.observation_rms.var[subset]
        assert torch.allclose(mean, mean)
        assert torch.allclose(var, var)

    for state_stat_group in env.spec.state_stat_groups:
        subset = slice(*state_stat_group)
        mean = hook.state_rms.mean[subset]
        var = hook.state_rms.var[subset]
        assert torch.allclose(mean, mean)
        assert torch.allclose(var, var)

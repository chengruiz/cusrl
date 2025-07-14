from functools import partial

import gymnasium as gym
import pytest

import cusrl


def test_gym_env():
    cusrl.Trainer(
        environment=partial(cusrl.make_gym_env, id="MountainCarContinuous-v0"),
        agent_factory=cusrl.preset.ppo.AgentFactory(),
        num_iterations=5,
    ).run_training_loop()


@pytest.mark.parametrize("num_envs", [1, 16])
def test_gym_vec_env(num_envs):
    cusrl.Trainer(
        environment=partial(
            cusrl.make_gym_vec,
            id="MountainCarContinuous-v0",
            num_envs=num_envs,
            vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.DISABLED},
        ),
        agent_factory=cusrl.preset.ppo.AgentFactory(),
        num_iterations=10,
    ).run_training_loop()

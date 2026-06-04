import numpy as np
import torch

from cusrl.testing.environment import DummyNumpyEnvironment, DummyTorchEnvironment


def test_dummy_torch_environment_shapes_and_indexed_reset():
    environment = DummyTorchEnvironment(
        num_instances=4,
        observation_dim=3,
        action_dim=2,
        state_dim=5,
        reward_dim=2,
    )

    observation, state, info = environment.reset(indices=torch.tensor([0, 2]))
    assert observation.shape == (2, 3)
    assert state.shape == (2, 5)
    assert info == {}

    step = environment.step(torch.zeros(4, 2))
    next_observation, next_state, reward, terminated, truncated, info = step
    assert next_observation.shape == (4, 3)
    assert next_state.shape == (4, 5)
    assert reward.shape == (4, 2)
    assert terminated.shape == truncated.shape == (4, 1)
    assert terminated.dtype == truncated.dtype == torch.bool
    assert info == {}


def test_dummy_numpy_environment_shapes_and_dtypes():
    environment = DummyNumpyEnvironment(
        num_instances=3,
        observation_dim=4,
        action_dim=2,
        state_dim=None,
        reward_dim=1,
    )

    observation, state, info = environment.reset(indices=np.array([True, False, True]))
    assert observation.shape == (2, 4)
    assert observation.dtype == np.float32
    assert state is None
    assert info == {}

    step = environment.step(np.zeros((3, 2), dtype=np.float32))
    next_observation, next_state, reward, terminated, truncated, info = step
    assert next_observation.shape == (3, 4)
    assert next_state is None
    assert reward.shape == (3, 1)
    assert reward.dtype == np.float32
    assert terminated.dtype == truncated.dtype == np.bool_
    assert info == {}

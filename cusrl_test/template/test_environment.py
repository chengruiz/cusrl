import numpy as np
import torch

from cusrl.template.environment import get_done_indices, update_observation_and_state


def test_get_done_indices_accepts_torch_and_numpy_arrays():
    terminated = torch.tensor([[False], [True], [False], [False]])
    truncated = torch.tensor([[False], [False], [True], [False]])

    assert get_done_indices(terminated, truncated) == [1, 2]

    terminated_np = np.array([[False], [False], [True]])
    truncated_np = np.array([[True], [False], [False]])

    assert get_done_indices(terminated_np, truncated_np) == [0, 2]


def test_update_observation_and_state_replaces_partial_reset_rows():
    last_observation = torch.zeros(4, 2)
    last_state = torch.zeros(4, 3)
    init_observation = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    init_state = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])

    observation, state = update_observation_and_state(
        last_observation,
        last_state,
        [1, 3],
        init_observation,
        init_state,
    )

    assert observation is last_observation
    assert state is last_state
    assert torch.allclose(observation[[1, 3]], init_observation)
    assert torch.allclose(state[[1, 3]], init_state)
    assert torch.allclose(observation[[0, 2]], torch.zeros(2, 2))


def test_update_observation_and_state_uses_full_reset_output_directly():
    last_observation = torch.zeros(2, 2)
    init_observation = torch.ones(2, 2)

    observation, state = update_observation_and_state(
        last_observation,
        None,
        slice(None),
        init_observation,
        None,
    )

    assert observation is init_observation
    assert state is None

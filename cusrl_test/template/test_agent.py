import numpy as np
import torch

import cusrl


class _StubAgent(cusrl.Agent):
    def act(self, observation, state=None):
        raise NotImplementedError

    def step(self, next_observation, reward, terminated, truncated, next_state=None, **kwargs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


def test_to_tensor_does_not_share_storage_with_numpy_input():
    agent = _StubAgent(
        cusrl.EnvironmentSpec(observation_dim=2, action_dim=1),
        num_steps_per_update=1,
        device="cpu",
    )
    observation = np.array([[1.0, 2.0]], dtype=np.float32)
    tensor = agent.to_tensor(observation)

    observation[:] = -1.0
    assert torch.equal(tensor, torch.tensor([[1.0, 2.0]]))

    tensor[:] = 3.0
    np.testing.assert_array_equal(observation, np.array([[-1.0, -1.0]], dtype=np.float32))

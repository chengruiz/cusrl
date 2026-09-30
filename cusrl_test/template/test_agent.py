import numpy as np
import pytest
import torch

import cusrl


class _StubAgent(cusrl.Agent):
    def act(self, observation, state=None):
        raise NotImplementedError

    def step(self, next_observation, reward, terminated, truncated, next_state=None, **kwargs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


@pytest.fixture
def agent():
    return _StubAgent(
        cusrl.EnvironmentSpec(observation_dim=2, action_dim=1),
        num_steps_per_update=1,
        device="cpu",
    )


def test_to_tensor_does_not_share_storage_with_numpy_input(agent):
    observation = np.array([[1.0, 2.0]], dtype=np.float32)
    tensor = agent.to_tensor(observation)

    observation[:] = -1.0
    assert torch.equal(tensor, torch.tensor([[1.0, 2.0]]))

    tensor[:] = 3.0
    np.testing.assert_array_equal(observation, np.array([[-1.0, -1.0]], dtype=np.float32))


@pytest.mark.parametrize("observation", [np.array([[1.0, 2.0]], dtype=np.float32), [[1.0, 2.0]]], ids=["numpy", "list"])
def test_to_tensor_non_tensor_input_defaults_to_no_grad(agent, observation):
    tensor = agent.to_tensor(observation)

    assert not tensor.requires_grad
    torch.testing.assert_close(tensor, torch.tensor([[1.0, 2.0]]))


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("copy", [False, True])
def test_to_tensor_inherits_tensor_grad_and_preserves_graph(agent, requires_grad, copy):
    source = torch.tensor([[1.0, 2.0]], requires_grad=requires_grad)
    observation = source * 3.0

    tensor = agent.to_tensor(observation, copy=copy)

    assert tensor.requires_grad is requires_grad
    assert observation.requires_grad is requires_grad
    assert (tensor.data_ptr() != observation.data_ptr()) is copy
    torch.testing.assert_close(tensor, observation)
    if requires_grad:
        tensor.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 3.0))


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("copy", [False, True])
def test_to_tensor_explicit_grad_overrides_follow_asarray_copy_semantics(agent, requires_grad, copy):
    source = torch.tensor([[1.0, 2.0]], requires_grad=not requires_grad)
    observation = source * 3.0

    tensor = agent.to_tensor(observation, copy=copy, requires_grad=requires_grad)

    assert tensor.requires_grad is requires_grad
    if requires_grad and not copy:
        assert tensor is observation
        assert observation.requires_grad
    else:
        assert observation.requires_grad is (not requires_grad)
    assert source.requires_grad is (not requires_grad)
    torch.testing.assert_close(tensor, observation)
    if requires_grad:
        tensor.sum().backward()
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))
        assert source.grad is None
    else:
        assert tensor.grad_fn is None
        observation.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 3.0))

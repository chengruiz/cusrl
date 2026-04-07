import pytest
import torch
from torch import nn

from cusrl.nn.module.module import Module, resolve_activation_fn


class LinearModule(Module):
    def __init__(self):
        super().__init__(input_dim=2, output_dim=2)
        self.weight = nn.Parameter(torch.eye(2))

    def forward(self, input: torch.Tensor):
        return input @ self.weight


class RecurrentAccumulator(Module):
    def __init__(self):
        super().__init__(input_dim=2, output_dim=2, is_recurrent=True)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input: torch.Tensor, memory=None, **kwargs):
        if memory is None:
            memory = torch.zeros_like(input)
        next_memory = memory + input
        return input * self.scale, next_memory


class BareModule(Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_module_requires_dimensions_when_like_is_not_provided():
    with pytest.raises(ValueError, match="must be specified"):
        BareModule()


def test_resolve_activation_fn_supports_strings_and_rejects_invalid_values():
    assert resolve_activation_fn("ReLU") is nn.ReLU
    assert resolve_activation_fn("torch.nn.GELU") is nn.GELU

    with pytest.raises(ValueError, match="No activation function named"):
        resolve_activation_fn("MissingActivation")
    with pytest.raises(TypeError, match="issubclass\\(\\) arg 1 must be a class"):
        resolve_activation_fn(torch.relu)  # type: ignore[arg-type]


def test_module_device_property_uses_parameter_device():
    module = LinearModule()

    assert module.device == torch.device("cpu")


def test_rnn_compatible_wraps_non_recurrent_modules():
    module = LinearModule().rnn_compatible()
    input = torch.tensor([[1.0, 2.0]])
    memory = {"hidden": torch.ones(1, 2)}

    output = module(input)
    output_with_memory, returned_memory = module(input, memory=memory)

    assert torch.allclose(output, input)
    assert torch.allclose(output_with_memory, input)
    assert returned_memory is memory


def test_step_memory_uses_forward_memory_output_for_recurrent_modules():
    module = RecurrentAccumulator()
    input = torch.tensor([[1.0, 2.0]])
    memory = torch.tensor([[3.0, 4.0]])

    next_memory = module.step_memory(input, memory=memory)

    assert torch.allclose(next_memory, torch.tensor([[4.0, 6.0]]))


def test_reset_memory_zeros_requested_nested_entries():
    module = RecurrentAccumulator()
    memory = {
        "hidden": torch.ones(3, 2),
        "nested": {"cell": torch.full((3, 2), 2.0)},
    }

    module.reset_memory(memory, done=torch.tensor([True, False, True]))

    assert torch.allclose(memory["hidden"][0], torch.zeros(2))
    assert torch.allclose(memory["hidden"][1], torch.ones(2))
    assert torch.allclose(memory["hidden"][2], torch.zeros(2))
    assert torch.allclose(memory["nested"]["cell"][0], torch.zeros(2))
    assert torch.allclose(memory["nested"]["cell"][1], torch.full((2,), 2.0))
    assert torch.allclose(memory["nested"]["cell"][2], torch.zeros(2))

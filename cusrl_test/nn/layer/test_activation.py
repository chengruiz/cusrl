import pytest
import torch
from torch.nn.functional import gelu, silu

from cusrl.nn.layer import GeGlu, SwiGlu


@pytest.mark.parametrize(
    ("module_cls", "activation"),
    [
        (GeGlu, gelu),
        (SwiGlu, silu),
    ],
)
def test_glu_activations_match_reference_formula(module_cls, activation):
    module = module_cls(dim=-1)
    input = torch.tensor([[1.0, -2.0, 0.5, 1.5]])

    output = module(input)

    x, gate = input.chunk(2, dim=-1)
    expected = x * activation(gate)
    assert output.shape == (1, 2)
    assert torch.allclose(output, expected)

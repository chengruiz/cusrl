import pytest
import torch

import cusrl
from cusrl.nn.module.stub import Identity


def test_stub_module_returns_zero_output_with_requested_dimension():
    module = cusrl.StubModule.Factory()(4, 2)

    output = module(torch.randn(3, 4))

    assert output.shape == (3, 2)
    assert torch.allclose(output, torch.zeros(3, 2))


def test_identity_module_returns_input_unchanged():
    module = Identity.Factory()(4, None)
    input = torch.randn(3, 4)

    output = module(input)

    assert output is input


def test_identity_factory_rejects_mismatched_output_dimension():
    with pytest.raises(AssertionError):
        Identity.Factory()(4, 3)

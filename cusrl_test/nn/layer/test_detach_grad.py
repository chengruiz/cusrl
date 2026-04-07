import pytest
import torch

from cusrl.nn.layer import DetachGradient


def test_detach_gradient_breaks_backward_graph():
    input = torch.randn(4, requires_grad=True)

    output = DetachGradient()(input)

    assert torch.allclose(output, input)
    assert not output.requires_grad
    assert output.grad_fn is None
    with pytest.raises(RuntimeError):
        output.sum().backward()

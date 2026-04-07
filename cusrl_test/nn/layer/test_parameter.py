import torch
from torch import nn

from cusrl.nn.layer import ParameterWrapper


def test_parameter_wrapper_returns_trainable_parameter():
    module = ParameterWrapper([[1.0, 2.0], [3.0, 4.0]])

    output = module()
    output.sum().backward()

    assert isinstance(module.param, nn.Parameter)
    assert output is module.param
    assert torch.allclose(module.param.grad, torch.ones_like(module.param))

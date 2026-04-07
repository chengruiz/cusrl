import torch
from torch import nn

import cusrl
from cusrl.nn.module.simba import SimbaBlock


def test_simba_block_preserves_input_when_residual_branch_is_zero():
    block = SimbaBlock(hidden_dim=4, activation_fn="ReLU")
    for module in block:
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.zero_()
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()
    input = torch.randn(3, 4)

    output = block(input)

    assert torch.allclose(output, input)


def test_simba_factory_supports_default_and_projected_shapes():
    module1 = cusrl.Simba.Factory()(64)
    module2 = cusrl.Simba.Factory(hidden_dim=96, num_blocks=2, activation_fn=nn.SiLU)(64, 32)

    output1 = module1(torch.randn(8, 64))
    output2 = module2(torch.randn(8, 64))

    assert module1.hidden_dim == 64
    assert output1.shape == (8, 64)
    assert module2.hidden_dim == 96
    assert output2.shape == (8, 32)

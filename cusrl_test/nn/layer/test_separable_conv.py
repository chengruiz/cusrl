import torch
import torch.nn.functional as F

from cusrl.nn.layer import SeparableConv2d


def test_separable_conv2d_matches_depthwise_then_pointwise_convolution():
    module = SeparableConv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1)
    input = torch.arange(1, 1 + 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)

    module.depthwise.weight.data.copy_(
        torch.tensor([
            [[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]],
            [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]],
        ])
    )
    module.depthwise.bias.data.copy_(torch.tensor([0.5, -0.5]))
    module.pointwise.weight.data.copy_(
        torch.tensor([
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[1.0]], [[-1.0]]],
        ])
    )
    module.pointwise.bias.data.copy_(torch.tensor([1.0, 2.0, 3.0]))

    output = module(input)
    expected = F.conv2d(
        F.conv2d(
            input,
            module.depthwise.weight,
            module.depthwise.bias,
            padding=1,
            groups=2,
        ),
        module.pointwise.weight,
        module.pointwise.bias,
    )

    assert torch.allclose(output, expected)

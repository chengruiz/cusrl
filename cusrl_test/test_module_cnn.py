from functools import partial

import torch
from torch import nn

import cusrl


def test_cnn_output_shape():
    print("─" * 29, "CNN", "─" * 29)
    for i in range(4):
        input_flattened = i % 2 == 0
        flatten_output = i // 2 == 0
        print("input_flattened:", input_flattened, end="; ")
        print("flatten_output:", flatten_output)

        net = cusrl.Cnn.Factory(
            [
                partial(nn.Conv2d, 1, 16, 3, padding=1),
                partial(nn.ReLU, inplace=True),
                partial(nn.MaxPool2d, kernel_size=2),
                partial(nn.Conv2d, 16, 8, 3, padding=1),
                partial(nn.ReLU, inplace=True),
                partial(nn.MaxPool2d, kernel_size=2),
            ],
            (28, 20),
            input_flattened=input_flattened,
            flatten_output=flatten_output,
        )()

        input = torch.randn(28 * 20)
        if not input_flattened:
            input = input.reshape(1, 28, 20)
        for j in range(4):
            output = net(input)
            print(input.shape, "->", output.shape)
            assert output.ndim - input.ndim == (input_flattened - flatten_output) * 2
            input = input.unsqueeze(0)
        print("─" * 63)


def test_separable_conv2d():
    print("─" * 25, "SeparableConv2d", "─" * 25)
    in_ch, out_ch = 4, 8
    m = cusrl.SeparableConv2d(in_ch, out_ch, kernel_size=3, padding=1)
    x = torch.randn(2, in_ch, 16, 12)
    y = m(x)
    print("output shape:", y.shape)
    assert y.shape == (2, out_ch, 16, 12)

    # Build equivalent sequential depthwise + pointwise convs and copy weights
    seq = nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=True),
    )
    seq[0].weight.data.copy_(m.depthwise.weight.data)
    if m.depthwise.bias is not None:
        seq[0].bias.data.copy_(m.depthwise.bias.data)
    seq[1].weight.data.copy_(m.pointwise.weight.data)
    if m.pointwise.bias is not None:
        seq[1].bias.data.copy_(m.pointwise.bias.data)

    y_ref = seq(x)
    max_diff = (y - y_ref).abs().max().item()
    print("max diff vs sequential:", max_diff)
    assert torch.allclose(y, y_ref, atol=1e-6)

    # Gradient flow smoke test
    y.mean().backward()
    assert m.depthwise.weight.grad is not None
    assert m.pointwise.weight.grad is not None
    print("gradients OK")
    print("─" * 63)

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

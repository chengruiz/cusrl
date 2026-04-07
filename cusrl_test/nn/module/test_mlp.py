import torch
from torch import nn

import cusrl


def test_mlp_factory_builds_expected_output_shape_and_indexable_layers():
    module = cusrl.Mlp.Factory(hidden_dims=[6, 5], activation_fn=nn.ReLU)(4, 3)

    output = module(torch.randn(2, 4))

    assert output.shape == (2, 3)
    assert isinstance(module[0], nn.Linear)
    assert isinstance(module[1], nn.ReLU)
    assert isinstance(module[2], nn.Linear)


def test_mlp_can_end_with_activation_and_dropout():
    module = cusrl.Mlp(
        input_dim=4,
        hidden_dims=[5],
        output_dim=3,
        activation_fn="Tanh",
        ends_with_activation=True,
        dropout=0.2,
    )

    assert [type(layer) for layer in module.layers] == [
        nn.Linear,
        nn.Tanh,
        nn.Dropout,
        nn.Linear,
        nn.Tanh,
        nn.Dropout,
    ]


def test_mlp_uses_last_hidden_dim_as_output_dim_when_output_dim_is_omitted():
    module = cusrl.Mlp(input_dim=4, hidden_dims=[6, 5], activation_fn="ReLU")

    output = module(torch.randn(2, 4))

    assert module.output_dim == 5
    assert output.shape == (2, 5)

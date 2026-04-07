from functools import partial

import pytest
import torch
from torch import nn

import cusrl


def _cnn_layer_factories():
    return [
        partial(nn.Conv2d, 1, 16, 3, padding=1),
        partial(nn.ReLU, inplace=True),
        partial(nn.MaxPool2d, kernel_size=2),
        partial(nn.Conv2d, 16, 8, 3, padding=1),
        partial(nn.ReLU, inplace=True),
        partial(nn.MaxPool2d, kernel_size=2),
    ]


@pytest.mark.parametrize("input_flattened", [False, True])
@pytest.mark.parametrize("flatten_output", [False, True])
def test_cnn_output_shape_matches_flattening_configuration(input_flattened, flatten_output):
    net = cusrl.Cnn.Factory(
        _cnn_layer_factories(),
        (28, 20),
        input_flattened=input_flattened,
        flatten_output=flatten_output,
    )()

    input = torch.randn(28 * 20)
    if not input_flattened:
        input = input.reshape(1, 28, 20)

    for _ in range(4):
        output = net(input)
        assert output.ndim - input.ndim == (input_flattened - flatten_output) * 2
        input = input.unsqueeze(0)


def test_cnn_factory_supports_output_projection():
    net = cusrl.Cnn.Factory(
        _cnn_layer_factories(),
        (28, 20),
        input_flattened=True,
        flatten_output=True,
    )(output_dim=7)

    output = net(torch.randn(3, 28 * 20))

    assert output.shape == (3, 7)
    assert net.output_dim == 7


def test_cnn_rejects_invalid_shapes_and_projection_configuration():
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        cusrl.Cnn(layers=[], input_shape=(1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be True if 'output_dim' is set"):
        cusrl.Cnn(layers=[nn.Identity()], input_shape=(1, 4, 4), flatten_output=False, output_dim=3)


def test_cnn_factory_rejects_input_dim_mismatch():
    factory = cusrl.Cnn.Factory(_cnn_layer_factories(), (28, 20))

    with pytest.raises(ValueError, match="Input dimension mismatch"):
        factory(input_dim=10)

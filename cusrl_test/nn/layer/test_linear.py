import torch
from torch import nn

import cusrl


def test_disable_autocast_uses_explicit_device_type():
    layer = nn.Linear(4, 2)
    latent = torch.randn(3, 4)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_output = layer(latent)
        with cusrl.nn.disable_autocast("cpu"):
            fp32_output = layer(latent)

    assert autocast_output.dtype == torch.bfloat16
    assert fp32_output.dtype == torch.float32


def test_linear_fp32_disables_autocast_and_supports_half_parameters():
    torch.manual_seed(0)
    layer = cusrl.LinearFp32(4, 2)
    latent = torch.randn(3, 4)
    expected = nn.functional.linear(latent, layer.weight, layer.bias)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(latent)

    assert isinstance(layer, nn.Linear)
    assert output.dtype == torch.float32
    assert torch.equal(output, expected)

    layer = layer.half()
    half_latent = latent.half()
    half_expected = nn.functional.linear(half_latent.float(), layer.weight.float(), layer.bias.float())

    assert layer(half_latent).dtype == torch.float32
    assert torch.equal(layer(half_latent), half_expected)

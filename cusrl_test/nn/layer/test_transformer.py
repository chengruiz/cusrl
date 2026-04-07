import pytest
import torch

from cusrl.nn.layer import FeedForward, GeGlu, TransformerEncoderLayer


def test_feedforward_supports_glu_style_activations():
    module = FeedForward(
        input_dim=4,
        feedforward_dim=10,
        activation_fn=GeGlu,
        output_dim=3,
    )
    input = torch.randn(2, 4)

    output = module(input)

    assert module.layers[-1].in_features == 5
    assert output.shape == (2, 3)


@pytest.mark.parametrize("block_norm_order", ["pre", "post"])
def test_transformer_encoder_layer_supports_projection_and_norm_orders(block_norm_order):
    module = TransformerEncoderLayer(
        embed_dim=8,
        num_heads=2,
        input_dim=6,
        output_dim=5,
        block_norm="layer",
        block_norm_order=block_norm_order,
        qk_norm="layer",
        gate_type="highway",
        dropout=0.0,
    ).eval()
    input = torch.randn(2, 4, 6)

    output = module(input)

    assert output.shape == (2, 4, 5)

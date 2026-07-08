import pytest
import torch

from cusrl.nn.layer import FeedForward, GeGlu, NormFactory, TransformerDecoderLayer, TransformerEncoderLayer


def _assert_norm_config(norms, norm_cls):
    for norm in norms:
        assert isinstance(norm, norm_cls)
        assert norm.eps == pytest.approx(1e-5)
        assert norm.elementwise_affine is False


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


def test_transformer_encoder_layer_block_norm_accepts_norm_config():
    module = TransformerEncoderLayer(
        embed_dim=8,
        num_heads=2,
        block_norm=NormFactory("layer", eps=1e-5, elementwise_affine=False),
    )

    _assert_norm_config([module.norm1, module.norm2], torch.nn.LayerNorm)


def test_transformer_decoder_layer_block_norm_accepts_norm_config():
    module = TransformerDecoderLayer(
        embed_dim=8,
        num_heads=2,
        block_norm=NormFactory("rms", eps=1e-5, elementwise_affine=False),
    )

    _assert_norm_config([module.norm1, module.norm2, module.norm3], torch.nn.RMSNorm)


def test_transformer_encoder_layer_qk_norm_accepts_norm_config():
    module = TransformerEncoderLayer(
        embed_dim=8,
        num_heads=2,
        qk_norm=NormFactory("layer", eps=1e-5, elementwise_affine=False),
    )

    _assert_norm_config([module.self_attn.q_norm, module.self_attn.k_norm], torch.nn.LayerNorm)


def test_transformer_decoder_layer_qk_norm_accepts_norm_config():
    module = TransformerDecoderLayer(
        embed_dim=8,
        num_heads=2,
        qk_norm=NormFactory("rms", eps=1e-5, elementwise_affine=False),
    )

    _assert_norm_config(
        [
            module.self_attn.q_norm,
            module.self_attn.k_norm,
            module.cross_attn.q_norm,
            module.cross_attn.k_norm,
        ],
        torch.nn.RMSNorm,
    )

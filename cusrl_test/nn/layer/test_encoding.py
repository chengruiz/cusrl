import pytest
import torch

from cusrl.nn.layer.encoding import (
    LearnablePositionalEncoding2D,
    RotaryEmbedding,
    SinusoidalPositionalEncoding2D,
    apply_rotary_emb,
    rotate_half,
    sinusoidal_positional_encoding_2d,
)


def test_sinusoidal_positional_encoding_2d_rejects_invalid_channel_count():
    with pytest.raises(ValueError, match="divisible by 4"):
        sinusoidal_positional_encoding_2d(height=2, width=3, num_channels=6)


def test_sinusoidal_positional_encoding_2d_module_adds_fixed_encoding():
    module = SinusoidalPositionalEncoding2D(num_channels=8, height=2, width=3)
    input = torch.zeros(4, 8, 2, 3, dtype=torch.float64)

    output = module(input)

    assert not module.pe.requires_grad
    assert output.dtype == input.dtype
    assert torch.allclose(output, module.pe.type_as(input).expand_as(output))


def test_learnable_positional_encoding_2d_preserves_shape_and_requires_grad():
    torch.manual_seed(0)
    module = LearnablePositionalEncoding2D(num_channels=8, height=2, width=3)
    input = torch.zeros(4, 8, 2, 3)

    output = module(input)

    assert module.pe.requires_grad
    assert output.shape == input.shape
    assert torch.allclose(output, module.pe.expand_as(output))


def test_rotate_half_rotates_last_dimension_pairs():
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    output = rotate_half(input)

    assert torch.equal(output, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rotary_emb_matches_manual_formula():
    input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]])
    cos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sin = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    output = apply_rotary_emb(input, cos, sin)

    repeated_cos = cos.repeat(1, 2).unsqueeze(-2)
    repeated_sin = sin.repeat(1, 2).unsqueeze(-2)
    expected = input * repeated_cos + rotate_half(input) * repeated_sin
    assert torch.allclose(output, expected)


def test_rotary_embedding_builds_cache_and_rotates_qk_only():
    module = RotaryEmbedding(head_dim=4, max_seq_len=2)
    input = torch.randn(2, 4, 1, 4)
    qkv = torch.randn(2, 4, 3, 1, 4)

    output = module(input)
    output_qkv = module.apply_qkv(qkv.clone())

    cos, sin = module._get_cos_sin(4, device=input.device, dtype=input.dtype)
    q, k, v = qkv.unbind(dim=-3)
    expected_qkv = torch.stack([apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin), v], dim=-3)

    assert module.cos_cached.shape[0] >= 4
    assert module.sin_cached.shape[0] >= 4
    assert torch.allclose(output, apply_rotary_emb(input, cos, sin))
    assert torch.allclose(output_qkv, expected_qkv)


def test_rotary_embedding_rejects_odd_head_dimension():
    with pytest.raises(ValueError, match="must be even"):
        RotaryEmbedding(head_dim=3)

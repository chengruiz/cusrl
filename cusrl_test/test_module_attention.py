import pytest
import torch

from cusrl.module.attention import MultiheadSelfAttention, TransformerEncoderLayer
from cusrl_test import test_module_consistency


@pytest.mark.skipif(not MultiheadSelfAttention.is_available(), reason="Attention not available")
def test_self_mha_step_by_step_consistency():
    batch, seq, embed_dim, num_heads, window = 1, 7, 8, 2, 3
    attn = MultiheadSelfAttention(embed_dim, num_heads, window).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(seq, batch, embed_dim, device="cuda", dtype=torch.bfloat16)

    # full sequence computation
    out_full, _ = attn(x)

    # step-by-step computation
    memory = None
    outputs = []
    for t in range(seq):
        xt = x[t, :, :]
        out_step, memory = attn(xt, memory=memory)
        outputs.append(out_step)
    out_seq = torch.stack(outputs, dim=0)

    # compare full vs step-by-step outputs
    assert torch.allclose(out_full, out_seq, atol=1e-2)


@pytest.mark.skipif(not MultiheadSelfAttention.is_available(), reason="Attention not available")
def test_self_mha():
    batch, seq, embed_dim, num_heads, window = 1, 8, 2, 1, 3
    attn = MultiheadSelfAttention(embed_dim, num_heads, window).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(seq, batch, embed_dim, device="cuda", dtype=torch.bfloat16)

    # full sequence computation
    out1, (x_cache, kv_cache, mask) = attn(x)
    out2, _ = attn(x, memory=(x_cache, kv_cache, mask))
    assert out1.shape == (seq, batch, embed_dim)
    assert x_cache.shape == (window, batch, embed_dim)
    assert kv_cache.shape == (window, batch, embed_dim * 2)
    assert mask.shape == (window, batch, 1)
    assert out2.shape == (seq, batch, embed_dim)


@pytest.mark.skipif(not MultiheadSelfAttention.is_available(), reason="Attention not available")
def test_transformer_encoder_layer():
    batch, seq, embed_dim, num_heads, window = 1, 16, 32, 4, 6
    input_dim, output_dim = 24, 12
    attn = TransformerEncoderLayer(
        embed_dim,
        num_heads,
        window,
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(seq, batch, input_dim, device="cuda", dtype=torch.bfloat16)

    # full sequence computation
    out1, (x_cache, kv_cache, mask) = attn(x)
    out2, _ = attn(x, memory=(x_cache, kv_cache, mask))
    assert out1.shape == (seq, batch, output_dim)
    assert x_cache.shape == (window, batch, embed_dim)
    assert kv_cache.shape == (window, batch, embed_dim * 2)
    assert mask.shape == (window, batch, 1)
    assert out2.shape == (seq, batch, output_dim)


@pytest.mark.skipif(not MultiheadSelfAttention.is_available(), reason="Attention not available")
@pytest.mark.parametrize("gate_type", [None, "residual", "highway", "output", "input", "sig_tanh", "gru"])
@pytest.mark.parametrize("layer_norm", [None, "pre", "post"])
@pytest.mark.parametrize("use_alibi", [False, True])
@pytest.mark.parametrize("rope_base", [None, 100.0])
def test_transformer_alibi_consistency(gate_type, layer_norm, use_alibi, rope_base):
    test_module_consistency(
        TransformerEncoderLayer.Factory(
            embed_dim=32,
            num_heads=2,
            window_size=4,
            gate_type=gate_type,
            layer_norm=layer_norm,
            use_alibi=use_alibi,
            rope_base=rope_base,
        ),
        is_recurrent=True,
        atol=1e-2,
    )

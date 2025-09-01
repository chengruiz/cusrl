import pytest
import torch

import cusrl
from cusrl.module.mha import FlashAttention, MultiheadAttention, MultiheadCrossAttention
from cusrl.module.transformer import MultiheadSelfAttention, TransformerEncoderLayer
from cusrl_test import test_module_consistency


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
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


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
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


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
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


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
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


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("is_causal", [False, True])
def test_mha_consistency_with_torch(dtype, is_causal):
    torch.manual_seed(0)
    batch, seq, embed_dim, num_heads = 2, 9, 32, 4
    device = cusrl.device()

    mha_flash = MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
        dtype=dtype,
    ).to(device)
    mha_flash.eval()

    mha_torch = torch.nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
    ).to(device)
    mha_torch.eval()

    # Align weights: in-proj (Q,K,V) and out-proj
    if mha_torch.in_proj_weight is not None:
        mha_torch.in_proj_weight.copy_(
            torch.cat([mha_flash.q_proj.weight, mha_flash.k_proj.weight, mha_flash.v_proj.weight], dim=0)
        )
    if mha_torch.in_proj_bias is not None:
        mha_torch.in_proj_bias.copy_(
            torch.cat([mha_flash.q_proj.bias, mha_flash.k_proj.bias, mha_flash.v_proj.bias], dim=0)
        )

    if (mha_torch.q_proj_weight) is not None:
        mha_torch.q_proj_weight.copy_(mha_flash.q_proj.weight)
    if (mha_torch.k_proj_weight) is not None:
        mha_torch.k_proj_weight.copy_(mha_flash.k_proj.weight)
    if (mha_torch.v_proj_weight) is not None:
        mha_torch.v_proj_weight.copy_(mha_flash.v_proj.weight)
    mha_torch.out_proj.weight.copy_(mha_flash.out_proj.weight)
    if mha_torch.out_proj.bias is not None and mha_flash.out_proj.bias is not None:
        mha_torch.out_proj.bias.copy_(mha_flash.out_proj.bias)

    x = torch.randn(batch, seq, embed_dim, device=device, dtype=dtype)

    # forward
    with torch.autocast(device.type, dtype=dtype):
        out_flash, _ = mha_flash(x, x, x, is_causal=is_causal)
        # Build causal mask for PyTorch MHA to avoid relying on is_causal arg
        attn_mask = None
        if is_causal:
            L = S = seq
            attn_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=device), diagonal=1)
        out_torch, _ = mha_torch(x, x, x, need_weights=False, average_attn_weights=False, attn_mask=attn_mask)

    assert out_flash.shape == out_torch.shape
    assert torch.allclose(out_flash, out_torch, atol=1e-6, rtol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cross_mha_consistency_with_torch(dtype):
    torch.manual_seed(0)
    batch, q_len, kv_len = 2, 5, 7
    embed_dim, num_heads, kv_dim = 32, 4, 24
    device = cusrl.device()

    mha_flash = MultiheadCrossAttention(
        embed_dim,
        num_heads,
        dropout=0.0,
        kv_dim=kv_dim,
        batch_first=True,
        dtype=dtype,
    ).to(device)
    mha_flash.eval()

    mha_torch = torch.nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
        kdim=kv_dim,
        vdim=kv_dim,
    ).to(device)
    mha_torch.eval()

    k_w, v_w = mha_flash.kv_proj.weight.chunk(2, dim=0)
    k_b, v_b = mha_flash.kv_proj.bias.chunk(2, dim=0)
    if mha_torch.q_proj_weight is not None:
        mha_torch.q_proj_weight.copy_(mha_flash.q_proj.weight)
        mha_torch.k_proj_weight.copy_(k_w)
        mha_torch.v_proj_weight.copy_(v_w)
    elif mha_torch.in_proj_weight is not None:
        mha_torch.in_proj_weight.copy_(torch.cat([mha_flash.q_proj.weight, k_w, v_w], dim=0))

    if mha_torch.in_proj_bias is not None:
        mha_torch.in_proj_bias.copy_(torch.cat([mha_flash.q_proj.bias, k_b, v_b], dim=0))
    mha_torch.out_proj.weight.copy_(mha_flash.out_proj.weight)
    mha_torch.out_proj.bias.copy_(mha_flash.out_proj.bias)

    q = torch.randn(batch, q_len, embed_dim, device=device, dtype=dtype)
    kv = torch.randn(batch, kv_len, kv_dim, device=device, dtype=dtype)
    with torch.autocast(device.type, dtype=dtype):
        out_flash, _ = mha_flash(q, kv)
        out_torch, _ = mha_torch(q, kv, kv, need_weights=False, average_attn_weights=False)

    assert out_flash.shape == out_torch.shape
    assert torch.allclose(out_flash, out_torch, atol=1e-6, rtol=1e-6)

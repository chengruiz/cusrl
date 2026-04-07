from contextlib import nullcontext

import pytest
import torch

import cusrl
from cusrl.nn import RotaryEmbedding
from cusrl.nn.layer.flash_attention import FlashAttention


def _autocast_if_cuda(device: torch.device):
    return torch.autocast(device.type) if device.type == "cuda" else nullcontext()


@torch.no_grad()
@pytest.mark.parametrize("is_causal", [False, True])
def test_mha_consistency_with_torch(is_causal):
    torch.manual_seed(0)
    batch, seq, embed_dim, num_heads = 2, 9, 32, 4
    device = cusrl.device()

    mha = cusrl.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True).to(device).eval()
    mhsa = cusrl.nn.MultiheadSelfAttention(embed_dim, num_heads, dropout=0.0, batch_first=True).to(device).eval()
    torch_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True).to(device).eval()

    # Align weights: in-proj (Q,K,V) and out-proj
    qkv_proj_weight = torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight], dim=0)
    mhsa.qkv_proj.weight.copy_(qkv_proj_weight)
    torch_mha.in_proj_weight.copy_(qkv_proj_weight)

    qkv_proj_bias = torch.cat([mha.q_proj.bias, mha.k_proj.bias, mha.v_proj.bias], dim=0)
    mhsa.qkv_proj.bias.copy_(qkv_proj_bias)
    torch_mha.in_proj_bias.copy_(qkv_proj_bias)

    mhsa.out_proj.weight.copy_(mha.out_proj.weight)
    torch_mha.out_proj.weight.copy_(mha.out_proj.weight)
    mhsa.out_proj.bias.copy_(mha.out_proj.bias)
    torch_mha.out_proj.bias.copy_(mha.out_proj.bias)

    x = torch.randn(batch, seq, embed_dim, device=device)

    # forward
    with _autocast_if_cuda(device):
        out_flash = mha(x, x, x, is_causal=is_causal)
        out_flash2 = mhsa(x, is_causal=is_causal)
        # Build causal mask for PyTorch MHA to avoid relying on is_causal arg
        attn_mask = None
        if is_causal:
            L = S = seq
            attn_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=device), diagonal=1)
        out_torch, _ = torch_mha(x, x, x, need_weights=False, average_attn_weights=False, attn_mask=attn_mask)

    assert out_flash.shape == out_torch.shape == out_flash2.shape
    assert torch.allclose(out_flash, out_flash2, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_flash, out_torch, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_cross_mha_consistency_with_torch():
    torch.manual_seed(0)
    batch, q_len, kv_len = 2, 5, 7
    embed_dim, num_heads, kv_dim = 32, 4, 24
    device = cusrl.device()

    mha = cusrl.nn.MultiheadCrossAttention(
        embed_dim,
        num_heads,
        dropout=0.0,
        kv_dim=kv_dim,
        batch_first=True,
    ).to(device)
    mha.eval()

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

    k_w, v_w = mha.kv_proj.weight.chunk(2, dim=0)
    k_b, v_b = mha.kv_proj.bias.chunk(2, dim=0)
    if mha_torch.q_proj_weight is not None:
        mha_torch.q_proj_weight.copy_(mha.q_proj.weight)
        mha_torch.k_proj_weight.copy_(k_w)
        mha_torch.v_proj_weight.copy_(v_w)
    elif mha_torch.in_proj_weight is not None:
        mha_torch.in_proj_weight.copy_(torch.cat([mha.q_proj.weight, k_w, v_w], dim=0))

    if mha_torch.in_proj_bias is not None:
        mha_torch.in_proj_bias.copy_(torch.cat([mha.q_proj.bias, k_b, v_b], dim=0))
    mha_torch.out_proj.weight.copy_(mha.out_proj.weight)
    mha_torch.out_proj.bias.copy_(mha.out_proj.bias)

    q = torch.randn(batch, q_len, embed_dim, device=device)
    kv = torch.randn(batch, kv_len, kv_dim, device=device)
    with _autocast_if_cuda(device):
        out_flash = mha(q, kv)
        out_torch, _ = mha_torch(q, kv, kv, need_weights=False, average_attn_weights=False)

    assert out_flash.shape == out_torch.shape
    assert torch.allclose(out_flash, out_torch, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_transformer_decoder_factory_forward():
    batch, target_len, context_len = 2, 5, 7
    input_dim, embed_dim, context_dim, output_dim, num_heads = 24, 32, 20, 12, 4

    decoder = cusrl.nn.TransformerDecoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        input_dim=input_dim,
        output_dim=output_dim,
        context_dim=context_dim,
        block_norm="layer",
        block_norm_order="post",
    ).eval()

    target = torch.randn(batch, target_len, input_dim)
    context = torch.randn(batch, context_len, context_dim)
    output = decoder(target, context)

    assert output.shape == (batch, target_len, output_dim)


@torch.no_grad()
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("qk_norm", ["rms", "layer"])
def test_mha_qk_norm_consistency_between_self_and_general(is_causal, qk_norm):
    torch.manual_seed(0)
    batch, seq, embed_dim, num_heads = 2, 9, 32, 4
    device = cusrl.device()

    mha = cusrl.nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=0.0,
        qk_norm=qk_norm,
        batch_first=True,
    ).to(device)
    mha.eval()

    mhsa = cusrl.nn.MultiheadSelfAttention(
        embed_dim,
        num_heads,
        dropout=0.0,
        qk_norm=qk_norm,
        batch_first=True,
    ).to(device)
    mhsa.eval()

    mhsa.qkv_proj.weight.copy_(torch.cat([mha.q_proj.weight, mha.k_proj.weight, mha.v_proj.weight], dim=0))
    mhsa.qkv_proj.bias.copy_(torch.cat([mha.q_proj.bias, mha.k_proj.bias, mha.v_proj.bias], dim=0))
    mhsa.q_norm.weight.copy_(mha.q_norm.weight)
    mhsa.k_norm.weight.copy_(mha.k_norm.weight)
    mhsa.out_proj.weight.copy_(mha.out_proj.weight)
    mhsa.out_proj.bias.copy_(mha.out_proj.bias)

    x = torch.randn(batch, seq, embed_dim, device=device)
    with _autocast_if_cuda(device):
        out_mha = mha(x, x, x, is_causal=is_causal)
        out_mhsa = mhsa(x, is_causal=is_causal)

    assert out_mha.shape == out_mhsa.shape
    assert torch.allclose(out_mha, out_mhsa, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
def test_rope_correctness():
    x = torch.randn(2, 16, 4, 8).to("cuda")
    qkv = torch.randn(2, 16, 3, 4, 8).to("cuda")
    module = RotaryEmbedding(head_dim=8, max_seq_len=16).to("cuda")
    cusrl.config.enable_flash_attention(False)
    out1_x, out1_qkv = module(x), module.apply_qkv(qkv)
    cusrl.config.enable_flash_attention(True)
    out2_x, out2_qkv = module(x), module.apply_qkv(qkv)
    assert torch.allclose(out1_x, out2_x, atol=1e-5)
    assert torch.allclose(out1_qkv, out2_qkv, atol=1e-5)

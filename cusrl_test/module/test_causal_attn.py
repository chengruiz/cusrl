import pytest
import torch

import cusrl
from cusrl.module import CausalMultiheadSelfAttention, CausalTransformerEncoderLayer
from cusrl.module.encoding import RotaryEmbedding
from cusrl.module.mha import FlashAttention
from cusrl.utils.nest import map_nested
from cusrl_test import test_module_consistency


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
def test_causal_self_mha_consistency():
    batch, seq, embed_dim, num_heads, window = 1, 7, 8, 2, 3
    attn = CausalMultiheadSelfAttention(embed_dim, num_heads, window).to(device="cuda", dtype=torch.bfloat16)
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
def test_causal_self_mha():
    batch, seq, embed_dim, num_heads, window = 1, 8, 2, 1, 3
    attn = CausalMultiheadSelfAttention(embed_dim, num_heads, window).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(seq, batch, embed_dim, device="cuda", dtype=torch.bfloat16)

    # full sequence computation
    out1, memory = attn(x)
    out2, _ = attn(x, memory=memory)
    assert out1.shape == (seq, batch, embed_dim)
    assert memory["input_cache"].shape == (batch, window * embed_dim)
    assert memory["kv_cache"].shape == (batch, window * embed_dim * 2)
    assert memory["cache_mask"].shape == (batch, window)
    assert out2.shape == (seq, batch, embed_dim)


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
def test_causal_transformer_encoder_layer():
    batch, seq, embed_dim, num_heads, window = 1, 16, 32, 4, 6
    input_dim, output_dim = 24, 12
    attn = CausalTransformerEncoderLayer(
        embed_dim,
        num_heads,
        window,
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(seq, batch, input_dim, device="cuda", dtype=torch.bfloat16)

    # full sequence computation
    out1, memory = attn(x)
    out2, _ = attn(x, memory=memory)
    assert out1.shape == (seq, batch, output_dim)
    assert memory["input_cache"].shape == (batch, window * embed_dim)
    assert memory["kv_cache"].shape == (batch, window * embed_dim * 2)
    assert memory["cache_mask"].shape == (batch, window)
    assert out2.shape == (seq, batch, output_dim)


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
def test_causal_self_mha_cache_mask_with_done():
    batch, seq, embed_dim, num_heads, window = 8, 24, 32, 4, 4
    attn = CausalMultiheadSelfAttention(embed_dim, num_heads, window).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(seq, batch, embed_dim, device="cuda", dtype=torch.bfloat16)
    done = torch.zeros(seq, batch, 1, device="cuda", dtype=torch.bool)
    done[7, 3] = True
    done[12, 0] = True
    done[18, 5] = True

    _, memory = attn(x[:window])
    out, next_memory = attn(x, memory=memory, done=done)

    assert out.shape == (seq, batch, embed_dim)
    assert next_memory["input_cache"].shape == (batch, window * embed_dim)
    assert next_memory["kv_cache"].shape == (batch, window * embed_dim * 2)
    assert next_memory["cache_mask"].shape == (batch, window)


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
@pytest.mark.parametrize("gate_type", [None, "residual", "highway", "output", "input", "sigmoid_tanh", "gru"])
@pytest.mark.parametrize("layer_norm", [None, "pre", "post"])
@pytest.mark.parametrize("use_alibi", [False, True])
@pytest.mark.parametrize("rope_base", [None, 100.0])
def test_transformer_alibi_consistency(gate_type, layer_norm, use_alibi, rope_base):
    test_module_consistency(
        CausalTransformerEncoderLayer.Factory(
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


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
def test_causal_self_mha_done_with_extra_batch_dims_matches_flattened_batch():
    torch.manual_seed(0)

    num_envs, num_augs = 2, 3
    seq_len, embed_dim, num_heads, window_size = 6, 16, 4, 4
    attn = CausalMultiheadSelfAttention(embed_dim, num_heads, window_size).to(device="cuda", dtype=torch.bfloat16)

    input_flat = torch.randn(seq_len, num_envs * num_augs, embed_dim, device="cuda", dtype=torch.bfloat16)
    input_multi = input_flat.unflatten(1, (num_envs, num_augs))

    done = torch.zeros(seq_len, num_envs, 1, device="cuda", dtype=torch.bool)
    done[2, 0] = True
    done[4, 1] = True
    done_flat = done.repeat_interleave(num_augs, dim=1)

    _, memory_flat = attn(input_flat[:window_size])
    _, memory_multi = attn(input_multi[:window_size])

    output_flat, next_memory_flat = attn(input_flat, memory=memory_flat, done=done_flat)
    output_multi, next_memory_multi = attn(input_multi, memory=memory_multi, done=done)

    assert torch.allclose(output_flat, output_multi.flatten(1, 2), atol=1e-2)
    assert torch.allclose(next_memory_flat["input_cache"], next_memory_multi["input_cache"].flatten(0, 1))
    assert torch.allclose(next_memory_flat["kv_cache"], next_memory_multi["kv_cache"].flatten(0, 1))
    assert torch.equal(next_memory_flat["cache_mask"], next_memory_multi["cache_mask"].flatten(0, 1))


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
def test_causal_self_mha_accepts_sequential_memory():
    torch.manual_seed(0)

    batch, seq_len, embed_dim, num_heads, window_size = 4, 6, 16, 4, 4
    warmup_len = 3
    attn = CausalMultiheadSelfAttention(embed_dim, num_heads, window_size).to(device="cuda", dtype=torch.bfloat16)

    warmup = torch.randn(warmup_len, batch, embed_dim, device="cuda", dtype=torch.bfloat16)
    observation = torch.randn(seq_len, batch, embed_dim, device="cuda", dtype=torch.bfloat16)
    done = torch.zeros(seq_len, batch, 1, device="cuda", dtype=torch.bool)
    done[2, 1] = True
    done[4, 3] = True

    _, initial_memory = attn(warmup)

    memory = map_nested(torch.clone, initial_memory)
    rollout_memories = []
    outputs = []
    for t in range(seq_len):
        rollout_memories.append(map_nested(torch.clone, memory))
        output, memory = attn(observation[t], memory=memory)
        attn.reset_memory(memory, done[t])
        outputs.append(output)

    sequential_memory = {
        key: torch.stack([step_memory[key] for step_memory in rollout_memories], dim=0) for key in rollout_memories[0]
    }
    output_seq = torch.stack(outputs, dim=0)
    output, _ = attn(observation, memory=sequential_memory, done=done)

    assert torch.allclose(output_seq, output, atol=1e-2)


@pytest.mark.skipif(not FlashAttention.is_available(), reason="FlashAttention not available")
def test_causal_self_mha_non_sequential_keeps_batch_shaped_memory():
    torch.manual_seed(0)

    num_batches1, num_batches2, embed_dim, num_heads, window_size = 2, 3, 16, 4, 4
    attn = CausalMultiheadSelfAttention(embed_dim, num_heads, window_size).to(device="cuda", dtype=torch.bfloat16)

    input_multi = torch.randn(num_batches1, num_batches2, embed_dim, device="cuda", dtype=torch.bfloat16)
    warmup_multi = torch.randn(num_batches1, num_batches2, embed_dim, device="cuda", dtype=torch.bfloat16)
    _, memory_multi = attn(warmup_multi, sequential=False)

    output_multi, next_memory_multi = attn(input_multi, memory=memory_multi, sequential=False)
    output_flat, next_memory_flat = attn(
        input_multi.flatten(0, 1),
        memory=map_nested(lambda mem: mem.flatten(0, 1), memory_multi),
        sequential=False,
    )

    assert torch.allclose(output_multi.flatten(0, 1), output_flat, atol=1e-2)
    assert torch.allclose(next_memory_multi["input_cache"].flatten(0, 1), next_memory_flat["input_cache"])
    assert torch.allclose(next_memory_multi["kv_cache"].flatten(0, 1), next_memory_flat["kv_cache"])
    assert torch.equal(next_memory_multi["cache_mask"].flatten(0, 1), next_memory_flat["cache_mask"])

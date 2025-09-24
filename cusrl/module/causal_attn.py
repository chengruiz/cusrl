import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from cusrl.module.gate import get_gate_cls
from cusrl.module.mha import FlashAttention
from cusrl.module.module import Module, ModuleFactory
from cusrl.module.transformer import FeedForward
from cusrl.utils.recurrent import (
    compute_reverse_cumulative_timesteps,
    compute_sequence_indices,
    compute_sequence_lengths,
    cumulate_sequence_lengths,
)
from cusrl.utils.typing import Slice

try:
    from flash_attn import flash_attn_varlen_kvpacked_func
    from flash_attn.layers.rotary import apply_rotary_emb, apply_rotary_emb_kv_
    from flash_attn.modules.mha import get_alibi_slopes
except ImportError:
    flash_attn_varlen_kvpacked_func = None
    apply_rotary_emb = apply_rotary_emb_kv_ = get_alibi_slopes = None


__all__ = ["CausalMultiheadSelfAttention", "CausalTransformerEncoderLayer", "FeedForward"]


@dataclass(slots=True)
class CausalMultiheadSelfAttentionFactory(ModuleFactory["CausalMultiheadSelfAttention"]):
    embed_dim: int
    num_heads: int
    window_size: int
    dropout: float = 0.0
    dtype: torch.dtype = torch.float16
    alibi_slopes: Tensor | None = None
    rope_base: float | None = None

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        return CausalMultiheadSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            dropout=self.dropout,
            dtype=self.dtype,
            alibi_slopes=self.alibi_slopes,
            rope_base=self.rope_base,
            input_dim=input_dim,
            output_dim=output_dim,
        )


class CausalMultiheadSelfAttention(Module, FlashAttention):
    Factory = CausalMultiheadSelfAttentionFactory

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16,
        alibi_slopes: Tensor | None = None,
        rope_base: float | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.dtype = dtype
        self.alibi_slopes = torch.as_tensor(alibi_slopes) if alibi_slopes is not None else None
        self.rope_base = rope_base  # Rotary Positional Embedding
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.rope_base is not None:
            if self.rope_base <= 0:
                raise ValueError("rope_base must be a positive float.")
            if self.head_dim // 2 == 0:
                raise ValueError("head_dim must be even for RoPE.")
        if self.alibi_slopes is not None and self.alibi_slopes.ndim != 1:
            raise ValueError("alibi_slopes must be a 1D tensor.")
        if self.alibi_slopes is not None and self.alibi_slopes.size(0) != num_heads:
            raise ValueError(f"alibi_slopes must have {num_heads=} elements.")
        if self.window_size <= 0:
            raise ValueError("window_size must be a non-negative integer.")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FlashAttention only supports float16 or bfloat16 dtypes.")
        super().__init__(
            input_dim=input_dim or embed_dim,
            output_dim=output_dim or embed_dim,
            is_recurrent=True,
        )

        # projections
        self.q_proj = nn.Linear(self.input_dim, embed_dim)
        self.kv_proj = nn.Linear(self.input_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, self.output_dim)
        self._inference = 0
        self._rotary_cos = self._rotary_sin = None

    def forward(
        self,
        input: Tensor,
        memory: tuple[Tensor, Tensor, Tensor] | None = None,
        *,
        done: Tensor | None = None,
        sequential: bool = True,
        **kwargs,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Computes multi-head self-attention with KV caching.

        Args:
            input (Tensor):
                Input tensor of shape :math:`(L, N, C)`, where :math:`L` is the
                sequence length, :math:`N` is the batch size, and :math:`C` is
                the input dimension.
            memory (tuple[Tensor, Tensor, Tensor] | None):
                A tuple containing the input cache, KV cache, and cache mask.
                  - input_cache (Tensor):
                      Tensor of shape :math:`(W, N, C)` storing past inputs,
                      where :math:`W` is the window size.
                  - kv_cache (Tensor):
                      Tensor of shape :math:`(W, N, 2 * E)` storing past keys
                      and values, where :math:`E` is the embedding dimension.
                  - cache_mask (Tensor):
                      Boolean tensor of shape :math:`(W, N, 1)` indicating valid
                      cache entries.
            done (Tensor | None):
                A boolean tensor of shape :math:`(L, N, 1)` indicating sequence
                terminations.
            sequential (bool):
                If ``True``, the input is treated as a sequences. Otherwise,
                it's treated as a single batch of data. Defaults to ``True``.

        Outputs:
            - **output** (Tensor):
                The attention output tensor of the same shape as ``input``.
            - **memory** (tuple[Tensor, Tensor, Tensor]):
                The updated memory tuple `(input_cache, kv_cache, cache_mask)`.
        """
        if seq_missing := (input.dim() == 2 or not sequential):
            input = input.unsqueeze(0)
        batch_dims = input.shape[1:-1]
        input = input.flatten(1, -2)

        # Convert inputs to batch first
        input = input.transpose(0, 1)
        batch_size, seq_len, _ = input.shape
        full_seq_len = seq_len + self.window_size

        # Compute query linear projections
        q = self.q_proj(input).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if memory is None:
            # Initialize KV and mask if no memory is provided
            kv_cache = input.new_zeros(batch_size, self.window_size, 2 * self.embed_dim)
            kv = torch.cat([kv_cache, self.kv_proj(input)], dim=1)
            kv = kv.unflatten(-1, (2, self.num_heads, self.head_dim))
            kv_mask = q.new_zeros(batch_size, full_seq_len, dtype=torch.bool)
            full_input = input.new_zeros(batch_size, full_seq_len, self.input_dim)
            kv_mask[:, -seq_len:] = True
            full_input[:, -seq_len:] = input
            seq_lens_mask = input.new_zeros(batch_size, dtype=torch.int32)
        else:
            # Concatenate past states and generate mask
            input_cache, kv_cache, cache_mask = memory
            input_cache = input_cache.flatten(1, -2).transpose(0, 1)
            kv_cache = kv_cache.flatten(1, -2).transpose(0, 1)
            cache_mask = cache_mask.flatten(1, -2).transpose(0, 1).squeeze(-1)
            full_input = torch.cat([input_cache, input], dim=1)
            self._inference = 0 if torch.is_grad_enabled() else self._inference + 1

            # Discard KV cache for the first inference step
            if self._inference <= 1:
                # Stop gradients for KV cache
                kv_cache = self.kv_proj(input_cache).detach()
            kv = torch.cat([kv_cache, self.kv_proj(input)], dim=1)
            kv = kv.unflatten(-1, (2, self.num_heads, self.head_dim))
            kv_mask = cache_mask.new_ones(batch_size, full_seq_len)
            kv_mask[:, : self.window_size] = cache_mask
            seq_lens_mask = cache_mask.sum(dim=-1, dtype=torch.int32)

        # Compute sequence lengths
        if done is None:
            seq_lens_q = seq_lens_mask.new_full((batch_size,), seq_len)
            seq_lens_k = seq_lens_mask + seq_len
        else:
            if len(batch_dims) > 1:
                done = done.repeat(1, math.prod(batch_dims[:-1]), 1)
            seq_lens_q = compute_sequence_lengths(done)
            seq_indices = compute_sequence_indices(done)
            seq_lens_k = seq_lens_q.clone()
            seq_lens_k[seq_indices[:-1]] += seq_lens_mask

        # Compute attention
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(device=input.device)
        original_kv = kv
        if self.rope_base is not None:
            self._update_cos_sin_cache(seq_len + self.window_size, q.device)
            original_kv = kv.clone()
            assert apply_rotary_emb is not None and apply_rotary_emb_kv_ is not None
            q = apply_rotary_emb(q, self._rotary_cos, self._rotary_sin, inplace=True, seqlen_offsets=self.window_size)
            kv = apply_rotary_emb_kv_(kv, self._rotary_cos, self._rotary_sin)

        assert flash_attn_varlen_kvpacked_func is not None
        attn_out = flash_attn_varlen_kvpacked_func(
            q=q.flatten(0, 1).to(self.dtype),
            kv=kv[kv_mask].to(self.dtype),
            cu_seqlens_q=cumulate_sequence_lengths(seq_lens_q).to(torch.int32),
            cu_seqlens_k=cumulate_sequence_lengths(seq_lens_k).to(torch.int32),
            max_seqlen_q=seq_lens_q.max().item(),
            max_seqlen_k=seq_lens_k.max().item(),
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),
            alibi_slopes=self.alibi_slopes,
        ).type_as(input)

        # Combine heads and project to output_dim
        output = self.out_proj(attn_out.flatten(-2))
        output = output.unflatten(0, (batch_size, seq_len))

        # Prepare new cache tensors
        new_input_cache = full_input[:, -self.window_size :]
        new_kv_cache = original_kv.flatten(-3)[:, -self.window_size :]
        new_cache_mask = kv_mask[:, -self.window_size :]

        # Restore outputs to sequence first ( L, N, * )
        output = output.transpose(0, 1)
        new_input_cache = new_input_cache.transpose(0, 1)
        new_kv_cache = new_kv_cache.transpose(0, 1)
        new_cache_mask = new_cache_mask.transpose(0, 1)

        # Update cache mask based on done tensor
        if done is not None:
            if done.size(0) < self.window_size:
                padded_done = done.new_zeros(self.window_size, done.size(1), 1)
                padded_done[-done.size(0) :] = done
                done = padded_done
            elif done.size(0) > self.window_size:
                done = done[-self.window_size :]

            cum_timesteps = compute_reverse_cumulative_timesteps(done).squeeze(-1)
            consecutive_timesteps = torch.arange(self.window_size - 1, -1, -1, device=done.device)
            new_cache_mask = new_cache_mask.logical_and(cum_timesteps == consecutive_timesteps.unsqueeze(-1))

        output = output.unflatten(1, batch_dims)
        new_input_cache = new_input_cache.unflatten(1, batch_dims)
        new_kv_cache = new_kv_cache.unflatten(1, batch_dims)
        new_cache_mask = new_cache_mask.unflatten(1, batch_dims)
        if seq_missing:
            output = output.squeeze(0)
        return output, (new_input_cache, new_kv_cache, new_cache_mask.unsqueeze(-1))

    def reset_memory(
        self,
        memory: tuple[Tensor, Tensor, Tensor] | None,
        done: Slice | Tensor | None = None,
    ):
        """Resets the memory cache for specific environments.

        This method selectively resets the memory components (input cache,
        key-value cache, and cache mask). If ``done`` is not provided, the entire
        memory is cleared. Otherwise, only the memory states corresponding to
        the ``done`` indices (e.g., for environments that are done) are reset.

        Args:
            memory (tuple[Tensor, Tensor, Tensor] | None):
                A tuple containing the input cache, KV cache, and cache mask. If
                ``None``, the function does nothing.
            done (SliceType | Tensor | None, optional):
                A mask or slice indicating which parts of the memory to reset.
                If it's a tensor, it should be of shape :math:`(N, 1)`. If
                ``None``, the entire memory is reset. Defaults to ``None``.
        """
        if memory is None:
            return
        input_cache, kv_cache, cache_mask = memory
        if done is None:
            input_cache.zero_()
            kv_cache.zero_()
            cache_mask.fill_(False)
        else:
            if isinstance(done, Tensor):
                done = done.squeeze(-1)
            input_cache[:, done, :] = 0.0
            kv_cache[:, done, :] = 0.0
            cache_mask[:, done, :] = False

    def _update_cos_sin_cache(self, seq_len, device):
        if self._rotary_sin is not None and self._rotary_sin.size(0) >= seq_len:
            return

        t = torch.arange(0.0, seq_len, device=device)
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0.0, self.head_dim, 2.0, device=device) / self.head_dim))
        freq = torch.outer(t, inv_freq)
        self._rotary_cos = freq.cos()
        self._rotary_sin = freq.sin()


@dataclass(slots=True)
class CausalTransformerEncoderLayerFactory(ModuleFactory["CausalTransformerEncoderLayer"]):
    embed_dim: int
    num_heads: int
    window_size: int
    feedforward_dim: int | None = None
    activation_fn: type[nn.Module] = nn.GELU
    dropout: float = 0.0
    dtype: torch.dtype = torch.float16
    gate_type: str | None = "residual"
    layer_norm: Literal[None, "pre", "post"] = "post"
    use_alibi: bool = False
    rope_base: float | None = None

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        return CausalTransformerEncoderLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            feedforward_dim=self.feedforward_dim,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            dtype=self.dtype,
            gate_type=self.gate_type,
            layer_norm=self.layer_norm,
            use_alibi=self.use_alibi,
            rope_base=self.rope_base,
            input_dim=input_dim,
            output_dim=output_dim,
        )


class CausalTransformerEncoderLayer(Module):
    Factory = CausalTransformerEncoderLayerFactory

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        feedforward_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float16,
        gate_type: str | None = "residual",
        layer_norm: Literal[None, "pre", "post"] = "post",
        use_alibi: bool = False,
        rope_base: float | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        self.embed_dim = embed_dim
        self.layer_norm = layer_norm
        gate_cls = get_gate_cls(gate_type)
        super().__init__(
            input_dim=input_dim or embed_dim,
            output_dim=output_dim or embed_dim,
            is_recurrent=True,
        )

        # modules
        if self.input_dim != self.embed_dim:
            self.in_proj = nn.Linear(self.input_dim, self.embed_dim)
        else:
            self.in_proj = nn.Identity()

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.self_attn = CausalMultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            dtype=dtype,
            input_dim=embed_dim,
            output_dim=self.embed_dim,
            alibi_slopes=get_alibi_slopes(num_heads) if use_alibi else None,
            rope_base=rope_base,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate1 = gate_cls(self.embed_dim)

        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.feedforward = FeedForward(
            input_dim=self.embed_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            output_dim=self.embed_dim,
            activation_fn=activation_fn,
        )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate2 = gate_cls(self.embed_dim)

        if self.output_dim != self.embed_dim:
            self.out_proj = nn.Linear(self.embed_dim, self.output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(
        self,
        input: Tensor,
        memory: tuple[Tensor, Tensor, Tensor] | None = None,
        *,
        done: Tensor | None = None,
        sequential: bool = True,
        **kwargs,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        input = self.in_proj(input)
        if self.layer_norm == "pre":
            # pre-norm: norm -> attn -> add -> norm -> ff -> add
            attn_out, memory = self.self_attn(self.norm1(input), memory, done=done, sequential=sequential)
            input = self.gate1(input, self.dropout1(attn_out))

            ff_out = self.feedforward(self.norm2(input))
            input = self.gate2(input, self.dropout2(ff_out))
        elif self.layer_norm == "post":
            # post-norm: attn -> add -> norm -> ff -> add -> norm
            attn_out, memory = self.self_attn(input, memory, done=done, sequential=sequential)
            input = self.norm1(self.gate1(input, self.dropout1(attn_out)))

            ff_out = self.feedforward(input)
            input = self.norm2(self.gate2(input, self.dropout2(ff_out)))
        else:
            # no norm: attn -> add -> ff -> add
            attn_out, memory = self.self_attn(input, memory, done=done, sequential=sequential)
            input = self.gate1(input, self.dropout1(attn_out))

            ff_out = self.feedforward(input)
            input = self.gate2(input, self.dropout2(ff_out))

        return self.out_proj(input), memory

    def step_memory(self, input, memory=None, **kwargs):
        input = self.in_proj(input)
        if self.layer_norm == "pre":
            input = self.norm1(input)
        return self.self_attn.step_memory(input, memory, **kwargs)

    def reset_memory(
        self,
        memory: tuple[Tensor, Tensor, Tensor],
        done: Slice | Tensor | None = None,
    ):
        self.self_attn.reset_memory(memory, done)

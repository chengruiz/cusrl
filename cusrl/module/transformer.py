from typing import Literal

import torch
from torch import Tensor, nn

from cusrl.module.gate import get_gate_cls
from cusrl.module.mha import MultiheadCrossAttention, MultiheadSelfAttention, make_norm

__all__ = ["FeedForward", "TransformerDecoderLayer", "TransformerEncoderLayer"]


class FeedForward(nn.Module):
    """A feed-forward network module.

    This module implements a standard feed-forward network, typically used as a
    sub-layer in a Transformer block. It consists of two linear layers with an
    activation function and optional dropout in between.

    Args:
        input_dim (int):
            The dimension of the input tensor.
        feedforward_dim (int | None, optional):
            The dimension of the hidden layer. Defaults to ``input_dim * 4``.
        activation_fn (type[nn.Module], optional):
            The activation function to use. Defaults to :class:`nn.GELU`.
        dropout (float, optional):
            The dropout rate. Defaults to ``0.0``.
        output_dim (int, optional):
            The dimension of the output tensor. Defaults to ``input_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        feedforward_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.feedforward_dim = feedforward_dim or input_dim * 4

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.feedforward_dim),
            activation_fn(),
        )
        if dropout > 0.0:
            self.layers.append(nn.Dropout(dropout))
        hidden_dim = self.layers(torch.zeros(1, self.input_dim)).size(-1)
        self.layers.append(nn.Linear(hidden_dim, self.output_dim))

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.GELU,
        rope_base: float | None = None,
        dropout: float = 0.0,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float16,
        gate_type: str | None = "residual",
        qk_norm: Literal["rms", "layer"] | None = None,
        block_norm: Literal["rms", "layer"] | None = None,
        block_norm_order: Literal["pre", "post"] = "post",
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim or embed_dim
        self.output_dim = output_dim or embed_dim
        self.block_norm_order = block_norm_order
        gate_cls = get_gate_cls(gate_type)

        # modules
        if self.input_dim != self.embed_dim:
            self.in_proj: nn.Module = nn.Linear(self.input_dim, self.embed_dim)
        else:
            self.in_proj = nn.Identity()

        self.norm1 = make_norm(block_norm, self.embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope_base=rope_base,
            qk_norm=qk_norm,
            dropout=dropout,
            batch_first=batch_first,
            dtype=dtype,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate1 = gate_cls(self.embed_dim)

        self.norm2 = make_norm(block_norm, self.embed_dim)
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
            self.out_proj: nn.Module = nn.Linear(self.embed_dim, self.output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, input: Tensor, is_causal: bool = False) -> Tensor:
        input = self.in_proj(input)
        if self.block_norm_order == "pre":
            # pre-norm: norm -> attn -> add -> norm -> ff -> add
            attn_out = self.self_attn(self.norm1(input), is_causal=is_causal)
            input = self.gate1(input, self.dropout1(attn_out))

            ff_out = self.feedforward(self.norm2(input))
            input = self.gate2(input, self.dropout2(ff_out))
        else:
            # post-norm: attn -> add -> norm -> ff -> add -> norm
            attn_out = self.self_attn(input, is_causal=is_causal)
            input = self.norm1(self.gate1(input, self.dropout1(attn_out)))

            ff_out = self.feedforward(input)
            input = self.norm2(self.gate2(input, self.dropout2(ff_out)))

        return self.out_proj(input)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_dim: int | None = None,
        feedforward_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.GELU,
        rope_base: float | None = None,
        dropout: float = 0.0,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float16,
        gate_type: str | None = "residual",
        qk_norm: Literal["rms", "layer"] | None = None,
        block_norm: Literal["rms", "layer"] | None = None,
        block_norm_order: Literal["pre", "post"] = "post",
        input_dim: int | None = None,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_dim = context_dim or embed_dim
        self.input_dim = input_dim or embed_dim
        self.output_dim = output_dim or embed_dim
        self.block_norm_order = block_norm_order
        gate_cls = get_gate_cls(gate_type)

        if self.input_dim != self.embed_dim:
            self.in_proj: nn.Module = nn.Linear(self.input_dim, self.embed_dim)
        else:
            self.in_proj = nn.Identity()

        self.norm1 = make_norm(block_norm, self.embed_dim)
        self.self_attn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope_base=rope_base,
            qk_norm=qk_norm,
            dropout=dropout,
            batch_first=batch_first,
            dtype=dtype,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate1 = gate_cls(self.embed_dim)

        self.norm2 = make_norm(block_norm, self.embed_dim)
        self.cross_attn = MultiheadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            dropout=dropout,
            kv_dim=self.context_dim,
            batch_first=batch_first,
            dtype=dtype,
        )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate2 = gate_cls(self.embed_dim)

        self.norm3 = make_norm(block_norm, self.embed_dim)
        self.feedforward = FeedForward(
            input_dim=self.embed_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            output_dim=self.embed_dim,
            activation_fn=activation_fn,
        )
        self.dropout3 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate3 = gate_cls(self.embed_dim)

        if self.output_dim != self.embed_dim:
            self.out_proj: nn.Module = nn.Linear(self.embed_dim, self.output_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, input: Tensor, context: Tensor, is_causal: bool = False) -> Tensor:
        input = self.in_proj(input)
        if self.block_norm_order == "pre":
            self_attn_out = self.self_attn(self.norm1(input), is_causal=is_causal)
            input = self.gate1(input, self.dropout1(self_attn_out))

            cross_attn_out = self.cross_attn(self.norm2(input), context)
            input = self.gate2(input, self.dropout2(cross_attn_out))

            ff_out = self.feedforward(self.norm3(input))
            input = self.gate3(input, self.dropout3(ff_out))
        else:
            self_attn_out = self.self_attn(input, is_causal=is_causal)
            input = self.norm1(self.gate1(input, self.dropout1(self_attn_out)))

            cross_attn_out = self.cross_attn(input, context)
            input = self.norm2(self.gate2(input, self.dropout2(cross_attn_out)))

            ff_out = self.feedforward(input)
            input = self.norm3(self.gate3(input, self.dropout3(ff_out)))

        return self.out_proj(input)

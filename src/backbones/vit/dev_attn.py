import math
from functools import partial
from typing import Optional, Union, Callable, Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear, NonDynamicallyQuantizableLinear
from torch.nn.modules.normalization import LayerNorm


def build_attention(
    embed_dim: int,
    num_heads: int,
    dropout: float,
    batch_first: bool,
    attn_type: str,
    **attn_kwargs,
) -> nn.Module:

    if attn_type.startswith("torch"):
        return MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
            **attn_kwargs,
        )
    elif attn_type.startswith("nested"):
        return MHA(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, **attn_kwargs
        )
    elif attn_type.startswith("xFormers"):
        raise NotImplementedError
        mha = xFormersMHSA(op=memory_efficient_attention)
        return mha
    else:
        raise NotImplementedError


class xFormersMHSA(nn.Module):
    r"""
    Args:
        op (Callable): a function from xformers.ops
    Maybe TODO: rewrite as Multihead Attention Module like:
        here https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
        or
    """

    def __init__(self, op: Callable) -> None:
        super().__init__()
        self.op = op

    def forward(self, x):
        raise NotImplementedError


class MHA(nn.Module):
    r"""Multi-Head Attention Layer

    Basic functionality to wrap around functional attention calls.
    Mostly mimics torch MHA behaviour, assuming
        Batch_first=True, k_dim = v_dim = embed_dim, need_weights=False
    Takes some implementation tricks from here:
        https://github.com/pytorch/pytorch/issues/67999

    TODO:
        1. add support for latent attn
        2. Maybe rewrite as fully self-attention for clarity.
            SDPA only supports fast self-attn and this block is not meant to be used for x-attn anyway.

    Args:
        embed_dim
        num_heads
        dropout
        bias
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        kernel: str | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.k_dim = embed_dim
        self.v_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.kernel = kernel
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # NOTE the code below is equivalent to
        # nn.Linear(embed_dim, 3*embed_dim, bias=bias)

        # NOTE: assumes k_dim == v_dim == embed_dim
        self.in_proj_weight = nn.Parameter(
            # matrix A in for linear(x) = xA^T+b
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def in_projection(self, q, k, v) -> list[Tensor]:
        # TODO this can be simplified with just one linear layer!
        if k is v:
            # self-attn
            if q is k:
                return F.linear(q, self.in_proj_weight, self.in_proj_bias).split(
                    (self.embed_dim, self.k_dim, self.v_dim), dim=-1
                )
            # cross-attn
            else:
                w_q, w_kv = self.in_proj_weight.split(
                    [self.embed_dim, self.k_dim + self.v_dim]
                )
                b_q, b_kv = (
                    None
                    if self.in_proj_bias is None
                    else self.in_proj_bias.split(
                        [self.embed_dim, self.k_dim, self.v_dim]
                    )
                )
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).split(
                    (self.k_dim, self.v_dim), dim=-1
                )

        else:
            w_q, w_k, w_v = self.in_proj_weight.split(
                [self.embed_dim, self.k_dim, self.v_dim]
            )
            b_q, b_k, b_v = (
                None
                if self.in_proj_bias is None
                else self.in_proj_bias.split([self.embed_dim, self.k_dim, self.v_dim])
            )
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def forward(self, q, k, v) -> tuple[Tensor, Optional[Tensor]]:
        q, k, v = self.in_projection(q, k, v)

        q = q.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        k = k.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)

        if self.training:
            dropout = self.dropout_p

        head_wise_out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=dropout,
            is_causal=False,
        )
        merged_out = head_wise_out.transpose(1, 2).flatten(-2)
        out = self.dropout(self.out_proj(merged_out))

        return (out, None)


# class MLA(nn.Module):
#     NOTE: This is just a copy of DeepSeekV3 code
#
#     """
#     Multi-Headed Attention Layer (MLA).

#     Attributes:
#         dim (int): Dimensionality of the input features.
#         n_heads (int): Number of attention heads.
#         n_local_heads (int): Number of local attention heads for distributed systems.
#         q_lora_rank (int): Rank for low-rank query projection.
#         kv_lora_rank (int): Rank for low-rank key/value projection.
#         qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
#         qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
#         qk_head_dim (int): Total dimensionality of query/key projections.
#         v_head_dim (int): Dimensionality of value projections.
#         softmax_scale (float): Scaling factor for softmax in attention computation.
#     """

#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.dim = args.dim
#         self.n_heads = args.n_heads
#         self.n_local_heads = args.n_heads // world_size
#         self.q_lora_rank = args.q_lora_rank
#         self.kv_lora_rank = args.kv_lora_rank
#         self.qk_nope_head_dim = args.qk_nope_head_dim
#         self.qk_rope_head_dim = args.qk_rope_head_dim
#         self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
#         self.v_head_dim = args.v_head_dim

#         if self.q_lora_rank == 0:
#             self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
#         else:
#             self.wq_a = Linear(self.dim, self.q_lora_rank)
#             self.q_norm = RMSNorm(self.q_lora_rank)
#             self.wq_b = ColumnParallelLinear(
#                 self.q_lora_rank, self.n_heads * self.qk_head_dim
#             )
#         self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
#         self.kv_norm = RMSNorm(self.kv_lora_rank)
#         self.wkv_b = ColumnParallelLinear(
#             self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
#         )
#         self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
#         self.softmax_scale = self.qk_head_dim**-0.5
#         if args.max_seq_len > args.original_seq_len:
#             mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
#             self.softmax_scale = self.softmax_scale * mscale * mscale

#         if attn_impl == "naive":
#             self.register_buffer(
#                 "k_cache",
#                 torch.zeros(
#                     args.max_batch_size,
#                     args.max_seq_len,
#                     self.n_local_heads,
#                     self.qk_head_dim,
#                 ),
#                 persistent=False,
#             )
#             self.register_buffer(
#                 "v_cache",
#                 torch.zeros(
#                     args.max_batch_size,
#                     args.max_seq_len,
#                     self.n_local_heads,
#                     self.v_head_dim,
#                 ),
#                 persistent=False,
#             )
#         else:
#             self.register_buffer(
#                 "kv_cache",
#                 torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank),
#                 persistent=False,
#             )
#             self.register_buffer(
#                 "pe_cache",
#                 torch.zeros(
#                     args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim
#                 ),
#                 persistent=False,
#             )

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         """
#         Forward pass for the Multi-Headed Attention Layer (MLA).

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
#             start_pos (int): Starting position in the sequence for caching.
#             freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
#             mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

#         Returns:
#             torch.Tensor: Output tensor with the same shape as the input.
#         """
#         bsz, seqlen, _ = x.size()
#         end_pos = start_pos + seqlen
#         if self.q_lora_rank == 0:
#             q = self.wq(x)
#         else:
#             q = self.wq_b(self.q_norm(self.wq_a(x)))
#         q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
#         q_nope, q_pe = torch.split(
#             q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
#         )
#         q_pe = apply_rotary_emb(q_pe, freqs_cis)
#         kv = self.wkv_a(x)
#         kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
#         k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
#         if attn_impl == "naive":
#             q = torch.cat([q_nope, q_pe], dim=-1)
#             kv = self.wkv_b(self.kv_norm(kv))
#             kv = kv.view(
#                 bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
#             )
#             k_nope, v = torch.split(
#                 kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
#             )
#             k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
#             self.k_cache[:bsz, start_pos:end_pos] = k
#             self.v_cache[:bsz, start_pos:end_pos] = v
#             scores = (
#                 torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
#                 * self.softmax_scale
#             )
#         else:
#             wkv_b = (
#                 self.wkv_b.weight
#                 if self.wkv_b.scale is None
#                 else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
#             )
#             wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
#             q_nope = torch.einsum(
#                 "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
#             )
#             self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
#             self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
#             scores = (
#                 torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
#                 + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
#             ) * self.softmax_scale
#         if mask is not None:
#             scores += mask.unsqueeze(1)
#         scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
#         if attn_impl == "naive":
#             x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
#         else:
#             x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
#             x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
#         x = self.wo(x.flatten(2))
#         return x

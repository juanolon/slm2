from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from shape_extensions import IntVar

    B = IntVar("B")
    L = IntVar("L")
    K = IntVar("K")


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, inputs):
        return F.layer_norm(inputs.float(), [self.dim]) * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time):
        half = self.frequency_embedding_size // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=time.device).float() / half
        )
        angles = time.float()[:, None] * freq[None, :]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))


class Rotary(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer(
            "inv_freq", 1 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        )

    def forward(self, queries, keys):
        pos = torch.arange(queries.shape[-2], device=queries.device).float()
        angles = pos[:, None] * self.inv_freq[None, :]
        angles = torch.cat([angles, angles], dim=-1)[None, None, :, :]
        cos, sin = (angles.cos().to(queries.dtype), angles.sin().to(queries.dtype))

        def rotate(inputs):
            left, right = inputs.chunk(2, dim=-1)
            return inputs * cos + torch.cat([-right, left], dim=-1) * sin

        return (rotate(queries), rotate(keys))


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        self.dropout = dropout

    def forward(self, residual_states, condition, rotary):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(
            condition
        )[:, None, :].chunk(6, dim=-1)
        branch_states = self.norm1(residual_states) * (1 + scale_a) + shift_a
        batch_size, sequence_length, hidden_size = branch_states.shape
        qkv = (
            self.attn_qkv(branch_states)
            .reshape(
                batch_size,
                sequence_length,
                3,
                self.n_heads,
                hidden_size // self.n_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = qkv.unbind(0)
        queries, keys = rotary(queries, keys)
        branch_states = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=0.0, is_causal=False
        )
        branch_states = branch_states.transpose(1, 2).reshape(
            batch_size, sequence_length, hidden_size
        )
        residual_states = residual_states + gate_a * F.dropout(
            self.attn_out(branch_states), p=self.dropout, training=self.training
        )
        branch_states = self.norm2(residual_states) * (1 + scale_m) + shift_m
        return residual_states + gate_m * F.dropout(
            self.mlp(branch_states), p=self.dropout, training=self.training
        )


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden, vocabulary_size, cond):
        super().__init__()
        self.norm_final = LayerNorm(hidden)
        self.linear = nn.Linear(hidden, vocabulary_size)
        self.adaLN_modulation = nn.Linear(cond, 2 * hidden)
        for layer in (self.linear, self.adaLN_modulation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden_states, conditioning):
        shift, scale = self.adaLN_modulation(conditioning)[:, None, :].chunk(2, dim=-1)
        return self.linear(self.norm_final(hidden_states) * (1 + scale) + shift)


class BFN_DIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (
            config.hidden % config.heads == 0 and config.hidden // config.heads % 2 == 0
        )
        self.prob_adapter = nn.Linear(config.vocab, config.hidden, bias=False)
        nn.init.kaiming_uniform_(self.prob_adapter.weight, a=math.sqrt(5))
        self.sigma_map = TimestepEmbedder(config.cond)
        self.rotary_emb = Rotary(config.hidden // config.heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(config.hidden, config.heads, config.cond, config.dropout)
                for _ in range(config.blocks)
            ]
        )
        self.output_layer = DDitFinalLayer(config.hidden, config.vocab, config.cond)

    def forward(
        self, centroids: Tensor[[B, L, K]], time: Tensor[[B]]
    ) -> Tensor[[B, L, K]]:
        assert centroids.ndim == 3 and time.shape == (centroids.shape[0],)
        hidden_states = self.prob_adapter(centroids)
        conditioning = F.silu(self.sigma_map(time))
        for block in self.blocks:
            hidden_states = block(hidden_states, conditioning, self.rotary_emb)
        output = self.output_layer(hidden_states, conditioning)
        assert output.shape == centroids.shape
        return output

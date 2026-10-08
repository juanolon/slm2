#!/usr/bin/env python3
# Exported from slm.sync.ipynb; edit the notebook, then re-export.

from __future__ import annotations

from dataclasses import dataclass

import math

import os

import random

from typing import TYPE_CHECKING, assert_type

import numpy as np

import torch

import torch.nn.functional as F

from shape_extensions import Int, IntVar

from torch import Tensor, nn

B = IntVar('B')

L = IntVar('L')

K = IntVar('K')

N = IntVar('N')

D = IntVar('D')

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def one_hot_tokens(token_ids: Tensor[[B, L]], num_classes: Int[K]) -> Tensor[[B, L, K]]:
    """Convert token IDs into one-hot vectors."""
    assert token_ids.dtype == torch.long
    assert num_classes > 0
    category_ids = torch.arange(num_classes, device=token_ids.device)
    return (token_ids[..., None] == category_ids).to(torch.long)

def encode_ascii_bytes(raw: bytes, alphabet: str) -> Tensor:
    lookup = np.full(256, 255, dtype=np.uint8)
    for i, char in enumerate(alphabet):
        lookup[ord(char)] = i
    data = lookup[np.frombuffer(raw, dtype=np.uint8)]
    if (data == 255).any():
        raise ValueError('raw contains characters outside alphabet')
    return torch.from_numpy(data.copy())

def split_stream(ids: Tensor[[N]], train_end: int, valid_end: int) -> tuple[Tensor, Tensor, Tensor]:
    assert 0 < train_end < valid_end < ids.numel(), 'train_end and valid_end must be in [0, N]'
    return (ids[:train_end], ids[train_end:valid_end], ids[valid_end:])

def gather_windows(ids: Tensor[[N]], starts: Tensor[[B]], length: Int[L]) -> Tensor[[B, L]]:
    assert length > 0 and ids.numel() >= length, 'length must be in [1, N]'
    assert ((starts >= 0) & (starts <= ids.numel() - length)).all(), 'starts must be in [0, N-L]'
    offsets = torch.arange(length, device=ids.device)
    indices = starts[:, None] + offsets[None, :]
    return ids[indices].long()

ALPHABET = ' abcdefghijklmnopqrstuvwxyz'

FAKE = encode_ascii_bytes(b' ab ' * 128, ' ab')

def expected_count(time: Tensor[[B]], vocabulary_size: int) -> Tensor[[B]]:
    assert ((time >= 0) & (time <= 1)).all(), 'time must be in [0,1]'
    return torch.exp(time * math.log(vocabulary_size))

def centroid(candidate_mask: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    check_candidates(candidate_mask)
    c_sum = candidate_mask.sum(-1, keepdim=True)
    return candidate_mask.float() / c_sum

def marginal_probs(clean_tokens: Tensor[[B, L]], time: Tensor[[B]], vocabulary_size: Int[K]) -> Tensor[[B, L, K]]:
    assert clean_tokens.dtype == torch.long
    assert ((clean_tokens >= 0) & (clean_tokens < vocabulary_size)).all()
    e0 = one_hot_tokens(clean_tokens, vocabulary_size).to(time.dtype)
    inclusion_prob = ((expected_count(time, vocabulary_size) - 1) / (vocabulary_size - 1)).clamp(0, 1)
    return e0 + (1 - e0) * inclusion_prob[:, None, None]

def sample_forward(clean_tokens: Tensor[[B, L]], time: Tensor[[B]], vocabulary_size: Int[K], generator=None) -> Tensor[[B, L, K]]:
    predicted_probs = marginal_probs(clean_tokens, time, vocabulary_size)
    uniform_samples = torch.rand(predicted_probs.shape, dtype=predicted_probs.dtype, device=predicted_probs.device, generator=generator)
    return uniform_samples < predicted_probs

def survival_ratio(earlier_time: Tensor[[B]], time: Tensor[[B]], vocabulary_size: int) -> Tensor[[B]]:
    assert ((0 <= earlier_time) & (earlier_time <= time) & (time <= 1)).all(), 'time must be in [0,1] and earlier_time <= time'
    return torch.expm1(earlier_time * math.log(vocabulary_size)) / torch.expm1(time * math.log(vocabulary_size))

def posterior_probs(clean_tokens: Tensor[[B, L]], candidate_mask: Tensor[[B, L, K]], earlier_time: Tensor[[B]], time: Tensor[[B]]) -> Tensor[[B, L, K]]:
    check_candidates(candidate_mask)
    assert candidate_mask.gather(-1, clean_tokens[..., None]).all(), 'clean token must be in candidate set'
    clean_one_hot = one_hot_tokens(clean_tokens, candidate_mask.shape[-1]).to(time.dtype)
    sr = survival_ratio(earlier_time, time, candidate_mask.shape[-1])[:, None, None]
    return clean_one_hot + (1 - clean_one_hot) * candidate_mask * sr

def masked_log_probs(logits: Tensor[[B, L, K]], candidate_mask: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    check_candidates(candidate_mask)
    assert logits.shape == candidate_mask.shape
    mask = logits.masked_fill(~candidate_mask, float('-inf'))
    return F.log_softmax(mask, dim=-1)

def reverse_probs(clean_probabilities: Tensor[[B, L, K]], candidate_mask: Tensor[[B, L, K]], earlier_time: Tensor[[B]], time: Tensor[[B]]) -> Tensor[[B, L, K]]:
    check_candidates(candidate_mask)
    check_simplex(clean_probabilities)
    assert clean_probabilities.shape == candidate_mask.shape and (clean_probabilities[~candidate_mask] == 0).all(), 'clean_probabilities must be zero outside candidate_mask'
    survival = survival_ratio(earlier_time, time, candidate_mask.shape[-1])[:, None, None]
    return candidate_mask.to(clean_probabilities.dtype) * (clean_probabilities + (1 - clean_probabilities) * survival)

def bernoulli_kl(target_probs: Tensor[[B, L, K]], predicted_probs: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    assert target_probs.shape == predicted_probs.shape, 'target_probs and predicted_probs must have the same shape'
    check_probabilities(target_probs)
    check_probabilities(predicted_probs)
    target_probs, predicted_probs = (target_probs.double(), predicted_probs.double())
    return torch.special.xlogy(target_probs, target_probs) - torch.special.xlogy(target_probs, predicted_probs) + torch.special.xlogy(1 - target_probs, 1 - target_probs) - torch.special.xlogy(1 - target_probs, 1 - predicted_probs)

def reconstruction_nll(clean_tokens: Tensor[[B, L]], predicted_probs: Tensor[[B, L, K]]) -> Tensor[[B, L]]:
    check_probabilities(predicted_probs)
    predcited_probs = predicted_probs.double()
    target_one_hot = one_hot_tokens(clean_tokens, predicted_probs.shape[-1]).double()
    return -(torch.special.xlogy(target_one_hot, predicted_probs) + torch.special.xlogy(1 - target_one_hot, 1 - predicted_probs)).sum(-1)

def elbo_term(log_probabilities: Tensor[[B, L, K]], clean_tokens: Tensor[[B, L]], candidate_mask: Tensor[[B, L, K]], timestep_indices: Tensor[[B]], T: int) -> Tensor[[B, L]]:
    assert timestep_indices.dtype == torch.long and ((timestep_indices >= 1) & (timestep_indices <= T)).all()
    time = timestep_indices.double() / T
    earlier_time = (timestep_indices - 1).double() / T
    clean_probs = log_probabilities.double().exp()
    predicted_probs = reverse_probs(clean_probs, candidate_mask, earlier_time, time)
    posterior_params = posterior_probs(clean_tokens, candidate_mask, earlier_time, time)
    kl_div = bernoulli_kl(posterior_params, predicted_probs).sum(-1)
    reconstruction = reconstruction_nll(clean_tokens, predicted_probs)
    return torch.where((timestep_indices == 1)[:, None], reconstruction, kl_div)

def sample_steps(batch: Int[B], T: int, device='cpu', generator=None, stratified=True) -> Tensor[[B]]:
    assert batch > 0 and T > 0, 'batch and T must be positive integers'
    if not stratified:
        return torch.randint(1, T + 1, (batch,), device=device, generator=generator)
    uniform_samples = (torch.arange(batch, device=device, dtype=torch.float64) + torch.rand(batch, device=device, generator=generator)) / batch
    uniform_samples = uniform_samples[torch.randperm(batch, device=device, generator=generator)]
    return (uniform_samples * T).long().clamp(max=T - 1) + 1

def training_loss(log_probabilities: Tensor[[B, L, K]], clean_tokens: Tensor[[B, L]], timestep_indices: Tensor[[B]], T: int, objective: str) -> Tensor[[B, L]]:
    cross_entropy = -log_probabilities.gather(-1, clean_tokens[..., None]).squeeze(-1)
    if objective == 'simple':
        return T * cross_entropy
    elif objective == 'weighted':
        time = timestep_indices.double() / T
        earlier_time = (timestep_indices.double() - 1) / T
        step_weights = 1 - survival_ratio(earlier_time, time, log_probabilities.shape[-1]).to(cross_entropy.dtype)
        return T * step_weights[:, None] * cross_entropy
    else:
        raise ValueError(f"objective must be 'simple' or 'weighted', got {objective}")

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
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, time):
        half = self.frequency_embedding_size // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=time.device).float() / half)
        angles = time.float()[:, None] * freq[None, :]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))

class Rotary(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer('inv_freq', 1 / 10000 ** (torch.arange(0, dim, 2).float() / dim))

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
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate='tanh'), nn.Linear(4 * dim, dim))
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        self.dropout = dropout

    def forward(self, residual_states, condition, rotary):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(condition)[:, None, :].chunk(6, dim=-1)
        branch_states = self.norm1(residual_states) * (1 + scale_a) + shift_a
        batch_size, sequence_length, hidden_size = branch_states.shape
        qkv = self.attn_qkv(branch_states).reshape(batch_size, sequence_length, 3, self.n_heads, hidden_size // self.n_heads).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)
        queries, keys = rotary(queries, keys)
        branch_states = F.scaled_dot_product_attention(queries, keys, values, dropout_p=0.0, is_causal=False)
        branch_states = branch_states.transpose(1, 2).reshape(batch_size, sequence_length, hidden_size)
        residual_states = residual_states + gate_a * F.dropout(self.attn_out(branch_states), p=self.dropout, training=self.training)
        branch_states = self.norm2(residual_states) * (1 + scale_m) + shift_m
        return residual_states + gate_m * F.dropout(self.mlp(branch_states), p=self.dropout, training=self.training)

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
        assert config.hidden % config.heads == 0 and config.hidden // config.heads % 2 == 0
        self.prob_adapter = nn.Linear(config.vocab, config.hidden, bias=False)
        nn.init.kaiming_uniform_(self.prob_adapter.weight, a=math.sqrt(5))
        self.sigma_map = TimestepEmbedder(config.cond)
        self.rotary_emb = Rotary(config.hidden // config.heads)
        self.blocks = nn.ModuleList([DDiTBlock(config.hidden, config.heads, config.cond, config.dropout) for _ in range(config.blocks)])
        self.output_layer = DDitFinalLayer(config.hidden, config.vocab, config.cond)

    def forward(self, centroids: Tensor[[B, L, K]], time: Tensor[[B]]) -> Tensor[[B, L, K]]:
        assert centroids.ndim == 3 and time.shape == (centroids.shape[0],)
        hidden_states = self.prob_adapter(centroids)
        conditioning = F.silu(self.sigma_map(time))
        for block in self.blocks:
            hidden_states = block(hidden_states, conditioning, self.rotary_emb)
        output = self.output_layer(hidden_states, conditioning)
        assert output.shape == centroids.shape
        return output

@dataclass(frozen=True)
class Config:
    vocab: int = 4
    length: int = 12
    T: int = 8
    hidden: int = 32
    heads: int = 4
    blocks: int = 2
    cond: int = 16
    dropout: float = 0.0
    global_batch: int = 8
    microbatch: int = 4
    lr: float = 0.0005
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    warmup: int = 1
    max_steps: int = 80
    ema: float = 0.9
    clip: float = 1.0
    seed: int = 1
    objective: str = 'weighted'
    validate_every: int = 20
    validation_windows: int = 4096
    validation_draws: int = 1
    amp: bool = False

def predict_log_probs(model, candidate_mask: Tensor[[B, L, K]], time: Tensor[[B]]) -> Tensor[[B, L, K]]:
    logits = model(centroid(candidate_mask), time)
    if logits.dtype != torch.float64:
        logits = logits.double()
    return masked_log_probs(logits, candidate_mask)

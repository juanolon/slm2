#!/usr/bin/env python3
# Exported from slm.sync.ipynb; edit the notebook, then re-export.

from __future__ import annotations

import math

import os

import random

from dataclasses import dataclass

from typing import TYPE_CHECKING, assert_type

import numpy as np

import torch

import torch.nn.functional as F

from torch import Tensor

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
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

def assert_shape(inputs, *shape):
    assert isinstance(inputs, torch.Tensor)
    assert tuple(inputs.shape) == tuple(shape), (tuple(inputs.shape), shape)

def assert_probabilities(predicted_probs):
    assert torch.isfinite(predicted_probs).all(), 'NaN/inf probability'
    assert ((predicted_probs >= 0) & (predicted_probs <= 1)).all(), 'outside [0,1]'

def assert_candidates(candidate_mask):
    assert candidate_mask.dtype == torch.bool and candidate_mask.ndim == 3
    assert candidate_mask.any(-1).all(), 'empty candidate set'

def assert_simplex(predicted_probs, atol=1e-06):
    assert_probabilities(predicted_probs)
    torch.testing.assert_close(predicted_probs.sum(-1), torch.ones_like(predicted_probs.sum(-1)), atol=atol, rtol=0)

def one_hot_tokens(token_ids: Tensor[[B, L]], num_classes: Int[K]) -> Tensor[[B, L, K]]:
    """Convert token IDs into one-hot vectors."""
    assert token_ids.dtype == torch.long
    assert num_classes > 0
    category_ids = torch.arange(num_classes, device=token_ids.device)
    return (token_ids[..., None] == category_ids).to(torch.long)

def expected_count(time: Tensor[[B]], vocabulary_size: int) -> Tensor[[B]]:
    assert ((time >= 0) & (time <= 1)).all(), 'time must be in [0,1]'
    return torch.exp(time * math.log(vocabulary_size))

def centroid(candidate_mask: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    assert_candidates(candidate_mask)
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
    assert_candidates(candidate_mask)
    assert candidate_mask.gather(-1, clean_tokens[..., None]).all(), 'clean token must be in candidate set'
    clean_one_hot = one_hot_tokens(clean_tokens, candidate_mask.shape[-1]).to(time.dtype)
    sr = survival_ratio(earlier_time, time, candidate_mask.shape[-1])[:, None, None]
    return clean_one_hot + (1 - clean_one_hot) * candidate_mask * sr

def masked_log_probs(logits: Tensor[[B, L, K]], candidate_mask: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    assert_candidates(candidate_mask)
    assert logits.shape == candidate_mask.shape
    mask = logits.masked_fill(~candidate_mask, float('-inf'))
    return F.log_softmax(mask, dim=-1)

def reverse_probs(clean_probabilities: Tensor[[B, L, K]], candidate_mask: Tensor[[B, L, K]], earlier_time: Tensor[[B]], time: Tensor[[B]]) -> Tensor[[B, L, K]]:
    assert_candidates(candidate_mask)
    assert_simplex(clean_probabilities)
    assert clean_probabilities.shape == candidate_mask.shape and (clean_probabilities[~candidate_mask] == 0).all(), 'clean_probabilities must be zero outside candidate_mask'
    survival = survival_ratio(earlier_time, time, candidate_mask.shape[-1])[:, None, None]
    return candidate_mask.to(clean_probabilities.dtype) * (clean_probabilities + (1 - clean_probabilities) * survival)

def bernoulli_kl(target_probs: Tensor[[B, L, K]], predicted_probs: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    assert target_probs.shape == predicted_probs.shape, 'target_probs and predicted_probs must have the same shape'
    assert_probabilities(target_probs)
    assert_probabilities(predicted_probs)
    target_probs, predicted_probs = (target_probs.double(), predicted_probs.double())
    return torch.special.xlogy(target_probs, target_probs) - torch.special.xlogy(target_probs, predicted_probs) + torch.special.xlogy(1 - target_probs, 1 - target_probs) - torch.special.xlogy(1 - target_probs, 1 - predicted_probs)

def reconstruction_nll(clean_tokens: Tensor[[B, L]], predicted_probs: Tensor[[B, L, K]]) -> Tensor[[B, L]]:
    assert_probabilities(predicted_probs)
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
        step_weights = (1 - survival_ratio(earlier_time, time, log_probabilities.shape[-1])).to(cross_entropy.dtype)
        return T * step_weights[:, None] * cross_entropy
    else:
        raise ValueError(f"objective must be 'simple' or 'weighted', got {objective}")

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
        logits = logits.float()
    return masked_log_probs(logits, candidate_mask)

def repair_empty(sampled, probs):
    assert sampled.dtype == torch.bool and sampled.shape == probs.shape
    assert (probs.max(-1).values > 0).all()
    fallback = one_hot_tokens(probs.argmax(-1), probs.shape[-1]).bool()
    return torch.where(sampled.any(-1, keepdim=True), sampled, fallback)

def prune_step(model, candidate_mask, earlier_time, time, generator=None):
    with torch.no_grad():
        clean_probabilities = predict_log_probs(model, candidate_mask, time.float()).double().exp()
        probs = reverse_probs(clean_probabilities, candidate_mask, earlier_time.double(), time.double())
        uniform_draws = torch.rand(probs.shape, dtype=torch.float64, device=probs.device, generator=generator)
        return repair_empty(uniform_draws < probs, probs)

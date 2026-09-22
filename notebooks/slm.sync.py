# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
import urllib.request
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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

def check_shape(inputs, *shape):
    assert isinstance(inputs, torch.Tensor)
    assert tuple(inputs.shape) == tuple(shape), (tuple(inputs.shape), shape)

def check_probabilities(predicted_probs):
    assert torch.isfinite(predicted_probs).all(), 'NaN/inf probability'
    assert ((predicted_probs >= 0) & (predicted_probs <= 1)).all(), 'outside [0,1]'

def check_candidates(candidate_mask):
    assert candidate_mask.dtype == torch.bool and candidate_mask.ndim == 3
    assert candidate_mask.any(-1).all(), 'empty candidate set'

def check_simplex(predicted_probs, atol=1e-06):
    check_probabilities(predicted_probs)
    torch.testing.assert_close(predicted_probs.sum(-1), torch.ones_like(predicted_probs.sum(-1)), atol=atol, rtol=0)
seed_all(1)
torch.set_num_threads(min(4, os.cpu_count() or 1))

# %% [markdown]
# # Data
# ## Encode
# $$\mathrm{token\_id}(a)=\operatorname{index}(a,\mathrm{alphabet}).$$

# %%
def encode_ascii_bytes(raw: bytes, alphabet: str) -> Tensor:
    lookup = np.full(256, 255, dtype = np.uint8)
    for i, char in enumerate(alphabet):
        lookup[ord(char)] = i

    data = lookup[np.frombuffer(raw, dtype = np.uint8)]
    if (data == 255).any():
        raise ValueError("raw contains characters outside alphabet")

    return torch.from_numpy(data.copy())

# %%
def check_encode_text():
    inputs = encode_ascii_bytes(b' abba ', ' ab')
    assert inputs.dtype == torch.uint8 and inputs.tolist() == [0, 1, 2, 2, 1, 0]
    assert encode_ascii_bytes(b'', ' ab').numel() == 0
    try:
        encode_ascii_bytes(b'?', ' ab')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown characters must fail')
check_encode_text()

# %% [markdown]
# ## Split dataset (text stream) into train, validation, and test sets.
# $$x_{\rm train}=x[0:a],\quad x_{\rm valid}=x[a:b],\quad x_{\rm test}=x[b:N].$$

# Here `a=train_end`, `b=valid_end`, and `N` is the stream length. These are dataset slices, not an SLM probability formula.

# %%
def split_stream(ids: Tensor[[N]], train_end: int, valid_end: int) -> tuple[Tensor, Tensor, Tensor]:
    assert 0 < train_end < valid_end < ids.numel(), 'train_end and valid_end must be in [0, N]'
    return ids[:train_end], ids[train_end:valid_end], ids[valid_end:]


# %%
def check_split_stream():
    train_tokens, validation_tokens, test_tokens = split_stream(torch.arange(100), 90, 95)
    assert (train_tokens.numel(), validation_tokens.numel(), test_tokens.numel()) == (90, 5, 5)
    assert train_tokens[-1] == 89 and validation_tokens[0] == 90 and (test_tokens[0] == 95)
    assert torch.equal(torch.cat([train_tokens, validation_tokens, test_tokens]), torch.arange(100))
check_split_stream()


# %% [markdown]
# ## Gather contiguous training windows
# $$X_{b,\ell}=\mathrm{ids}[\mathrm{starts}_b+\ell],\qquad 0\leq\ell<L.$$

# `starts` chooses the beginning of each contiguous example. `L=length`. This helper does not predict the next character.

# %%
def gather_windows(ids: Tensor[[N]], starts: Tensor[[B]], length: Int[L]) -> Tensor[[B, L]]:
    assert length > 0 and ids.numel() >= length, 'length must be in [1, N]'
    assert ((starts >= 0) & (starts <= ids.numel() - length)).all(), 'starts must be in [0, N-L]'
    offsets = torch.arange(length, device=ids.device)

    # build matrix: [B, 1] + [1, L] = [B, L]
    #             offsets
    #              0  1  2
    # starts  0   [0, 1, 2]
    #         2   [2, 3, 4]
    #         3   [3, 4, 5]
    indices = starts[:, None] + offsets[None, :]

    return ids[indices].long()

# %%
def check_gather_windows():
    got = gather_windows(torch.arange(8, dtype=torch.uint8), torch.tensor([0, 5]), 3)
    assert got.dtype == torch.long
    assert got.tolist() == [[0, 1, 2], [5, 6, 7]]
    check_shape(got, 2, 3)
check_gather_windows()
ALPHABET = ' abcdefghijklmnopqrstuvwxyz'
FAKE = encode_ascii_bytes(b' ab ' * 128, ' ab')
assert len(ALPHABET) == 27

# %% [markdown]
# # Forward process $q(x^c_t | x^c_0)$

# %% [markdown]
# ## Write candidate-count schedule

# $$n(t)=\exp(t\log K)=K^t,\qquad n(0)=1,\quad n(1)=K.$$

# Paper §3.5 uses integer time divided by `T`; we use normalized `time=t` directly.
# `n(t)` is the expected number of surviving candidates, not a fixed integer size.

# %%
def expected_count(time: Tensor[[B]], vocabulary_size: int) -> Tensor[[B]]:
    assert ((time >= 0) & (time <= 1)).all(), 'time must be in [0,1]'
    return torch.exp(time * math.log(vocabulary_size))

# %%
def check_expected_count():
    time = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    update_count = expected_count(time, 16)
    torch.testing.assert_close(update_count, torch.tensor([1.0, 4.0, 16.0], dtype=time.dtype))
    assert (update_count[1:] >= update_count[:-1]).all()
check_expected_count()

# %% [markdown]
# ## Convert a set to its simplex centroid
# $$z_{b,\ell,k}=\frac{c_{b,\ell,k}}{\sum_{v=1}^{K}c_{b,\ell,v}}.$$

# `c` is Boolean membership; `z` is the simplex centroid fed to the backbone. This normalized representation appears in §3.5 and Appendix 8.2.

# %%
def centroid(candidate_mask: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    check_candidates(candidate_mask)
    c_sum = candidate_mask.sum(-1, keepdim=True)

    return candidate_mask.float() / c_sum

# %%
def check_centroid():
    candidate_mask = torch.tensor([[[True, False, True], [False, True, False]]])
    centroids = centroid(candidate_mask)
    torch.testing.assert_close(centroids, torch.tensor([[[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]]))
    check_simplex(centroids)
    if TYPE_CHECKING:
        example = torch.ones(2, 3, 4, dtype=torch.bool)
        assert_type(centroid(example), Tensor[[2, 3, 4]])
check_centroid()

# %% [markdown]
# ## Marginal inclusion probability
# $$\phi_{t,k} = P(c_{t,k} = 1 | x_0)$$
# is the probability that token $k$ belongs to the candidate set at time $t$

# $$\qquad\phi_t=e_{x_0}+(1-e_{x_0})\frac{n(t)-1}{K-1}.$$

# Compute the bernoulli parameter $\phi_{t,k}$ for each token. The candidate vector $c_t = (c_{t,1}, c_{t,2}, \ldots, c_{t,K})$ is a vector of independent Bernoulli random variables. The probability that token $k$ is included in the candidate set at time $t$ is given by $\phi_{t,k}$, which depends on whether the token is the true token $x_0$ or not. If it is the true token, it is always included; otherwise, it is included with probability proportional to the expected count of candidates at time $t$.

# %%
def marginal_probs(clean_tokens: Tensor[[B, L]], time: Tensor[[B]], vocabulary_size: Int[K]) -> Tensor[[B, L, K]]:
    assert clean_tokens.dtype == torch.long
    assert ((clean_tokens >= 0) & (clean_tokens < vocabulary_size)).all()

    e0 = F.one_hot(clean_tokens, vocabulary_size).to(time.dtype)

    inclusion_prob = (
            (expected_count(time, vocabulary_size) - 1) / (vocabulary_size - 1)
            ).clamp(0, 1)

    return e0 + (1 - e0) * inclusion_prob[:, None, None]


# %%
def check_marginal_probs():
    inputs = torch.tensor([[1, 2], [0, 1]])
    predicted_probs = marginal_probs(inputs, torch.tensor([0.0, 1.0]), 3)
    assert torch.equal(predicted_probs[0], F.one_hot(inputs[0], 3).float())
    torch.testing.assert_close(predicted_probs[1], torch.ones(2, 3))
    predicted_probs = marginal_probs(torch.tensor([[2]]), torch.tensor([0.5]), 4)
    torch.testing.assert_close(predicted_probs, torch.tensor([[[1 / 3, 1 / 3, 1.0, 1 / 3]]]))
    check_probabilities(predicted_probs)
check_marginal_probs()


# %% [markdown]
# ## sample forward marginal
# $$q(c_t\mid x_0)=\prod_k\operatorname{Bern}(c_{t,k};\phi_{t,k})$$
# $$U_{b,\ell,k}\sim\mathcal U[0,1),\qquad c_{t,b,\ell,k}=\mathbf1[U_{b,\ell,k}<\phi_{t,b,\ell,k}].$$

Paper Algorithm 1 is independent Bernoulli sampling from the marginal parameters. The number of candidates is random; drawing exactly `round(n(t))` candidates would be a different process.

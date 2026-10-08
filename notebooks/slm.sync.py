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
from dataclasses import dataclass
from __future__ import annotations

import math
import os
import random
from typing import TYPE_CHECKING, assert_type

import numpy as np
import torch
import torch.nn.functional as F
from shape_extensions import Int, IntVar
from torch import Tensor, nn

B = IntVar("B")
L = IntVar("L")
K = IntVar("K")
N = IntVar("N")
D = IntVar("D")


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
    assert torch.isfinite(predicted_probs).all(), "NaN/inf probability"
    assert ((predicted_probs >= 0) & (predicted_probs <= 1)).all(), "outside [0,1]"


def assert_candidates(candidate_mask):
    assert candidate_mask.dtype == torch.bool and candidate_mask.ndim == 3
    assert candidate_mask.any(-1).all(), "empty candidate set"


def assert_simplex(predicted_probs, atol=1e-06):
    assert_probabilities(predicted_probs)
    torch.testing.assert_close(
        predicted_probs.sum(-1),
        torch.ones_like(predicted_probs.sum(-1)),
        atol=atol,
        rtol=0,
    )


seed_all(1)
torch.set_num_threads(min(4, os.cpu_count() or 1))


# reimplement the original F.one_hot but with Pyrefly compatibility
def one_hot_tokens(
    token_ids: Tensor[[B, L]],
    num_classes: Int[K],
) -> Tensor[[B, L, K]]:
    """Convert token IDs into one-hot vectors."""
    assert token_ids.dtype == torch.long
    assert num_classes > 0

    category_ids = torch.arange(num_classes, device=token_ids.device)
    return (token_ids[..., None] == category_ids).to(torch.long)


# %% [markdown]
# # Data
# ## Encode
# $$\mathrm{token\_id}(a)=\operatorname{index}(a,\mathrm{alphabet}).$$


# %%
def encode_ascii_bytes(raw: bytes, alphabet: str) -> Tensor:
    lookup = np.full(256, 255, dtype=np.uint8)
    for i, char in enumerate(alphabet):
        lookup[ord(char)] = i

    data = lookup[np.frombuffer(raw, dtype=np.uint8)]
    if (data == 255).any():
        raise ValueError("raw contains characters outside alphabet")

    return torch.from_numpy(data.copy())


# %%
def check_encode_text():
    inputs = encode_ascii_bytes(b" abba ", " ab")
    assert inputs.dtype == torch.uint8 and inputs.tolist() == [0, 1, 2, 2, 1, 0]
    assert encode_ascii_bytes(b"", " ab").numel() == 0
    try:
        encode_ascii_bytes(b"?", " ab")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown characters must fail")


check_encode_text()

# %% [markdown]
# ## Split dataset (text stream) into train, validation, and test sets.
# $$x_{\rm train}=x[0:a],\quad x_{\rm valid}=x[a:b],\quad x_{\rm test}=x[b:N].$$

# Here `a=train_end`, `b=valid_end`, and `N` is the stream length. These are dataset slices, not an SLM probability formula.


# %%
def split_stream(
    ids: Tensor[[N]], train_end: int, valid_end: int
) -> tuple[Tensor, Tensor, Tensor]:
    assert 0 < train_end < valid_end < ids.numel(), (
        "train_end and valid_end must be in [0, N]"
    )
    return ids[:train_end], ids[train_end:valid_end], ids[valid_end:]


# %%
def check_split_stream():
    train_tokens, validation_tokens, test_tokens = split_stream(
        torch.arange(100), 90, 95
    )
    assert (train_tokens.numel(), validation_tokens.numel(), test_tokens.numel()) == (
        90,
        5,
        5,
    )
    assert (
        train_tokens[-1] == 89 and validation_tokens[0] == 90 and (test_tokens[0] == 95)
    )
    assert torch.equal(
        torch.cat([train_tokens, validation_tokens, test_tokens]), torch.arange(100)
    )


check_split_stream()


# %% [markdown]
# ## Gather contiguous training windows
# $$X_{b,\ell}=\mathrm{ids}[\mathrm{starts}_b+\ell],\qquad 0\leq\ell<L.$$

# `starts` chooses the beginning of each contiguous example. `L=length`. This helper does not predict the next character.


# %%
def gather_windows(
    ids: Tensor[[N]], starts: Tensor[[B]], length: Int[L]
) -> Tensor[[B, L]]:
    assert length > 0 and ids.numel() >= length, "length must be in [1, N]"
    assert ((starts >= 0) & (starts <= ids.numel() - length)).all(), (
        "starts must be in [0, N-L]"
    )
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
    assert_shape(got, 2, 3)


check_gather_windows()
ALPHABET = " abcdefghijklmnopqrstuvwxyz"
FAKE = encode_ascii_bytes(b" ab " * 128, " ab")
assert len(ALPHABET) == 27

# %% [markdown]
# # Forward process $q(x^c_t | x^c_0)$

# %% [markdown]
# Some notations:
# > * Forward transition $q(x_t | x_{t-1})$: given previous step, how do i get next state
# > * Forward marginal $q(x_t | x_0)$: given clean state, what is the dist. at time $t$ (ignoring intermediate steps)
# > * Forward posterior $q(x_{t-1} | x_t, x_0)$: given clean state and state at time $t$, what was the likely state at $t-1$


# %% [markdown]
# ## Write candidate-count schedule

# $$n(t)=\exp(t\log K)=K^t,\qquad n(0)=1,\quad n(1)=K.$$

# Paper §3.5 uses integer time divided by `T`; we use normalized `time=t` directly.
# `n(t)` is the expected number of surviving candidates, not a fixed integer size.


# %%
def expected_count(time: Tensor[[B]], vocabulary_size: int) -> Tensor[[B]]:
    assert ((time >= 0) & (time <= 1)).all(), "time must be in [0,1]"
    return torch.exp(time * math.log(vocabulary_size))


# %%
def check_expected_count():
    time = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    update_count = expected_count(time, 16)
    torch.testing.assert_close(
        update_count, torch.tensor([1.0, 4.0, 16.0], dtype=time.dtype)
    )
    assert (update_count[1:] >= update_count[:-1]).all()


check_expected_count()

# %% [markdown]
# ## Convert a set to its simplex centroid
# $$z_{b,\ell,k}=\frac{c_{b,\ell,k}}{\sum_{v=1}^{K}c_{b,\ell,v}}.$$

# `c` is Boolean membership; `z` is the simplex centroid fed to the backbone. This normalized representation appears in §3.5 and Appendix 8.2.


# %%
def centroid(candidate_mask: Tensor[[B, L, K]]) -> Tensor[[B, L, K]]:
    assert_candidates(candidate_mask)
    c_sum = candidate_mask.sum(-1, keepdim=True)

    return candidate_mask.float() / c_sum


# %%
def check_centroid():
    candidate_mask = torch.tensor([[[True, False, True], [False, True, False]]])
    centroids = centroid(candidate_mask)
    torch.testing.assert_close(
        centroids, torch.tensor([[[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]])
    )
    assert_simplex(centroids)
    if TYPE_CHECKING:
        example = torch.ones(2, 3, 4, dtype=torch.bool)
        assert_type(centroid(example), Tensor[[2, 3, 4]])


check_centroid()

# %% [markdown]
# # Forward candidate Appending (Paper Sec. 3.1)

# %% [markdown]
# ## Marginal probability (Derive Bernoulli parameter $\phi_{t,k}$)
# $$\phi_{t,k} = P(c_{t,k} = 1 | x_0)$$
# Because $c_{t,k}$ is binary, we know
# $$P(c_{t,k} = 0 | x_0) = 1 - \phi_{t,k}$$
# This is a Bernoulli distribution: $$c_{t,k} | x_0 \sim \operatorname{Bern}(\phi_{t,k})$$
# Or $$q(c_{t,k} | x_0) = \operatorname{Bern}(c_{t,k}; \phi_{t,k})$$

# We want $\phi_t$, so:
#
# The candidate vector is binary:
# $$
# c_t = (c_{t,1}, c_{t,2}, \ldots, c_{t,K}) \in \{0, 1\}^K
# $$
# $c_{t,k} = 1$, token is included, else $c_{t,k} = 0$, token is excluded.

# SLM chooses the expected candidate-set size
# $$
# n(t) = K^t
# $$
# Where $n(0) = K^0 = 1$ and $n(1) = K^1 = K$.
#
# > Equivalently: $\log n(t) = t \log K$, the expected candidate count grows linearly with time.
#
# Summing over all entries tells us how uncertain the state is
# $$
# | c_t | = \sum_{k=1}^{K} c_{t,k}
# $$
# being $| c_t | = 1$ means we are certain about the token, else $| c_t | = K$ every token could be possible.

# There are $K$ tokens total, $1$ true token and $K - 1$ incorrect tokens
# Since $c_t$ is random, the expected number of included tokens is
# $$
# \begin{align}
# \mathbb{E}[|c_t| | x_0] &= \mathbb{E} \big[ \sum_{k=1}^{K} c_{t,k} | x_0 \big] \\
# \mathbb{E}[|c_t| | x_0] &= \sum_{k=1}^{K} \mathbb{E} \big[ c_{t,k} | x_0 \big]
# \end{align}
# $$
# Since $c_{t,k}$ is Bernoulli ($\mathbb{E} [c_{t,k}] = P(c_{t,k} = 1 | x_0) = \phi_{t,k}$), we have
# $$
# \mathbb{E}[|c_t| | x_0] = \sum_{k=1}^{K}  \phi_{t,k} \\
# $$
# Let their **shared inclusion probability** be $p_t$, then the expected number of included tokens is
# $$
# \mathbb{E}[|c_t| | x_0] = 1 + (K-1)p_t
# $$
#
# To derive $p_t$: We want this expectation to be equal $n(t)$:
# $$
# 1 + (K-1)p_t = n(t)
# $$
# Solving for $n(t)$,
# $$
# p_t = \frac{n(t) - 1}{K - 1}
# $$
#
# Therefore,
# $$
# \phi_{t,k} =
# \begin{cases}
#       1                   , & k=x_0,\\
#       \frac{n(t)-1}{K-1}  , & k\neq x_0.
# \end{cases}
# $$
#
# Now we can write $\phi_t$ as a vector of length $K$: $\phi_t = (\phi_{t,1}, \phi_{t,2}, \ldots, \phi_{t,K})$.
# Or using the one-hot vector $e_{x_0}$:
# $$
# \phi_t = e_{x_0} + (1 - e_{x_0}) \frac{n(t) - 1}{K - 1}
# $$


# Which is the probability that token $k$ belongs to the candidate set at time $t$:
# * For the true token $e_{x_0,k} = 1$:
# $$
# \sigma_{t,k} = 1 + (1 - 1) p_t = 1
# $$
# * For the incorrect tokens $e_{x_0,k} = 0$:
# $$
# \sigma_{t,k} = 0 + (1 - 0) p_t = p_t
# $$


# %%
def marginal_probs(
    clean_tokens: Tensor[[B, L]], time: Tensor[[B]], vocabulary_size: Int[K]
) -> Tensor[[B, L, K]]:
    assert clean_tokens.dtype == torch.long
    assert ((clean_tokens >= 0) & (clean_tokens < vocabulary_size)).all()

    e0 = one_hot_tokens(clean_tokens, vocabulary_size).to(time.dtype)

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
    torch.testing.assert_close(
        predicted_probs, torch.tensor([[[1 / 3, 1 / 3, 1.0, 1 / 3]]])
    )
    assert_probabilities(predicted_probs)


check_marginal_probs()

# %% [markdown]
# Notation:
# * $x^c_t$ corresponds to a complete candidate vector $c_t$
# * $c_{t,k}$ corresponds to a complete candidate vector $c_t$


# %% [markdown]
# ## Forward marginal sampling (Paper Eq. 4)
# We can use **inverse-transform sampling** to sample from the Bernoulli distribution. For each candidate $c_{t,k}$, we can draw a uniform random number $U_{b,l,k} \sim \mathcal U[0,1)$ and include the token if $U_{b,l,k} < \phi_{t,b,l,k}$.
#
# $$c_{t,b,\ell,k}=\mathbf1[U_{b,\ell,k}<\phi_{t,b,\ell,k}].$$
#
# This is possible because $c_{t,k}$ are independently sampled from $U_k$, so the probability of obtaining a particular vector is:
# $$
# P(c_t | x_0) = \prod_k P(c_{t,k} | x_0) = q(c_t | x_0) = \prod_k\operatorname{Bern}(c_{t,k};\phi_{t,k})
# $$
# $$
# q(c_t | x_0) = \operatorname{Bern}(e_{x_0} + (1 - e_{x_0}) \frac{n(t) - 1}{K - 1})
# $$


# %%
def sample_forward(
    clean_tokens: Tensor[[B, L]],
    time: Tensor[[B]],
    vocabulary_size: Int[K],
    generator=None,
) -> Tensor[[B, L, K]]:
    predicted_probs = marginal_probs(clean_tokens, time, vocabulary_size)

    uniform_samples = torch.rand(
        predicted_probs.shape,
        dtype=predicted_probs.dtype,
        device=predicted_probs.device,
        generator=generator,
    )

    return uniform_samples < predicted_probs


# %%
def check_sample_forward():
    inputs = torch.zeros(12000, 1, dtype=torch.long)
    generator = torch.Generator().manual_seed(2026)
    candidate_mask = sample_forward(inputs, torch.full((12000,), 0.5), 4, generator)
    assert_candidates(candidate_mask)
    assert candidate_mask[:, :, 0].all()
    assert abs(candidate_mask.float().sum(-1).mean().item() - 2) < 0.04
    first_wrong_included, second_wrong_included = (
        candidate_mask[:, 0, 1].float(),
        candidate_mask[:, 0, 2].float(),
    )
    assert (
        abs((first_wrong_included * second_wrong_included).mean().item() - 1 / 9) < 0.02
    )
    assert sample_forward(inputs[:2], torch.ones(2), 4).all()
    assert sample_forward(inputs[:2], torch.zeros(2), 4).sum(-1).eq(1).all()


check_sample_forward()

# %% [markdown]
# ## Reverse survival ratio (Paper Eq. 5, 6, 10)

# $$\begin{align}
# r(s, t) &= \frac{n(s) - 1}{n(t) - 1} = \frac{\exp(s \log K) - 1}{\exp(t \log K) - 1} \\
#         &= \frac{\operatorname{expm1}(s \log K)}{\operatorname{expm1}(t \log K)}
# \end{align}$$
# where $\operatorname{expm1}$ is the numerically stable function $\operatorname{expm1}(x) = \exp(x) - 1$.
#
# This is the probability that a wrong category survives the backward transition (when moving from time $t$ back to an earlier time $s$), conditional on being present now.


# %%
def survival_ratio(
    earlier_time: Tensor[[B]], time: Tensor[[B]], vocabulary_size: int
) -> Tensor[[B]]:
    assert ((0 <= earlier_time) & (earlier_time <= time) & (time <= 1)).all(), (
        "time must be in [0,1] and earlier_time <= time"
    )
    return torch.expm1(earlier_time * math.log(vocabulary_size)) / torch.expm1(
        time * math.log(vocabulary_size)
    )


# %%
def check_survival_ratio():
    retention_ratio = survival_ratio(
        torch.tensor([0.0, 0.5], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        4,
    )
    torch.testing.assert_close(
        retention_ratio, torch.tensor([0.0, 1 / 3], dtype=torch.float64)
    )
    retention_ratio = survival_ratio(
        torch.tensor([1e-10], dtype=torch.float64),
        torch.tensor([2e-10], dtype=torch.float64),
        27,
    )
    assert abs(retention_ratio.item() - 0.5) < 1e-08


check_survival_ratio()


# %% [markdown]
# ## Forward posterior distribution (Paper Eq. 5)
# $$
# q(x_{t-1}\mid x_t,x_0) = \operatorname{Bern}\left(
# x_0+
# \left[(1-x_0)\odot x_t\right]
# \frac{n(t-1)-1}{n(t)-1}
# \right)
# $$


# %%
def posterior_probs(
    clean_tokens: Tensor[[B, L]],
    candidate_mask: Tensor[[B, L, K]],
    earlier_time: Tensor[[B]],
    time: Tensor[[B]],
) -> Tensor[[B, L, K]]:
    assert_candidates(candidate_mask)
    assert candidate_mask.gather(-1, clean_tokens[..., None]).all(), (
        "clean token must be in candidate set"
    )

    clean_one_hot = one_hot_tokens(clean_tokens, candidate_mask.shape[-1]).to(
        time.dtype
    )
    sr = survival_ratio(earlier_time, time, candidate_mask.shape[-1])[:, None, None]

    return clean_one_hot + (1 - clean_one_hot) * candidate_mask * sr


# %%
def check_posterior_probs():
    inputs = torch.tensor([[0]])
    candidate_mask = torch.tensor([[[True, True, False, True]]])
    posterior_params = posterior_probs(
        inputs, candidate_mask, torch.tensor([0.5]), torch.tensor([1.0])
    )
    torch.testing.assert_close(
        posterior_params, torch.tensor([[[1.0, 1 / 3, 0.0, 1 / 3]]])
    )
    q0 = posterior_probs(
        inputs, candidate_mask, torch.tensor([0.0]), torch.tensor([0.5])
    )
    torch.testing.assert_close(q0, F.one_hot(inputs, 4).float())


check_posterior_probs()


# %% [markdown]
# Notation:
# Since $x^c$ is binary, we can count its ones with $\mathbf 1^T$:
# $$
# \mathbf 1^T x^c = \sum_{k=1}^{K} c_k = \mid x^c \mid
# $$
# * $\mathbf 1^T x^c > 0$: The set should not be empty
# * $\mathbf 1^T x^c_t \geq \mathbf 1^T x^c_{t-1}$: The number of candidates cannot decrease in the forward process
# * $ [x_t^c]^T x_{t-1}^c=\mathbf 1^T x_{t-1}^c$: The candidates must remain in the forward process
#

# %% [markdown]
# ## Sampling
# The forward process starts with one clean token: $x_0^c = e_{x_0}$
# and it ends with every token being a candidate $x_T^c = \mathbf 1$
# Conceptually:
# $$
# \{x_0\}=C_0\subseteq C_1\subseteq\cdots\subseteq C_T = \{1,\ldots,K\}.
# $$
# > Candidates may be added, but never removed
# 1. The candidate count must not decrease
# $$
# \mathbf 1^T x_{t-1}^c \leq \mathbf 1^T x_t^c
# $$
# Or $\mid C_{t-1} \mid \leq \mid C_t \mid$.
# 2. Every previous candidate must remain
# $$
# [x_t^c]^T x_{t-1}^c=\mathbf 1^T x_{t-1}^c
# $$
# We can write $\mid C_t \cap C_{t-1} \mid = \mid C_{t-1} \mid$.
#
# This is also only possible when all early candidates are still present:
# $$
# C_{t-1}\subseteq C_t
# $$
#
# ---
#
# The forward transition:
# $$
# q(x_t^c\mid x_{t-1}^c) = \operatorname{Bern}(\phi)
# $$
# with (eq. 3)
# $$
# \phi = x_{t-1}^c + (1-x_{t-1}^c) \frac{n(t)-n(t-1)}{K-n(t-1)}
# $$
# We have two cases:
# 1. Token already present $$[x^c_{t-1}]_k = 1$$
# then $$\phi_k = 1 + (1-1)a = 1$$
# thus $$[x^c_t]_k \sim \operatorname{Bern}(1)$$
#
# 2. Token currently absent
# If $$[x^c_{t-1}]_k = 0$$
# then $$\phi_k = 0 + (1-0)a = a$$
# so the token may be added with probability $a$
#
# This direcly implementes candidate appending
# $$
# C_{t-1}\subseteq C_t
# $$
#
# ---
#
# The reverse process should instead remove candidates $C_t\supseteq C_{t-1}$. Starting from $C_T=\{1,\ldots,K\}$, remove candidates until only the clean token remain $C_0 = \{x_0\}$
#
# We can calculate the exact reverse posterior as we already have seen:
# $$
# q(c_s\mid c_t,x_0) = \operatorname{Bern}(\psi)
# $$
# where
# $$
# \psi = e_{x_0} + [(1-e_{x_0})\odot c_t]r(s,t)
# $$
# and
# $$
# r(s,t) = \frac{n(s)-1}{n(t)-1}
# $$
#
# We have three possible cases:
# 1. clean token
# $$
# [e_{x_0}]_k = 1 \implies \psi_k = 1 + (1-1)r(s,t) = 1
# $$
# The clean token must survive
# 2. An incorrect token still present in $c_t$
# $$
# [e_{x_0}]_k = 0 \text{ and } [c_t]_k = 1 \implies \psi_k = 0 + (1-0)r(s,t) = r(s,t)
# $$
# It survives backward to the earlier time with probability $r(s, t)$.
# A token absent from $c_t$
# $$
# c{t,k} = 0 \implies \psi_k = 0 + (1-0)0r(s,t) = 0
# $$
# A token already absent from $C_t$ cannot reappear in the smaller earlier set $C_s$. This guarantees $C_s \subseteq C_t$.
#
# ---
#
# During generation $x_0$ is unknown. The network predicts a categorical distribution over the possible clean token
# $$
# [\operatorname{NN}_\theta(x_t^c,t)]^T (\mathbf1-x_t^c) = 0
# $$
# Using $p_\theta = \operatorname{NN}_\theta(x_t^c,t)$ and expanding the dot product:
# $$
# \sum_k p_{\theta,k}(1-[x_t^c]_k)=0.
# $$
# We have two cases:
# 1. The candidate is 1 $[x_t^c]_k = 1$, the token contributes nothing to the sum
# 2. The candidate is 0 $[x_t^c]_k = 0$, the token contributes $p_{\theta,k}$ to the sum
#
# Since the sum must be zero, the network must assign zero probability to all tokens not present in the candidate set. This guarantees that the network will never predict a token that is not present in the candidate set.


# %% [markdown]
# ## Restrict prediction to candidate set (Paper Seq. 3.2)
# The network return unrestricted logits:
# $$
# \ell=(\ell_1,\ldots,\ell_K)
# $$
# The softmax will be then:
# $$
# p_{\theta,k} = \frac{\exp(\ell_k)} {\sum_v\exp(\ell_v)}
# $$
# which assigns non-zero probability to all tokens. We want to restrict the network to only predict candidates:
# We can set
# $$
# \widetilde\ell_k = \begin{cases} \ell_k,&c_{t,k}=1,\\ -\infty,&c_{t,k}=0. \end{cases}
# $$
# exponentiating gives:
# $$
# \exp(\widetilde\ell_k) = \begin{cases} \exp(\ell_k),&c_{t,k}=1,\\ 0,&c_{t,k}=0. \end{cases}
# $$
# since $c_{t,k}$ is binary we can write the condition as:
# $$
# \exp(\widetilde\ell_k) = c_{t,k} \exp(\ell_k)
# $$
# Now the softmax looks like
# $$
# p_{\theta,k} = \frac{c_{t,k}\exp(\ell_k)} {\sum_vc_{t,v}\exp(\ell_v)}
# $$
#
# Finally, the cross-entropy loss needs the $\log p_{\theta, k}$, so we take the logarithm of it

# $$
# \begin{aligned}
# \log p_{\theta,k} &= \log\exp(\widetilde\ell_k) - \log\sum_v\exp(\widetilde\ell_v) \\
#             &= \widetilde\ell_k - \operatorname{logsumexp}_v(\widetilde\ell_v)
# \end{aligned}
# $$


# %%
def masked_log_probs(
    logits: Tensor[[B, L, K]], candidate_mask: Tensor[[B, L, K]]
) -> Tensor[[B, L, K]]:
    assert_candidates(candidate_mask)
    assert logits.shape == candidate_mask.shape
    mask = logits.masked_fill(~candidate_mask, float("-inf"))
    return F.log_softmax(mask, dim=-1)


# %%
def check_masked_log_probs():
    centroids = torch.tensor([[[0.0, 999.0, math.log(3.0)]]], requires_grad=True)
    candidate_mask = torch.tensor([[[True, False, True]]])
    log_probabilities = masked_log_probs(centroids, candidate_mask)
    torch.testing.assert_close(
        log_probabilities.exp(), torch.tensor([[[0.25, 0.0, 0.75]]])
    )
    assert torch.isneginf(log_probabilities[0, 0, 1])
    (-log_probabilities[0, 0, 0]).backward()
    assert torch.isfinite(centroids.grad).all() and centroids.grad[0, 0, 1] == 0


check_masked_log_probs()

# %% [markdown]
# # Reverse candidate pruning (Paper Seq. 3.2)

# %% [markdown]
# ## Parameterized reverse transition (Paper Eq. 6)
# The posterior can be formulate using a neural network $\theta$ like:
# $$
# p_\theta(x_{t-1}^c\mid x_t^c) = \operatorname{Bern}\left( \left[ \operatorname{NN}_\theta(x_t^c,t) + \left(1-\operatorname{NN}_\theta(x_t^c,t)\right) \frac{n(t-1)-1}{n(t)-1} \right] \odot x_t^c \right)
# $$


# %%
def reverse_probs(
    clean_probabilities: Tensor[[B, L, K]],
    candidate_mask: Tensor[[B, L, K]],
    earlier_time: Tensor[[B]],
    time: Tensor[[B]],
) -> Tensor[[B, L, K]]:
    assert_candidates(candidate_mask)
    assert_simplex(clean_probabilities)
    assert (
        clean_probabilities.shape == candidate_mask.shape
        and (clean_probabilities[~candidate_mask] == 0).all()
    ), "clean_probabilities must be zero outside candidate_mask"

    survival = survival_ratio(earlier_time, time, candidate_mask.shape[-1])[
        :, None, None
    ]
    return candidate_mask.to(clean_probabilities.dtype) * (
        clean_probabilities + (1 - clean_probabilities) * survival
    )


# %%
def check_reverse_probs():
    predicted_probs = torch.tensor([[[0.25, 0.75, 0.0, 0.0]]])
    candidate_mask = torch.tensor([[[True, True, False, False]]])
    output = reverse_probs(
        predicted_probs, candidate_mask, torch.tensor([0.5]), torch.tensor([1.0])
    )
    torch.testing.assert_close(output, torch.tensor([[[0.5, 5 / 6, 0.0, 0.0]]]))
    torch.testing.assert_close(
        reverse_probs(
            predicted_probs, candidate_mask, torch.tensor([0.0]), torch.tensor([1.0])
        ),
        predicted_probs,
    )
    inputs = torch.tensor([[1]])
    oracle = F.one_hot(inputs, 4).float()
    torch.testing.assert_close(
        reverse_probs(oracle, candidate_mask, torch.tensor([0.5]), torch.tensor([1.0])),
        posterior_probs(
            inputs, candidate_mask, torch.tensor([0.5]), torch.tensor([1.0])
        ),
    )


check_reverse_probs()

# %% [markdown]
# # Training Procedure (Paper Seq. 3.3)

# %% [markdown]
# ## Bernoulli KL (Paper Eq. 8)
# The KL divergence between two multivariate Bernoulli distributions is:
# > https://math.stackexchange.com/questions/2604566/kl-divergence-between-two-multivariate-bernoulli-distribution
#
# $$
# D_{\mathrm{KL}} \left[ \operatorname{Bern}(q) \| \operatorname{Bern}(p) \right] = \sum_{k:c_{t,k}=1} \left[ q_k\log\frac{q_k}{p_k} + (1-q_k)\log\frac{1-q_k}{1-p_k} \right]
# $$
# The paper uses:
# * $q_k = \text{gd}^k(c_t)$: exact posterior parameter (`posterior_params`)
# * $p_k = \text{pred}_\theta^k(c_t)$: learned reverse parameter (`predicted_probs`)
#
# Using the log identity $\log(a/b) = \log a - \log b$ we can write the KL (for one token) as:
# $$
# d(q, p) = q \log q - q \log p + (1 - q) \log (1 - q) - (1 - q) \log (1 - p)
# $$
#
# > The $D_{\mathrm{KL}} = \sum_k d(\psi_k,\widehat\psi_k) $ is the sum of the scalar bernoulli KLs for each token $k$. `bernoulli_kl` return only $d$.


# %%
def bernoulli_kl(
    target_probs: Tensor[[B, L, K]], predicted_probs: Tensor[[B, L, K]]
) -> Tensor[[B, L, K]]:
    assert target_probs.shape == predicted_probs.shape, (
        "target_probs and predicted_probs must have the same shape"
    )
    assert_probabilities(target_probs)
    assert_probabilities(predicted_probs)
    target_probs, predicted_probs = (target_probs.double(), predicted_probs.double())
    return (
        torch.special.xlogy(target_probs, target_probs)
        - torch.special.xlogy(target_probs, predicted_probs)
        + torch.special.xlogy(1 - target_probs, 1 - target_probs)
        - torch.special.xlogy(1 - target_probs, 1 - predicted_probs)
    )


# %%
def check_bernoulli_kl():
    posterior_params = torch.tensor([[[0.0, 1.0, 0.5]]], dtype=torch.float64)
    torch.testing.assert_close(
        bernoulli_kl(posterior_params, posterior_params),
        torch.zeros_like(posterior_params),
    )
    predicted_probs = torch.tensor([[[0.2, 0.8, 0.25]]], dtype=torch.float64)
    expected = torch.tensor(
        [[[-math.log(0.8), -math.log(0.8), 0.5 * math.log(4 / 3)]]], dtype=torch.float64
    )
    torch.testing.assert_close(
        bernoulli_kl(posterior_params, predicted_probs), expected
    )
    assert torch.isinf(bernoulli_kl(torch.ones(1, 1, 1), torch.zeros(1, 1, 1))).all()


check_bernoulli_kl()

# %% [markdown]
# Notation: The probability mass function of a Bernoulli:
# $$
# \operatorname{Bern}(x;p) = p^x (1 - p)^{1-p}
# $$
# This describes that sampling $1$ with probability $p$ and samping $0$ with probability $1-p$.
#
# For a multivariate Bernoulli, the probability of a vector $\mathbf x$ is the product of the probabilities of each entry:

# $$
# \operatorname{Bern}(\mathbf x;\boldsymbol\psi) = \prod_{k=1}^K P(X_k = x_k) = \prod_{k=1}^K \psi_k^{x_k}(1-\psi_k)^{1-x_k}.
# $$

# %% [markdown]
# ## Reconstruction term (Paper section 3.3)
# $$
# L_0 = -\mathbb E_{q(x_1^c\mid x_0^c)} \left[ \log p_\theta(x_0^c\mid x_1^c) \right].
# $$
# We set $t = 1$, so $t-1 = 0$ to the learned reverse transition $p_\theta(x_0^c\mid x_1^c)$.
#
# The reverse survival ration become $0$, therefore
# $$
# p_\theta(x_0^c\mid x_1^c) = \operatorname{Bern}(x_0^c; \operatorname{NN}_\theta(x_1^c, 1) \odot x_1^c)
# $$
#
# We can now expand the conditional bernoulli probability:
# $$
# p_\theta(x_0^c\mid x_1^c) = \prod_k \widehat\psi_k^{\,x_{0,k}^c} (1-\widehat\psi_k)^{1-x_{0,k}^c}.
# $$
# Take the logarithm
# $$
# \log p_\theta(x_0^c\mid x_1^c) = \sum_k \left[ x_{0,k}^c\log\widehat\psi_k + (1-x_{0,k}^c)\log(1-\widehat\psi_k) \right]
# $$
# We rewrite this sum as a inner product:
# 1. True-token term (first term):
# Becasue $x_0^c$ is one-hot, only one component is $1$. Suppose the true token is $j$:
# $$
# \left\langle \widehat{\boldsymbol\psi},x_0^c \right\rangle = \sum_k\widehat\psi_kx_{0,k}^c = \widehat\psi_j.
# $$
#
# Thus,
# $$
# \sum_kx_{0,k}^c\log\widehat\psi_k = \log \left\langle \widehat{\boldsymbol\psi},x_0^c \right\rangle
# $$
#
# 2. Wrong-token term (second term):
# The factor $1 - x_{0,k}^c$ is $1$ for all tokens except the true token.
#
# Tokens outside the candidate set also get $0$, because we have:
# $$
# x_{1,k}^c=0 \quad\Longrightarrow\quad \widehat\psi_k=0
# $$
# thus, $\log(1-\widehat\psi_k) = 0$
#
# Since the true token $x^c_0$ always belongs to the candidate set $x^c_0 \subseteq x^c_1$, consider $x_{1,k}^c-x_{0,k}^c$:
#
# | Token type | $x_{1,k}^c$ | $x_{0,k}^c$ | $x_{1,k}^c-x_{0,k}^c$ |
# |---|---:|---:|---:|
# | True token | 1 | 1 | 0 |
# | Wrong current candidate | 1 | 0 | 1 |
# | Outside candidate set | 0 | 0 | 0 |
#
# $$
# \sum_k (1-x_{0,k}^c)\log(1-\widehat\psi_k) = \sum_k (x_{1,k}^c-x_{0,k}^c)\log(1-\widehat\psi_k)
# $$
#
# We can now rewrite this as an inner product:
# $$
# \sum_k (x_{1,k}^c-x_{0,k}^c) \log(1-\widehat\psi_k) = \left\langle x_1^c-x_0^c, \log(1-\widehat{\boldsymbol\psi}) \right\rangle
# $$
#
# Combining both rewrites, we have:
# $$
# \log p_\theta(x_0^c\mid x_1^c) = \log \left\langle \widehat{\boldsymbol\psi},x_0^c \right\rangle + \left\langle x_1^c-x_0^c, \log(1-\widehat{\boldsymbol\psi}) \right\rangle
# $$
#
# The first term is the probability that the true token survives and the second term is the probability that every wrong current candidate disappears


# %%
def reconstruction_nll(
    clean_tokens: Tensor[[B, L]], predicted_probs: Tensor[[B, L, K]]
) -> Tensor[[B, L]]:
    assert_probabilities(predicted_probs)
    predcited_probs = predicted_probs.double()
    target_one_hot = one_hot_tokens(clean_tokens, predicted_probs.shape[-1]).double()
    return -(
        torch.special.xlogy(target_one_hot, predicted_probs)
        + torch.special.xlogy(1 - target_one_hot, 1 - predicted_probs)
    ).sum(-1)


# %%
def check_reconstruction_nll():
    predicted_probs = torch.tensor([[[0.25, 0.75, 0.0]]], dtype=torch.float64)
    loss = reconstruction_nll(torch.tensor([[0]]), predicted_probs)
    torch.testing.assert_close(
        loss, torch.tensor([[-2 * math.log(0.25)]], dtype=torch.float64)
    )
    assert (
        reconstruction_nll(
            torch.tensor([[1]]), torch.tensor([[[0.0, 1.0, 0.0]]])
        ).item()
        == 0
    )


check_reconstruction_nll()

# %% [markdown]
# ## Unified $\ell_{j-1}$ loss (Paper Eq. 7 + $L_0$)
# Paper Eq. 7 $L_{j-1}$
# $$
# L_{j-1} = \mathbb E_{q(c_j\mid x_0)} \left[ D_{\mathrm{KL}} \left( \operatorname{Bern}(\operatorname{gd}(c_j)) \| \operatorname{Bern}(\operatorname{pred}_\theta(c_j)) \right) \right]
# $$
# and Eq. $L_0$:
# $$
# L_0 = -\mathbb E_{q(c_1\mid x_0)} \left[ \log p_\theta(x_0^c\mid c_1) \right]
# $$
# combined in one function

# $$
# \ell_{j-1}(x_0,c_j) = \begin{cases} -\log p_\theta(x_0\mid c_1),&j=1,\\[2mm] D_{\mathrm{KL}}\left(q\|p_\theta\right),&j>1. \end{cases}
# $$
#
# > $L_{j-1}$ now becomes
# $$
# L_{j-1} = \mathbb E_{q(c_j\mid x_0)}[\ell_{j-1}(x_0,c_j)]
# $$


# %%
def elbo_term(
    log_probabilities: Tensor[[B, L, K]],
    clean_tokens: Tensor[[B, L]],
    candidate_mask: Tensor[[B, L, K]],
    timestep_indices: Tensor[[B]],
    T: int,
) -> Tensor[[B, L]]:
    assert (
        timestep_indices.dtype == torch.long
        and ((timestep_indices >= 1) & (timestep_indices <= T)).all()
    )

    time = timestep_indices.double() / T
    earlier_time = (timestep_indices - 1).double() / T
    clean_probs = log_probabilities.double().exp()

    predicted_probs = reverse_probs(clean_probs, candidate_mask, earlier_time, time)
    posterior_params = posterior_probs(clean_tokens, candidate_mask, earlier_time, time)

    kl_div = bernoulli_kl(posterior_params, predicted_probs).sum(-1)
    reconstruction = reconstruction_nll(clean_tokens, predicted_probs)

    return torch.where((timestep_indices == 1)[:, None], reconstruction, kl_div)


# %%
def check_elbo_term():
    inputs = torch.tensor([[0], [0]])
    candidate_mask = torch.ones(2, 1, 2, dtype=torch.bool)
    log_probabilities = torch.full((2, 1, 2), -math.log(2.0), dtype=torch.float64)
    value = elbo_term(
        log_probabilities, inputs, candidate_mask, torch.tensor([1, 2]), 2
    )
    assert torch.isfinite(value).all() and (value >= 0).all()
    torch.testing.assert_close(
        value[0], torch.tensor([2 * math.log(2.0)], dtype=torch.float64)
    )
    oracle = F.one_hot(inputs, 2).double().log()
    torch.testing.assert_close(
        elbo_term(oracle, inputs, candidate_mask, torch.tensor([1, 2]), 2),
        torch.zeros(2, 1, dtype=torch.float64),
    )


check_elbo_term()

# %% [markdown]
# ## Stratified sampling of timesteps
# Sampling timesteps for a batch uniformly $J \sim \operatorname{Uniform}\{1,\ldots,T\}$ could lead to clustered timesteps. E.g. ($T = 8$ and $B = 4$):
# $$
# J = (5, 5, 6, 5)
# $$
#
# Stratified sampling divides the batch into $B$ strata and samples one timestep from each stratum. This guarantees that the timesteps are evenly distributed across the batch.
#
# Start with a interval $[0, 1)$, which represents the complete difusion timeline. For a batch of size $B$, divide it into $B$ equal strata:
# $$
# \left[\frac{b}{B},\frac{b+1}{B}\right), \qquad b=0,\ldots,B-1.
# $$
# For every stratum, sample an independent random value
# $$
# V_b\sim\mathcal U[0,1).
# $$
# $V_b$ describes the relative position inside the stratum $b$
# Now we can assign a timestep to each stratum by scaling the relative position $V_b$ to the complete timeline $[0, 1)$:
# $$
# U_b = \frac{b + V_b}{B}
# $$
# Finally, we convert the continuous value $U_b$ into the discrete diffusion timestep
# $$
# J_b = \lfloor TU_b\rfloor + 1
# $$


# %%
def sample_steps(
    batch: Int[B], T: int, device="cpu", generator=None, stratified=True
) -> Tensor[[B]]:
    assert batch > 0 and T > 0, "batch and T must be positive integers"

    if not stratified:
        return torch.randint(1, T + 1, (batch,), device=device, generator=generator)

    uniform_samples = (
        torch.arange(batch, device=device, dtype=torch.float64)
        + torch.rand(batch, device=device, generator=generator)
    ) / batch
    uniform_samples = uniform_samples[
        torch.randperm(batch, device=device, generator=generator)
    ]

    return (uniform_samples * T).long().clamp(max=T - 1) + 1


# %%
def check_sample_steps():
    timestep_indices = sample_steps(
        1000, 1000, generator=torch.Generator().manual_seed(9)
    )
    assert timestep_indices.dtype == torch.long and torch.equal(
        timestep_indices.sort().values, torch.arange(1, 1001)
    )
    assert sample_steps(5, 1).eq(1).all()
    first_sample = sample_steps(
        10, 8, generator=torch.Generator().manual_seed(3), stratified=False
    )
    second_sample = sample_steps(
        10, 8, generator=torch.Generator().manual_seed(3), stratified=False
    )
    assert torch.equal(first_sample, second_sample)


check_sample_steps()

# %% [markdown]
# ## Simple and Weighted loss (Paper Eq. 10, 11)
# Since the candidate vector $x^c_0$ is one-hot, if the true token is $k^*$, then
# $$
# x_{0,k}^c = \begin{cases} 1,&k=k^\star,\\ 0,&k\neq k^\star. \end{cases}
# $$
# Let the model output probabilities be: $\mathbf p_\theta = \operatorname{NN}_\theta(x_t^c,t) = (p_{\theta,1},\ldots,p_{\theta,K})$
# then,
# $$
# \begin{aligned} \left\langle \log\mathbf p_\theta,x_0^c \right\rangle &= \sum_{k=1}^{K} x_{0,k}^c\log p_{\theta,k}\\ &= \log p_{\theta,k^\star}. \end{aligned}
# $$. Applying the negative sign, we have the negative log-likelihood of the true token:
# $$
# \operatorname{CE}(x_0^c,\mathbf p_\theta) = -\log p_{\theta,k^\star}.
# $$

# ### Weighted loss
# Normalize time $n(t) = n(\frac{j}{T})$ and $n(t-1) = n(\frac{j-1}{T})$. The weight
# $$
# \begin{aligned}
# \frac{n(t) - n(t-1)}{n(t)-1} &= \frac{n(t) - 1 - n(t-1) + 1}{n(t) - 1} \\
#         &= \frac{n(t) - 1}{n(t) - 1} - \frac{n(t-1) - 1}{n(t) - 1} \\
#         &= 1 - \frac{n(t-1) - 1}{n(t) - 1} \\
#         &= 1 - r(t-1, t)
# \end{aligned}
# $$
# can be calculated using the survival ratio $r(t-1, t)$.

# > Remember that $n(t)=\mathbb E[|C_t|\mid x_0]$ is the expected total number of candidates at time $t$. Because the true token is always included, $n(t)-1$ is the expected number of wrong candidates at time $t$.
# >
# > Now,
# $$
# 1-r(s,t) = \frac{\text{expected number of wrong candidates removed}} {\text{expected number of wrong candidates currently present}}
# $$
# >
# > $1-r(s,t)$ is the fraction of the currently present wrong
# > candidates that the schedule expects to remove during the reverse
# > transition from time $t$ to time $s$.
#
# TODO check this
# The weighted loss gives more weight to clean-token prediction errors at
# reverse steps that are scheduled to remove a larger fraction of the
# currently present wrong candidates. The weight is determined by the
# schedule, not by how many candidates the model actually removes.


# %%
def training_loss(
    log_probabilities: Tensor[[B, L, K]],
    clean_tokens: Tensor[[B, L]],
    timestep_indices: Tensor[[B]],
    T: int,
    objective: str,
) -> Tensor[[B, L]]:
    cross_entropy = -log_probabilities.gather(-1, clean_tokens[..., None]).squeeze(-1)

    if objective == "simple":
        return T * cross_entropy
    elif objective == "weighted":
        time = timestep_indices.double() / T
        earlier_time = (timestep_indices.double() - 1) / T
        step_weights = 1 - survival_ratio(
            earlier_time, time, log_probabilities.shape[-1]
        ).to(cross_entropy.dtype)

        return T * step_weights[:, None] * cross_entropy
    else:
        raise ValueError(f"objective must be 'simple' or 'weighted', got {objective}")


# %%
def check_training_loss():
    log_probabilities = torch.full(
        (2, 1, 4), -math.log(4.0), dtype=torch.float64, requires_grad=True
    )
    inputs = torch.zeros(2, 1, dtype=torch.long)
    timestep_indices = torch.tensor([1, 2])
    simple = training_loss(log_probabilities, inputs, timestep_indices, 2, "simple")
    weighted = training_loss(log_probabilities, inputs, timestep_indices, 2, "weighted")
    torch.testing.assert_close(
        simple, torch.full((2, 1), 2 * math.log(4.0), dtype=torch.float64)
    )
    torch.testing.assert_close(
        weighted[:, 0],
        torch.tensor([2 * math.log(4.0), 4 / 3 * math.log(4.0)], dtype=torch.float64),
    )
    weighted.mean().backward()
    assert torch.isfinite(log_probabilities.grad).all()


check_training_loss()

# %% [markdown]
# # Backbone


# %%
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

# %%
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

def check_backbone():
    TOY = Config()
    model = BFN_DIT(TOY)
    centroids = torch.full((2, TOY.length, TOY.vocab), 1 / TOY.vocab)
    output = model(centroids, torch.tensor([0.2, 0.8]))
    assert_shape(output, 2, TOY.length, TOY.vocab)
    assert output.eq(0).all()
    F.cross_entropy(
        output.reshape(-1, TOY.vocab), torch.zeros(2 * TOY.length, dtype=torch.long)
    ).backward()
    assert model.output_layer.linear.weight.grad.abs().sum() > 0
    assert all(
        torch.isfinite(predicted_probs.grad).all()
        for predicted_probs in model.parameters()
        if predicted_probs.grad is not None
    )


check_backbone()

# %% [markdown]
# ## Predicting log probabilities
# The model gets a binary candidate mask, which is converted to a centroid representation.
# The output is masked such that tokens not included in $c_t$ can be reintroduced.


# %%
def predict_log_probs(
    model, candidate_mask: Tensor[[B, L, K]], time: Tensor[[B]]
) -> Tensor[[B, L, K]]:
    logits = model(centroid(candidate_mask), time)
    if logits.dtype != torch.float64:
        logits = logits.double()

    return masked_log_probs(logits, candidate_mask)


# %%
def check_predict_log_probs():
    TOY = Config()
    model = BFN_DIT(TOY)
    candidate_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    candidate_mask[:, :, 3] = False
    log_probabilities = predict_log_probs(model, candidate_mask, torch.ones(2))
    assert_simplex(log_probabilities.exp())
    assert log_probabilities.exp()[:, :, 3].eq(0).all()
    torch.testing.assert_close(
        log_probabilities.exp()[:, :, :3], torch.full((2, 3, 3), 1 / 3)
    )


check_predict_log_probs()

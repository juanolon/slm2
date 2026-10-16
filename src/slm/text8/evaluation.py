from __future__ import annotations

import math

import numpy as np
import torch

from slm.common.seeds import VALIDATION_SEED
from slm.slm.core import (
    elbo_term,
    predict_log_probs,
    prune_step,
    sample_forward,
    sample_steps,
)

from slm.slm.sampling import protected_prune_step, confidence_gated_prune_step, topk_prune_step

from .data import gather_windows


def estimate_bpc(model, clean_tokens, config):
    old = model.training
    model.eval()
    try:
        with torch.no_grad():
            steps = sample_steps(
                len(clean_tokens), config.T, device=clean_tokens.device
            )
            candidates = sample_forward(
                clean_tokens, steps.double() / config.T, config.vocab
            )
            log_probs = predict_log_probs(model, candidates, steps.float() / config.T)
            terms = elbo_term(log_probs, clean_tokens, candidates, steps, config.T)
            return config.T * terms.mean(-1) / math.log(2)
    finally:
        model.train(old)


def evaluate_stream(model, ids, config, windows, draws, seed=VALIDATION_SEED):
    assert windows > 0 and draws > 0 and len(ids) >= config.length
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(
        low=0, high=len(ids) - config.length + 1, size=(windows,), generator=generator
    )
    scores = []
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )

    with torch.random.fork_rng(devices=devices):
        for draw in range(draws):
            torch.manual_seed(seed + 100 + draw)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 100 + draw)
            total = 0.0
            seen = 0
            for begin in range(0, windows, config.microbatch):
                inputs = gather_windows(
                    ids, starts[begin : begin + config.microbatch], config.length
                ).to(device)
                bpc = estimate_bpc(model, inputs, config)
                assert torch.isfinite(bpc).all(), (
                    "inspect exact probabilities; do not hide infinities"
                )
                total += bpc.sum().item()
                seen += len(inputs)
            scores.append(total / seen)
    score_array = np.asarray(scores, dtype=np.float64)
    return {
        "bpc": float(score_array.mean()),
        "mc_se": float(score_array.std(ddof=1) / math.sqrt(draws))
        if draws > 1
        else None,
        "draw_scores": scores,
        "windows": windows,
        "draws": draws,
        "window_seed": seed,
        "T": config.T,
    }


class Text8Evaluator:
    def __init__(self, ids, config):
        self.ids, self.config = ids, config

    def __call__(self, model):
        result = evaluate_stream(
            model,
            self.ids,
            self.config,
            self.config.validation_windows,
            self.config.validation_draws,
            seed=VALIDATION_SEED,
        )
        return {"validation/bpc": result["bpc"]}


def generate(
    model, batch, length, vocabulary_size, steps, device="cpu", return_history=False,
    *, use_protected_prune_step=False, use_confidence_gated_prune_step=False,
    use_topk_prune_step=False,
):
    assert steps >= 1
    if sum((use_protected_prune_step, use_confidence_gated_prune_step, use_topk_prune_step)) > 1:
        raise ValueError("Pruning modes are mutually exclusive")
    pruning = (topk_prune_step if use_topk_prune_step else
               confidence_gated_prune_step if use_confidence_gated_prune_step else
               protected_prune_step if use_protected_prune_step else prune_step)
    old = model.training
    model.eval()
    try:
        with torch.no_grad():
            candidates = torch.ones(
                batch, length, vocabulary_size, dtype=torch.bool, device=device
            )
            history = [candidates.sum(-1).cpu()]
            for index in range(steps, 0, -1):
                time = torch.full(
                    (batch,), index / steps, dtype=torch.float64, device=device
                )
                earlier = torch.full(
                    (batch,), (index - 1) / steps, dtype=torch.float64, device=device
                )
                candidates = pruning(model, candidates, earlier, time)
                if return_history:
                    history.append(candidates.sum(-1).cpu())
            tokens = predict_log_probs(
                model, candidates, torch.zeros(batch, device=device)
            ).argmax(-1)
            return (tokens, history) if return_history else tokens
    finally:
        model.train(old)

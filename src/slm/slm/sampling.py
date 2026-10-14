from __future__ import annotations

import torch

from .core import predict_log_probs, reverse_probs


def protected_prune_step(
    model, candidate_mask, earlier_time, time, generator=None, *, record=False
):
    """Prune normally, then retain one model-sampled protected digit per cell.

    ``record=True`` additionally returns diagnostics for a single-board heatmap.
    """
    if record and len(candidate_mask) != 1:
        raise ValueError("Recorded pruning requires a single board")
    with torch.no_grad():
        clean_probabilities = (
            predict_log_probs(model, candidate_mask, time.float()).double().exp()
        )
        clean_probabilities = clean_probabilities.masked_fill(~candidate_mask, 0.0)
        protected_digit = torch.multinomial(
            clean_probabilities.reshape(-1, clean_probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(*clean_probabilities.shape[:-1], 1)
        probs = reverse_probs(
            clean_probabilities, candidate_mask, earlier_time.double(), time.double()
        )
        uniform_draws = torch.rand(
            probs.shape, dtype=torch.float64, device=probs.device, generator=generator
        )
        sampled = uniform_draws < probs
        survivors = sampled & candidate_mask
        survivors.scatter_(-1, protected_digit, True)
        if not record:
            return survivors
        return survivors, {
            "before": candidate_mask[0].cpu().tolist(),
            "model_probability": clean_probabilities[0].cpu().tolist(),
            "retention_probability": probs[0].cpu().tolist(),
            "draws": uniform_draws[0].cpu().tolist(),
            "sampled": sampled[0].cpu().tolist(),
            "repaired": survivors[0].cpu().tolist(),
        }

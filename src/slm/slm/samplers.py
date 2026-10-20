from __future__ import annotations

SAMPLERS = {
    "default": "default_prune_step",
    "protected": "protected_prune_step",
    "confidence-gated": "confidence_gated_prune_step",
    "topk": "topk_prune_step",
    "budgeted": "BudgetedPruner",
}


def create_sampler(name, candidate_mask, generator=None):

    try:
        function = SAMPLERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown sampler: {name}. Choose from: {', '.join(SAMPLERS)}"
        ) from None
    # Keep CLI argument parsing independent of torch.
    from . import sampling

    sampler = getattr(sampling, function)
    return sampler(candidate_mask, generator) if isinstance(sampler, type) else sampler

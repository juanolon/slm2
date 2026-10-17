from __future__ import annotations

SAMPLERS = {
    "default": "default_prune_step",
    "protected": "protected_prune_step",
    "confidence-gated": "confidence_gated_prune_step",
    "topk": "topk_prune_step",
}


def get_sampler(name="default"):
    try:
        function = SAMPLERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown sampler: {name}. Choose from: {', '.join(SAMPLERS)}"
        ) from None
    # Keep CLI argument parsing independent of torch.
    from . import sampling

    return getattr(sampling, function)

"""Shared report-only seed selection and scalar score averaging."""
from __future__ import annotations

from secrets import SystemRandom


def generate_seeds(count):
    # Independent of the process's seeded RNGs; unique and persisted in reports.
    return SystemRandom().sample(range(2**32), count)


def sampling_settings(args, default_seed):
    count = getattr(args, "sampling_seeds", None)
    seed = getattr(args, "sampling_seed", None)
    if count is not None:
        if seed is not None:
            raise ValueError("sampling-seed and sampling-seeds are mutually exclusive")
        if type(count) is not int or count < 1:
            raise ValueError("sampling-seeds must be a positive integer")
        seeds = generate_seeds(count)
    else:
        seeds = [default_seed if seed is None else seed]
    return {
        "sampling_seed": seeds[0] if len(seeds) == 1 else None,
        "sampling_seeds": len(seeds),
        "sampling_seed_list": seeds,
        "protected_prune_step": getattr(args, "protected_prune_step", False),
    }


def phase_seeds(args, sampling, default_seed):
    if (getattr(args, "sampling_seed", None) is not None
            or getattr(args, "sampling_seeds", None) is not None):
        return sampling["sampling_seed_list"]
    return [default_seed]


def mean_scores(scores):
    return {
        key: sum(score[key] for score in scores) / len(scores)
        for key in scores[0]
    }

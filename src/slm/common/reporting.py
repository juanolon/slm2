from __future__ import annotations

from secrets import SystemRandom
from statistics import stdev


def generate_seeds(count):
    # Independent of the process's seeded RNGs; unique and persisted in reports.
    return SystemRandom().sample(range(2**32), count)


def sampling_settings(args, default_seed):
    if hasattr(args, "sampling_seed_list"):
        return {
            key: getattr(args, key)
            for key in (
                "sampling_seed",
                "sampling_seeds",
                "sampling_seed_list",
                "sampler",
            )
        }
    sampler = getattr(args, "sampler", None) or "default"
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
        "sampler": sampler,
    }


def phase_seeds(args, sampling, default_seed):
    if getattr(args, "sampling_explicit", None) is False:
        return [default_seed]
    if (
        getattr(args, "sampling_seed", None) is not None
        or getattr(args, "sampling_seeds", None) is not None
    ):
        return sampling["sampling_seed_list"]
    return [default_seed]


def mean_scores(scores):
    return {key: sum(score[key] for score in scores) / len(scores) for key in scores[0]}


def score_statistics(scores):
    result = {}
    for key in scores[0]:
        values = [score[key] for score in scores]
        result[key] = {
            "mean": sum(values) / len(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def log_seed_distribution(tracker, name, seeds, scores):
    """Each point is one complete evaluation, not one board or an MC draw."""
    if len(seeds) < 2:
        return
    import plotly.graph_objects as go
    import wandb

    metrics = list(scores[0])
    for metric in metrics:
        figure = go.Figure()
        figure.add_trace(
            go.Box(
                name=metric,
                y=[score[metric] for score in scores],
                text=[f"seed {seed}" for seed in seeds],
                boxpoints="all",
                boxmean=True,
                jitter=0.25,
                pointpos=0,
                hoverinfo="y+text+name",
            )
        )
        figure.update_layout(
            title=f"{name}: {metric} across {len(seeds)} seeds"
            "<br><sup>Box: middle 50% · solid: median · dashed: mean · dots: seeds</sup>",
            yaxis_title="per-seed evaluation score",
            showlegend=False,
        )
        tracker.artifact(f"{name}/{metric}_seed_distribution", wandb.Plotly(figure))
    tracker.artifact(
        f"{name}/by_seed",
        wandb.Table(
            columns=["seed", *metrics],
            data=[
                [seed, *[score[metric] for metric in metrics]]
                for seed, score in zip(seeds, scores)
            ],
        ),
    )
    tracker.run.summary[f"{name}/seed_count"] = len(seeds)
    for metric, statistics in score_statistics(scores).items():
        for statistic, value in statistics.items():
            tracker.run.summary[f"{name}/{metric}_{statistic}"] = value

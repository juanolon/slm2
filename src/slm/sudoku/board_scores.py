"""Local per-board CSVs and bounded W&B table previews, indexed like heatmaps."""

from __future__ import annotations

import csv
from pathlib import Path
from shutil import copyfile
import warnings

import torch


def board_score_files(suffix=""):
    return {
        "summary": f"sudoku_board_scores{suffix}.csv",
        "by_seed": f"sudoku_board_scores_by_seed{suffix}.csv",
    }


def write_board_scores(
    report_path, run_path, split, seeds, runs, steps, tracker, suffix=""
):
    import wandb

    priority = ("blank_accuracy", "cell_accuracy", "unit_validity")
    metrics = [metric for metric in priority if metric in runs[0]] + [
        metric for metric in runs[0] if metric not in priority
    ]
    values = torch.stack(
        [torch.stack([run[metric] for metric in metrics], -1) for run in runs]
    ).double()
    summary = torch.stack(
        (
            values.mean(0),
            values.amin(0),
            values.amax(0),
            values.std(0) if len(seeds) > 1 else torch.zeros_like(values[0]),
        ),
        1,
    )
    ratings = split["ratings"].tolist()
    blanks = (split["questions"] < 0).sum(-1).tolist()
    filenames = board_score_files(suffix)
    columns = {
        "summary": [
            "board_id",
            "sampling_steps",
            "rating",
            "blank_cells",
            "sampling_seeds",
        ]
        + [
            f"{metric}_{statistic}"
            for statistic in ("mean", "min", "max", "std")
            for metric in metrics
        ],
        "by_seed": [
            "board_id",
            "seed",
            "sampling_steps",
            "rating",
            "blank_cells",
            *metrics,
        ],
    }
    # Keep full CSVs even when a W&B preview would exceed its regular-table limit.
    limit = wandb.Table.MAX_ROWS
    previews = {"summary": [], "by_seed": []}
    with (Path(report_path) / filenames["summary"]).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns["summary"])
        for board_id, row in enumerate(summary.numpy().reshape(len(ratings), -1)):
            record = [
                board_id,
                steps,
                ratings[board_id],
                blanks[board_id],
                len(seeds),
                *row.tolist(),
            ]
            writer.writerow(record)
            if len(previews["summary"]) < limit:
                previews["summary"].append(record)
    with (Path(report_path) / filenames["by_seed"]).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns["by_seed"])
        for seed_index, (seed, scores) in enumerate(zip(seeds, values.numpy())):
            seed_limit = limit // len(seeds) + (seed_index < limit % len(seeds))
            for board_id, row in enumerate(scores):
                record = [
                    board_id,
                    seed,
                    steps,
                    ratings[board_id],
                    blanks[board_id],
                    *row.tolist(),
                ]
                writer.writerow(record)
                if board_id < seed_limit:
                    previews["by_seed"].append(record)
    for kind, filename in filenames.items():
        if Path(report_path) != Path(run_path):
            copyfile(Path(report_path) / filename, Path(run_path) / filename)
        tracker.artifact(
            f"boards/sudoku{suffix}/{kind}",
            wandb.Table(columns=columns[kind], data=previews[kind]),
        )
        total = len(ratings) * (len(seeds) if kind == "by_seed" else 1)
        if total > limit:
            warnings.warn(
                f"W&B {kind} board table shows {limit} of {total} rows; the complete list is in {Path(report_path) / filename}"
            )

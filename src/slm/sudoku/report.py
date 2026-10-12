from __future__ import annotations

import json

import torch

from slm.common.seeds import TEST_SEED

from .data import decode_board
from .evaluation import solve_sudoku, sudoku_scores

REPORT_STEPS = (32, 64, 128, 256, 512, 1000)


def evaluate_report(model, split, config, device, budgets=REPORT_STEPS):
    count = len(split["questions"])
    if count < 1:
        raise ValueError("No evaluation boards")
    ratings = sorted(split["ratings"].unique().tolist())
    result = {}
    for steps in budgets:
        generator = torch.Generator(device=device).manual_seed(TEST_SEED)
        groups = {
            "all": {"boards": count},
            **{
                str(rating): {"boards": int((split["ratings"] == rating).sum())}
                for rating in ratings
            },
        }
        totals = {key: {} for key in groups}
        # Each column stores surviving cells, intact boards, newly invalidated
        # boards, and newly invalidated cells. Only aggregate histories persist.
        histories = {
            key: torch.zeros(steps + 1, 4, dtype=torch.float64) for key in groups
        }
        blank_counts = dict.fromkeys(groups, 0)
        samples = []
        for start in range(0, count, config.microbatch):
            stop = min(start + config.microbatch, count)
            questions = split["questions"][start:stop].to(device)
            answers = split["answers"][start:stop].to(device)
            batch_ratings = split["ratings"][start:stop]
            blanks = questions < 0
            survivors = torch.empty(
                steps + 1, len(questions), device=device, dtype=torch.long
            )
            for_boards = torch.empty_like(survivors, dtype=torch.bool)

            def observe(index, candidates):
                survives = (
                    candidates.gather(-1, answers[..., None]).squeeze(-1) | ~blanks
                )
                survivors[index] = (survives & blanks).sum(-1)
                for_boards[index] = survives.all(-1)

            predictions = solve_sudoku(
                model, questions, steps, generator, observer=observe
            )
            scores = {
                key: values.cpu()
                for key, values in sudoku_scores(
                    predictions, questions, answers
                ).items()
            }
            survivors = survivors.cpu()
            intact = for_boards.cpu()
            new_boards = torch.zeros_like(intact)
            new_boards[1:] = intact[:-1] & ~intact[1:]
            new_cells = torch.zeros_like(survivors)
            new_cells[1:] = survivors[:-1] - survivors[1:]
            for key in groups:
                selected = (
                    torch.ones(len(questions), dtype=torch.bool)
                    if key == "all"
                    else batch_ratings == int(key)
                )
                if not selected.any():
                    continue
                blank_counts[key] += int(blanks.sum(-1).cpu()[selected].sum())
                for name, values in scores.items():
                    totals[key][name] = (
                        totals[key].get(name, 0.0)
                        + values[selected].double().sum().item()
                    )
                histories[key] += torch.stack(
                    (
                        survivors[:, selected].sum(-1),
                        intact[:, selected].sum(-1),
                        new_boards[:, selected].sum(-1),
                        new_cells[:, selected].sum(-1),
                    ),
                    -1,
                )
            for index in range(min(4 - len(samples), len(questions))):
                samples.append(
                    [
                        decode_board(questions[index].cpu()),
                        decode_board(predictions[index].cpu()),
                        decode_board(answers[index].cpu()),
                        int(batch_ratings[index]),
                    ]
                )
        for key, group in groups.items():
            n = group["boards"]
            h = histories[key]
            group["scores"] = {name: value / n for name, value in totals[key].items()}
            group["trajectory"] = {
                "step": list(range(steps + 1)),
                "invalidations": (h[:, 2] / n).tolist(),
                "cumulative_invalidations": (1 - h[:, 1] / n).tolist(),
                "invalidated_cells_per_board": (h[:, 3] / n).tolist(),
                "cell_survival_fraction": (h[:, 0] / blank_counts[key]).tolist()
                if blank_counts[key]
                else [1.0] * (steps + 1),
            }
        result[str(steps)] = {"groups": groups, "samples": samples}
        print(
            json.dumps(
                {
                    "sampling_steps": steps,
                    "boards": count,
                    "scores_by_rating": {
                        key: group["scores"] for key, group in groups.items()
                    },
                }
            ),
            flush=True,
        )
    return result


def log_report(tracker, completed, results):
    import wandb

    ratings = list(next(iter(results.values()))["groups"])
    budgets = [int(key) for key in results]
    for steps, result in results.items():
        metrics = {}
        for rating, group in result["groups"].items():
            prefix = "all" if rating == "all" else f"rating_{rating}"
            metrics[f"report/steps_{steps}/{prefix}/boards"] = group["boards"]
            for key, value in group["scores"].items():
                name = f"report/steps_{steps}/{prefix}/{key}"
                metrics[name] = value
                tracker.run.summary[name] = value
        tracker.log(completed, metrics)
    for metric in ("solve_rate", "cell_accuracy", "board_validity"):
        tracker.artifact(
            f"steps-{metric}",
            wandb.plot.line_series(
                xs=budgets,
                ys=[
                    [
                        result["groups"][rating]["scores"][metric]
                        for result in results.values()
                    ]
                    for rating in ratings
                ],
                keys=[
                    "all" if rating == "all" else f"rating {rating}"
                    for rating in ratings
                ],
                title=f"Sampling steps vs {metric}",
                xname="sampling steps",
            ),
        )
    for steps, result in results.items():
        for metric in ("invalidations", "cell_survival_fraction"):
            tracker.artifact(
                f"trajectory/{steps}-{metric}",
                wandb.plot.line_series(
                    xs=result["groups"]["all"]["trajectory"]["step"],
                    ys=[
                        result["groups"][rating]["trajectory"][metric]
                        for rating in ratings
                    ],
                    keys=[
                        "all" if rating == "all" else f"rating {rating}"
                        for rating in ratings
                    ],
                    title=f"{steps} steps: {metric}",
                    xname="reverse sampling step",
                ),
            )
        tracker.artifact(
            f"samples/sudoku-{steps}",
            wandb.Table(
                columns=["question", "prediction", "reference", "rating"],
                data=result["samples"],
            ),
        )

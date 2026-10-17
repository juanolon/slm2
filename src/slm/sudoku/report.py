from __future__ import annotations

import json
import warnings
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import torch

from slm.common.checkpoint import CheckpointManager, load_checkpoint
from slm.common.contracts import Report
from slm.common.reporting import mean_scores, phase_seeds, sampling_settings
from slm.common.seeds import SAMPLE_SEED, TEST_SEED
from slm.common.tracking import ExperimentTracker
from slm.slm.backbone import BFN_DIT

from .data import data_fingerprint, decode_board, load_sudoku
from .evaluation import evaluate_sudoku, solve_sudoku, sudoku_scores

REPORT_STEPS = (32, 64, 128, 256, 512, 1000)


class DefaultReport(Report):
    """Basic single-budget scores and samples, without trajectory diagnostics."""

    def run(self, args: Namespace) -> dict[str, Any]:
        sampling = sampling_settings(args, TEST_SEED)
        steps = getattr(args, "sampling_steps", None)
        if steps is None:
            steps = args.config.sudoku.get(
                "report_steps", args.config.sudoku["sample_steps"]
            )
        if type(steps) is not int or steps < 1:
            raise ValueError("sudoku.report_steps must be a positive integer")
        config, model, test, completed, checkpoint_fingerprint, tracking = _load_report(
            args
        )
        evaluation_config = SimpleNamespace(
            **{**vars(config), "sample_steps": steps, "sampling_steps": steps, **sampling}
        )
        device = torch.device(args.device)
        by_seed = [
            {
                "seed": seed,
                "scores": evaluate_sudoku(
                    model, test, evaluation_config, device, seed=seed
                ),
            }
            for seed in sampling["sampling_seed_list"]
        ]
        scores = mean_scores([run["scores"] for run in by_seed])
        sample_seeds = phase_seeds(args, sampling, SAMPLE_SEED)
        report = {
            "scores": scores,
            "boards": len(test["questions"]),
            "seed": sampling["sampling_seed"],
            **sampling,
            "sample_seed_list": sample_seeds,
            "sample_steps": steps,
            "checkpoint_step": completed,
            "max_rating": config.max_rating,
            "dataset_fingerprint": data_fingerprint({"test": test}),
            "checkpoint_dataset_fingerprint": checkpoint_fingerprint,
        }
        if len(by_seed) > 1:
            report["by_seed"] = by_seed
        report_path, tracker = _start_report(
            args, vars(evaluation_config), report, tracking
        )
        try:
            tracker.log(
                completed,
                {f"report/test_{key}": value for key, value in scores.items()},
            )
            for key, value in scores.items():
                tracker.run.summary[f"test/{key}"] = value
            count = min(4, config.microbatch, len(test["questions"]))
            questions = test["questions"][:count].to(device)
            samples = []
            for seed in sample_seeds:
                predictions = solve_sudoku(
                    model,
                    questions,
                    steps,
                    torch.Generator(device=device).manual_seed(seed),
                    sampler=sampling["sampler"],
                )
                for question, prediction, answer in zip(
                    questions, predictions, test["answers"][:count]
                ):
                    row = [
                        decode_board(question.cpu()),
                        decode_board(prediction.cpu()),
                        decode_board(answer),
                    ]
                    samples.append([seed, *row] if len(sample_seeds) > 1 else row)
            (Path(args.run_dir) / "sudoku_samples.json").write_text(
                json.dumps(samples, indent=2)
            )
            (report_path / "sudoku_samples.json").write_text(
                json.dumps(samples, indent=2)
            )
            import wandb

            tracker.artifact(
                "samples/sudoku",
                wandb.Table(
                    columns=(["seed"] if len(sample_seeds) > 1 else [])
                    + ["question", "prediction", "reference"],
                    data=samples,
                ),
            )
        except BaseException:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish()
        print(json.dumps(report, indent=2))
        return report


class SweepReport(Report):
    def run(self, args: Namespace) -> dict[str, Any]:
        sampling = sampling_settings(args, TEST_SEED)
        config, model, test, completed, checkpoint_fingerprint, tracking = _load_report(
            args
        )
        run_path = Path(args.run_dir)
        results = evaluate_report(
            model,
            test,
            config,
            torch.device(args.device),
            budgets=REPORT_STEPS,
            seeds=sampling["sampling_seed_list"],
            sampler=sampling["sampler"],
        )
        report = {
            "by_steps": results,
            "boards": len(test["questions"]),
            "seed": sampling["sampling_seed"],
            **sampling,
            "sampling_budgets": list(REPORT_STEPS),
            "checkpoint_step": completed,
            "max_rating": config.max_rating,
            "dataset_fingerprint": data_fingerprint({"test": test}),
            "checkpoint_dataset_fingerprint": checkpoint_fingerprint,
        }
        report_path, tracker = _start_report(
            args,
            {**vars(config), "report_steps": list(REPORT_STEPS), **sampling},
            report,
            tracking,
        )
        try:
            log_report(tracker, completed, results)
            samples = {steps: result["samples"] for steps, result in results.items()}
            (run_path / "sudoku_samples.json").write_text(json.dumps(samples, indent=2))
            (report_path / "sudoku_samples.json").write_text(
                json.dumps(samples, indent=2)
            )
            import wandb

            artifact = wandb.Artifact(
                f"sudoku-report-{report_path.name}", type="report"
            )
            artifact.add_file(str(report_path / "sudoku_report.json"))
            tracker.run.log_artifact(artifact)
        except BaseException:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish()
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "by_steps"},
                indent=2,
            )
        )
        return report


def _load_report(args: Namespace, weights="ema"):
    from .run import model_config

    config = model_config(args.config)
    run_path = Path(args.run_dir)
    device = torch.device(args.device)
    manifest_path = run_path / "manifest.json"
    if getattr(args, "legacy", False):
        state = torch.load(run_path / "best.pt", map_location="cpu", weights_only=False)
        for key in ("hidden", "heads", "blocks", "cond", "dropout", "vocab", "length"):
            if key in state["config"] and state["config"][key] != getattr(config, key):
                raise ValueError(
                    f"Legacy checkpoint model configuration differs: {key}"
                )
        model_weights = state["ema"] if weights == "ema" else state["model"]
        completed = state["next_step"]
        checkpoint_fingerprint = state["fingerprint"]
        data, metadata = load_sudoku(config, args.data_dir)
        if checkpoint_fingerprint != metadata["fingerprint"]:
            warnings.warn(
                "Legacy checkpoint dataset differs from the current selection; reporting on the current dataset"
            )
        tracking = (
            json.loads(manifest_path.read_text())["wandb"]
            if manifest_path.exists()
            else {"mode": "online", "project": "sudoku", "entity": None}
        )
    else:
        state = load_checkpoint(run_path / "best.pt")
        model_weights = state["ema"]["model"] if weights == "ema" else state["model"]
        completed = state["progress"]["completed_updates"]
        checkpoint_fingerprint = state["data_fingerprint"]
        validation_config = SimpleNamespace(
            **{**vars(config), "max_test_split": state["config"]["max_test_split"]}
        )
        data, metadata = load_sudoku(validation_config, args.data_dir)
        CheckpointManager(
            run_path, vars(validation_config), metadata["fingerprint"]
        ).validate(state)
        if config.max_test_split != validation_config.max_test_split:
            del data
            data, metadata = load_sudoku(config, args.data_dir)
        tracking = json.loads(manifest_path.read_text())["wandb"]

    model = BFN_DIT(config).to(device).eval()
    model.load_state_dict(model_weights)
    return config, model, data["test"], completed, checkpoint_fingerprint, tracking


def _start_report(args, config, report, tracking):
    run_path = Path(args.run_dir)
    (run_path / "sudoku_report.json").write_text(json.dumps(report, indent=2))
    report_path = run_path / "report" / uuid4().hex
    report_path.mkdir(parents=True)
    (report_path / "sudoku_report.json").write_text(json.dumps(report, indent=2))
    tracker = ExperimentTracker(
        report_path,
        vars(args.config),
        name=args.name or tracking.get("name") or config["objective"],
        mode=args.wandb_mode or tracking["mode"],
        project=getattr(args.config, "project", None) or f"{args.project or tracking['project']}-report",
        entity=args.entity if args.entity is not None else tracking.get("entity"),
        resume=False,
    )
    return report_path, tracker


def evaluate_report(
    model,
    split,
    config,
    device,
    budgets=REPORT_STEPS,
    *,
    seeds=None,
    sampler="default",
):
    seeds = [TEST_SEED] if seeds is None else list(seeds)
    if not seeds:
        raise ValueError("At least one sampling seed is required")
    runs = [
        _evaluate_report(
            model,
            split,
            config,
            device,
            budgets,
            seed,
            sampler=sampler,
        )
        for seed in seeds
    ]
    sampling_seeds = len(seeds)
    if sampling_seeds == 1:
        return runs[0]
    results = {}
    for budget, first in runs[0].items():
        groups = {}
        for rating, group in first["groups"].items():
            peers = [run[budget]["groups"][rating] for run in runs]
            groups[rating] = {
                "boards": group["boards"],
                "scores": mean_scores([peer["scores"] for peer in peers]),
                "trajectory": {
                    key: values
                    if key == "step"
                    else [
                        sum(peer["trajectory"][key][i] for peer in peers)
                        / sampling_seeds
                        for i in range(len(values))
                    ]
                    for key, values in group["trajectory"].items()
                },
            }
        results[budget] = {
            "groups": groups,
            "samples": [
                [seeds[index], *sample]
                for index, run in enumerate(runs)
                for sample in run[budget]["samples"]
            ],
        }
    return results


def _evaluate_report(
    model,
    split,
    config,
    device,
    budgets,
    seed,
    *,
    sampler="default",
):
    count = len(split["questions"])
    if count < 1:
        raise ValueError("No evaluation boards")
    ratings = sorted(split["ratings"].unique().tolist())
    result = {}
    for steps in budgets:
        generator = torch.Generator(device=device).manual_seed(seed)
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
                model,
                questions,
                steps,
                generator,
                observer=observe,
                sampler=sampler,
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
                    "sampling_seed": seed,
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
                columns=(
                    ["seed"]
                    if result["samples"] and len(result["samples"][0]) == 5
                    else []
                )
                + ["question", "prediction", "reference", "rating"],
                data=result["samples"],
            ),
        )

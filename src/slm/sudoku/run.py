from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import torch

from slm.common.checkpoint import CheckpointManager, load_checkpoint
from slm.common.config import PARAMETERS
from slm.common.contracts import TaskRunner
from slm.common.tracking import ExperimentTracker, NullTracker
from slm.common.trainer import LifecycleOptions, Trainer, UpdateOptions
from slm.slm.backbone import BFN_DIT

from .data import (
    RandomBoardSource,
    data_fingerprint,
    decode_board,
    load_sudoku,
    pack_rows,
)
from .evaluation import SudokuEvaluator, evaluate_sudoku, solve_sudoku
from .task import SudokuTask

SUDOKU_TRAINING_FIELDS = (
    "max_rating",
    "max_train_split",
    "max_validation_split",
    "max_test_split",
    "validation_boards",
    "sample_steps",
)


def model_config(config):
    return SimpleNamespace(
        **{key: vars(config)[key] for key in PARAMETERS},
        **{key: config.sudoku[key] for key in SUDOKU_TRAINING_FIELDS},
        vocab=9,
        length=81,
    )


class SudokuSelection:
    def __init__(self):
        self.best_score = (-1.0, -1.0)

    def consider(self, metrics):
        score = (
            metrics["validation/exact_match"],
            metrics["validation/blank_accuracy"],
        )
        improved = score > self.best_score
        if improved:
            self.best_score = score
        return improved

    def state_dict(self):
        return {"best_score": self.best_score}

    def load_state_dict(self, state):
        self.best_score = tuple(state["best_score"])


def _smoke_data():
    answer = (
        "123456789456789123789123456"
        "234567891567891234891234567"
        "345678912678912345912345678"
    )
    board = pack_rows([("." + answer[1:], answer, 0)])
    return {"train": board, "validation": board}


class Runner(TaskRunner):
    def validate_config(self, config: SimpleNamespace) -> None:
        settings = config.sudoku
        missing = (set(SUDOKU_TRAINING_FIELDS) | {"report_steps"}) - settings.keys()
        if missing:
            raise ValueError(f"Missing sudoku settings: {sorted(missing)}")
        for key in (
            "max_train_split",
            "max_validation_split",
            "max_test_split",
            "validation_boards",
            "sample_steps",
            "report_steps",
        ):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f"sudoku.{key} must be a positive integer")
        if type(settings["max_rating"]) is not int or settings["max_rating"] < 0:
            raise ValueError("sudoku.max_rating must be a nonnegative integer")

    def train(self, args: Namespace) -> Path:
        config = model_config(args.config)
        data, metadata = load_sudoku(config, args.data_dir)
        return Trainer(
            config,
            args.run_dir,
            metadata["fingerprint"],
            task=SudokuTask(config),
            source=RandomBoardSource(data["train"], config.global_batch),
            evaluator=SudokuEvaluator(data["validation"], config),
            selection=SudokuSelection(),
            update_options=UpdateOptions(
                config.microbatch, config.clip, config.amp, False, True
            ),
            lifecycle_options=LifecycleOptions(
                config.max_steps,
                config.validate_every,
                args.checkpoint_every,
                args.checkpoint_seconds,
                True,
                "systemlog/boards_per_second",
                config.global_batch,
                args.max_hours,
                args.stop_after_updates,
            ),
            selected_sizes=metadata["selected"],
            device=args.device,
            resume=args.resume,
            name=args.name or Path(args.experiment).stem,
            wandb_mode=args.wandb_mode,
            entity=args.entity,
            project=args.project,
            data_metadata=metadata,
        ).run()

    def report(self, args: Namespace) -> dict[str, Any]:
        config = model_config(args.config)
        run_path = Path(args.run_dir)
        state = load_checkpoint(run_path / "best.pt")

        device = torch.device(args.device)
        data, metadata = load_sudoku(config, args.data_dir)

        CheckpointManager(
            run_path,
            vars(config),
            metadata["fingerprint"],
        ).validate(state)
        model = BFN_DIT(config).to(device).eval()
        model.load_state_dict(state["ema"]["model"])

        evaluation_config = SimpleNamespace(
            **{**vars(config), "sample_steps": args.config.sudoku["report_steps"]}
        )
        scores = evaluate_sudoku(
            model, data["test"], evaluation_config, device, seed=5678
        )
        completed = state["progress"]["completed_updates"]

        report = {
            "scores": scores,
            "boards": len(data["test"]["questions"]),
            "seed": 5678,
            "sample_steps": evaluation_config.sample_steps,
            "checkpoint_step": completed,
            "max_rating": config.max_rating,
            "dataset_fingerprint": state["data_fingerprint"],
        }
        (run_path / "sudoku_report.json").write_text(json.dumps(report, indent=2))
        tracking = json.loads((run_path / "manifest.json").read_text())["wandb"]
        report_path = run_path / "report"

        tracker = ExperimentTracker(
            report_path,
            vars(evaluation_config),
            name=args.name or f"{tracking.get('name') or config.objective}-report",
            mode=args.wandb_mode or tracking["mode"],
            project=args.project or f"{tracking['project']}-report",
            entity=args.entity if args.entity is not None else tracking.get("entity"),
            resume=(report_path / "wandb_identity.json").exists(),
        )
        try:
            tracker.log(
                completed,
                {f"report/test_{key}": value for key, value in scores.items()},
            )
            for key, value in scores.items():
                tracker.run.summary[f"test/{key}"] = value
            count = min(4, config.microbatch, len(data["test"]["questions"]))
            questions = data["test"]["questions"][:count].to(device)
            predictions = solve_sudoku(
                model,
                questions,
                evaluation_config.sample_steps,
                torch.Generator(device=device).manual_seed(999),
            )
            rows = [
                [
                    decode_board(question.cpu()),
                    decode_board(prediction.cpu()),
                    decode_board(answer),
                ]
                for question, prediction, answer in zip(
                    questions, predictions, data["test"]["answers"][:count]
                )
            ]
            (run_path / "sudoku_samples.json").write_text(json.dumps(rows, indent=2))
            import wandb

            tracker.artifact(
                "samples/sudoku",
                wandb.Table(columns=["question", "prediction", "reference"], data=rows),
            )
        except BaseException:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish()
        print(json.dumps(report, indent=2))
        return report

    def smoke(self, config: SimpleNamespace) -> None:
        config = model_config(config)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        data = _smoke_data()
        with TemporaryDirectory(prefix="slm-sudoku-smoke-") as run_dir:
            Trainer(
                config,
                run_dir,
                data_fingerprint(data),
                task=SudokuTask(config),
                source=RandomBoardSource(data["train"], config.global_batch),
                evaluator=SudokuEvaluator(data["validation"], config),
                selection=SudokuSelection(),
                update_options=UpdateOptions(
                    config.microbatch, config.clip, config.amp, False, True
                ),
                lifecycle_options=LifecycleOptions(
                    config.max_steps,
                    config.validate_every,
                    2,
                    None,
                    True,
                    "systemlog/boards_per_second",
                    config.global_batch,
                ),
                selected_sizes={
                    name: len(split["questions"]) for name, split in data.items()
                },
                device="cpu",
                tracker_factory=NullTracker,
            ).run()
            assert (Path(run_dir) / "best.pt").is_file()
        print("Sudoku smoke check passed.")

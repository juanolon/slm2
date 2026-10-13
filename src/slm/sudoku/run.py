from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import torch

from slm.common.config import PARAMETERS
from slm.common.contracts import Report, TaskRunner
from slm.common.tracking import NullTracker
from slm.common.trainer import LifecycleOptions, Trainer, UpdateOptions

from .data import (
    RandomBoardSource,
    data_fingerprint,
    load_sudoku,
    pack_rows,
)
from .evaluation import SudokuEvaluator
from .report import DefaultReport, SweepReport
from .task import SudokuTask

SUDOKU_TRAINING_FIELDS = (
    "dataset",
    "max_rating",
    "max_train_split",
    "max_validation_split",
    "max_test_split",
    "validation_boards",
    "sample_steps",
)


def model_config(config):
    small = config.sudoku["dataset"] == "sudoku4x4"
    return SimpleNamespace(
        **{key: vars(config)[key] for key in PARAMETERS},
        **{key: config.sudoku[key] for key in SUDOKU_TRAINING_FIELDS},
        vocab=4 if small else 9,
        length=16 if small else 81,
    )


class SudokuSelection:
    """Rank ambiguous puzzles by valid completions, not reference agreement."""

    def __init__(self, ambiguous=False):
        self.ambiguous = ambiguous
        self.best_score = (-1.0, -1.0)

    def consider(self, metrics):
        score = (
            metrics[
                "validation/solve_rate" if self.ambiguous else "validation/exact_match"
            ],
            metrics[
                "validation/unit_validity"
                if self.ambiguous
                else "validation/blank_accuracy"
            ],
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
        missing = set(SUDOKU_TRAINING_FIELDS) - settings.keys()
        if missing:
            raise ValueError(f"Missing sudoku settings: {sorted(missing)}")
        for key in (
            "max_train_split",
            "max_validation_split",
            "max_test_split",
            "validation_boards",
            "sample_steps",
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
            selection=SudokuSelection(ambiguous=config.vocab == 4),
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
        reports: dict[str, Report] = {
            "default": DefaultReport(),
            "sweep": SweepReport(),
        }
        name = getattr(args, "report", "default")
        if name not in reports:
            raise ValueError(f"Unknown Sudoku report: {name}")
        return reports[name].run(args)

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

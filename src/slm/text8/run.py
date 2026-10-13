from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import torch

from slm.common.config import PARAMETERS
from slm.common.contracts import TaskRunner
from slm.common.tracking import NullTracker
from slm.common.trainer import LifecycleOptions, Trainer, UpdateOptions

from .data import ALPHABET, RandomWindowSource, load_text8
from .evaluation import Text8Evaluator
from .report import DefaultReport
from .task import Text8Task

TABLE1_BPC = {"weighted": 1.38, "simple": 1.42}


class BPCSelection:
    def __init__(self):
        self.best_bpc = float("inf")

    def consider(self, metrics):
        score = metrics["validation/bpc"]
        improved = score < self.best_bpc
        self.best_bpc = min(self.best_bpc, score)
        return improved

    def state_dict(self):
        return {"policy": "min-validation-bpc", "best_bpc": self.best_bpc}

    def load_state_dict(self, state):
        if state["policy"] != "min-validation-bpc":
            raise ValueError("Selection policy differs")
        self.best_bpc = state["best_bpc"]


TEXT8_TRAINING_FIELDS = ("vocab", "length", "validation_windows", "validation_draws")


def model_config(config):
    """Resolve shared and text8 training values for the notebook model interface."""
    return SimpleNamespace(
        **{key: vars(config)[key] for key in PARAMETERS},
        **{key: config.text8[key] for key in TEXT8_TRAINING_FIELDS},
    )


class Runner(TaskRunner):
    def validate_config(self, config: SimpleNamespace) -> None:
        settings = vars(config).get("text8")
        if not isinstance(settings, dict):
            raise TypeError("Config must include a text8: settings mapping")
        required = set(TEXT8_TRAINING_FIELDS) | {"draws", "windows"}
        missing = required - settings.keys()
        if missing:
            raise ValueError(f"Missing text8 settings: {sorted(missing)}")
        for key in (*TEXT8_TRAINING_FIELDS, "draws"):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f"text8.{key} must be a positive integer")
        if settings["vocab"] != len(ALPHABET):
            raise ValueError("text8.vocab must be 27")
        windows = settings["windows"]
        if windows is not None and (type(windows) is not int or windows < 1):
            raise ValueError("text8.windows must be a positive integer or null")

    def train(self, args: Namespace) -> Path:
        config = model_config(args.config)
        train_tokens, validation_tokens, _, fingerprint = load_text8(args.data_dir)
        return Trainer(
            config,
            args.run_dir,
            fingerprint,
            task=Text8Task(config),
            source=RandomWindowSource(train_tokens, config.global_batch, config.length),
            evaluator=Text8Evaluator(validation_tokens, config),
            selection=BPCSelection(),
            update_options=UpdateOptions(
                config.microbatch, config.clip, config.amp, True, False
            ),
            lifecycle_options=LifecycleOptions(
                config.max_steps,
                config.validate_every,
                args.checkpoint_every,
                args.checkpoint_seconds,
                False,
                "systemlog/characters_per_second",
                config.global_batch * config.length,
                args.max_hours,
                args.stop_after_updates,
            ),
            device=args.device,
            name=args.name or Path(args.experiment).stem,
            resume=args.resume,
            wandb_mode=args.wandb_mode,
            entity=args.entity,
            project=args.project,
        ).run()

    def report(self, args: Namespace) -> dict[str, Any]:
        return DefaultReport().run(args)

    def smoke(self, config: SimpleNamespace) -> None:
        config = model_config(config)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        train_tokens = torch.arange(512) % config.vocab
        validation_tokens = torch.arange(128) % config.vocab
        with TemporaryDirectory(prefix="slm-smoke-") as run_dir:
            Trainer(
                config,
                run_dir,
                "periodic-fake-v1",
                task=Text8Task(config),
                source=RandomWindowSource(
                    train_tokens, config.global_batch, config.length
                ),
                evaluator=Text8Evaluator(validation_tokens, config),
                selection=BPCSelection(),
                update_options=UpdateOptions(
                    config.microbatch, config.clip, config.amp, True, False
                ),
                lifecycle_options=LifecycleOptions(
                    config.max_steps,
                    config.validate_every,
                    2,
                    None,
                    False,
                    "systemlog/characters_per_second",
                    config.global_batch * config.length,
                ),
                device="cpu",
                tracker_factory=NullTracker,
            ).run()
        print("Smoke check passed.")

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
from slm.slm.core import seed_all

from .data import ALPHABET, RandomWindowSource, load_text8
from .evaluation import Text8Evaluator, evaluate_stream, generate
from .task import Text8Task


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
        config = model_config(args.config)
        settings = args.config.text8
        run_path = Path(args.run_dir)

        state = load_checkpoint(run_path / "best.pt")

        device = torch.device(args.device)
        _, valid, test, fingerprint = load_text8(args.data_dir)

        CheckpointManager(run_path, vars(config), fingerprint).validate(state)
        model = BFN_DIT(config).to(device).eval()
        model.load_state_dict(state["ema"]["model"])
        completed = state["progress"]["completed_updates"]
        tracking = json.loads((run_path / "manifest.json").read_text())["wandb"]
        draws, windows = settings["draws"], settings["windows"]

        validation = evaluate_stream(
            model,
            valid,
            config,
            windows or len(valid) // config.length,
            draws,
            seed=1234,
        )
        testing = evaluate_stream(
            model, test, config, windows or len(test) // config.length, draws, seed=5678
        )
        report = {
            "checkpoint_step": completed,
            "objective": config.objective,
            "validation": validation,
            "test": testing,
        }
        (run_path / "final_report.json").write_text(json.dumps(report, indent=2))
        report_path = run_path / "report"

        tracker = ExperimentTracker(
            report_path,
            vars(config),
            name=args.name or f"{tracking.get('name') or config.objective}-report",
            mode=args.wandb_mode or tracking["mode"],
            project=args.project or f"{tracking['project']}-report",
            entity=args.entity if args.entity is not None else tracking.get("entity"),
            resume=(report_path / "wandb_identity.json").exists(),
        )
        try:
            tracker.log(
                completed,
                {
                    "report/test_bpc": testing["bpc"],
                    "report/test_mc_se": testing["mc_se"],
                    "report/validation_bpc": validation["bpc"],
                },
            )

            tracker.run.summary["final_test_bpc"] = testing["bpc"]
            seed_all(123)

            tokens = generate(model, 4, 256, 27, steps=999, device=device)
            texts = [
                "".join(ALPHABET[token] for token in row)
                for row in tokens.cpu().tolist()
            ]
            (run_path / "samples.txt").write_text("\n\n".join(texts))
            import wandb

            tracker.artifact(
                "samples/text8",
                wandb.Table(columns=["text"], data=[[text] for text in texts]),
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

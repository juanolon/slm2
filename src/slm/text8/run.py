from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import torch

from slm.common.checkpoint import CheckpointManager, load_checkpoint
from slm.common.config import PARAMETERS
from slm.common.contracts import Progress, TaskRunner
from slm.common.ema import EMA
from slm.common.schedules import LinearWarmup
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


def run_training(
    train_tokens,
    validation_tokens,
    fingerprint,
    config,
    run_dir,
    *,
    device="cuda",
    resume=False,
    wandb_mode="online",
    project="slm-paper",
    entity=None,
    checkpoint_every=250,
    checkpoint_seconds=600,
    max_hours=None,
    stop_after_updates=None,
    stop_request=None,
    tracker_factory=ExperimentTracker,
    name=None,
):
    device = torch.device(device)
    lifecycle = LifecycleOptions(
        config.max_steps,
        config.validate_every,
        checkpoint_every,
        checkpoint_seconds,
        False,
        "systemlog/characters_per_second",
        config.global_batch * config.length,
        max_hours,
        stop_after_updates,
    )
    run_path = Path(run_dir)
    manager = CheckpointManager(run_path, vars(config), fingerprint)
    if manager.last_path.exists() and not resume:
        raise FileExistsError("Use a fresh --run-dir or specify --resume")
    tracker = None
    try:
        tracker = tracker_factory(
            run_path,
            vars(config),
            name=name or config.objective,
            mode=wandb_mode,
            project=project,
            entity=entity,
            resume=resume,
        )
        seed_all(config.seed)

        model = BFN_DIT(config).to(device)
        ema = EMA(model, config.ema)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        source = RandomWindowSource(train_tokens, config.global_batch, config.length)
        lr_policy = LinearWarmup(config.lr, config.warmup)
        selection = BPCSelection()

        progress = (
            manager.restore(model, optimizer, ema, lr_policy, selection)
            if resume
            else Progress()
        )
        if progress.completed_updates >= config.max_steps:
            print(
                "All configured optimizer updates are complete; use the report command."
            )
            tracker.finish()
            return manager.last_path

        manifest = {
            "config": vars(config),
            "data_fingerprint": fingerprint,
            "torch": torch.__version__,
            "python": sys.version,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "wandb": dict(tracker.identity),
        }
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise

    return Trainer(
        model=model,
        task=Text8Task(config),
        source=source,
        optimizer=optimizer,
        lr_policy=lr_policy,
        ema=ema,
        evaluator=Text8Evaluator(validation_tokens, config),
        selection=selection,
        tracker=tracker,
        checkpoint_manager=manager,
        update_options=UpdateOptions(
            config.microbatch, config.clip, config.amp, True, False
        ),
        lifecycle_options=lifecycle,
        progress=progress,
        stop_request=stop_request,
    ).run()


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
        return run_training(
            train_tokens,
            validation_tokens,
            fingerprint,
            config,
            args.run_dir,
            device=args.device,
            name=args.name or Path(args.experiment).stem,
            resume=args.resume,
            wandb_mode=args.wandb_mode,
            entity=args.entity,
            project=args.project,
            checkpoint_every=args.checkpoint_every,
            checkpoint_seconds=args.checkpoint_seconds,
            max_hours=args.max_hours,
            stop_after_updates=args.stop_after_updates,
        )

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
            run_training(
                train_tokens,
                validation_tokens,
                "periodic-fake-v1",
                config,
                run_dir,
                device="cpu",
                tracker_factory=NullTracker,
                checkpoint_every=2,
                checkpoint_seconds=None,
            )
        print("Smoke check passed.")

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from slm.slm.backbone import BFN_DIT
from slm.slm.core import seed_all

from .checkpoint import CheckpointManager
from .contracts import Progress, Task
from .ema import EMA
from .schedules import LinearWarmup
from .tracking import ExperimentTracker


@dataclass(frozen=True)
class UpdateOptions:
    microbatch: int
    clip: float
    amp: bool
    # Notebook text8 explicitly checks the loss; Sudoku only checks gradient norm.
    check_finite_loss: bool
    # Preserve text8's Python accumulation versus Sudoku's device scalar accumulation.
    accumulate_on_device: bool


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def optimizer_update(
    model: nn.Module,
    task: Task,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    batch: Any,
    options: UpdateOptions,
) -> tuple[float, float]:
    """Call only after setting this update's LR and transferring the full batch.

    Exceptions propagate; callers must never checkpoint a failed update as a
    completed boundary. Preparation samples all timesteps before slicing.
    """
    if options.microbatch < 1:
        raise ValueError("microbatch must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prepared = task.prepare(batch)
    batch_size = task.batch_size(prepared)
    if batch_size < 1:
        raise ValueError("Empty effective batch")
    device = next(model.parameters()).device
    total = torch.zeros((), device=device) if options.accumulate_on_device else 0.0
    for start in range(0, batch_size, options.microbatch):
        stop = min(start + options.microbatch, batch_size)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if options.amp and device.type == "cuda"
            else nullcontext()
        )
        with context:
            loss = task.loss(model, task.slice_batch(prepared, start, stop))
        if loss.ndim != 0:
            raise ValueError("Task loss must be a scalar microbatch mean")
        if options.check_finite_loss and not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite task loss")
        fraction = (stop - start) / batch_size
        (loss * fraction).backward()
        if options.accumulate_on_device:
            total += loss.detach() * fraction
        else:
            total += loss.detach().item() * fraction
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), options.clip, error_if_nonfinite=True
    )
    optimizer.step()
    ema.update(model)
    return float(total), float(norm)


@dataclass(frozen=True)
class LifecycleOptions:
    max_steps: int
    validate_every: int
    checkpoint_every: int
    checkpoint_seconds: float | None
    validate_when_stopping: bool
    throughput_metric: str
    units_per_update: int
    max_hours: float | None = None
    stop_after_updates: int | None = None
    log_every: int = 100

    def __post_init__(self):
        for value in (
            self.max_steps,
            self.validate_every,
            self.checkpoint_every,
            self.units_per_update,
            self.log_every,
        ):
            if value < 1:
                raise ValueError("Update counts and intervals must be positive")
        for value in (self.checkpoint_seconds, self.max_hours, self.stop_after_updates):
            if value is not None and value <= 0:
                raise ValueError("Budgets and checkpoint interval must be positive")


class StopRequest:
    """Signals only set a flag; saving waits for a completed optimizer update."""

    def __init__(self):
        self.requested = False
        self.previous_handlers = {}

    def install(self):
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._request)
        return self

    def _request(self, signum, frame):
        self.requested = True
        print(
            "Stop requested; finishing the current update/evaluation before checkpointing.",
            flush=True,
        )

    def restore(self):
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)
        self.previous_handlers.clear()


class Trainer:
    def __init__(
        self,
        config,
        run_dir,
        fingerprint,
        *,
        task,
        source,
        evaluator,
        selection,
        update_options: UpdateOptions,
        lifecycle_options: LifecycleOptions,
        device="cuda",
        resume=False,
        wandb_mode="online",
        project=None,
        entity=None,
        name=None,
        stop_request=None,
        tracker_factory=ExperimentTracker,
        selected_sizes=None,
        data_metadata=None,
    ):
        device = torch.device(device)
        self.task, self.source = task, source
        self.evaluator, self.selection = evaluator, selection
        self.stop = stop_request or StopRequest()
        self.update_options = update_options
        self.options = lifecycle_options
        run_path = Path(run_dir)
        self.checkpoints = CheckpointManager(
            run_path,
            vars(config),
            fingerprint,
        )
        if self.checkpoints.last_path.exists() and not resume:
            raise FileExistsError("Use a fresh --run-dir or specify --resume")
        tracker = None
        try:
            # Tracker startup randomness is discarded by fresh seeding or RNG restore.
            tracker = tracker_factory(
                run_path,
                vars(config),
                name=name or config.objective,
                mode=wandb_mode,
                project=project,
                entity=entity,
                resume=resume,
            )
            self.tracker = tracker
            seed_all(config.seed)
            self.model = BFN_DIT(config).to(device)
            self.ema = EMA(self.model, config.ema)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
            self.lr_policy = LinearWarmup(config.lr, config.warmup)
            self.progress = (
                self.checkpoints.restore(
                    self.model, self.optimizer, self.ema, self.lr_policy, self.selection
                )
                if resume
                else Progress()
            )
            if self.progress.completed_updates < self.options.max_steps:
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
                    "parameter_count": sum(p.numel() for p in self.model.parameters()),
                    "wandb": dict(tracker.identity),
                }
                if selected_sizes is not None:
                    manifest["selected"] = selected_sizes
                run_path.mkdir(parents=True, exist_ok=True)
                (run_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
                if data_metadata is not None:
                    (run_path / "data_selection.json").write_text(
                        json.dumps(data_metadata, indent=2)
                    )
        except BaseException:
            if tracker is not None:
                tracker.finish(exit_code=1)
            raise

    def run(self):
        options = self.options
        if self.progress.completed_updates >= options.max_steps:
            print(
                "All configured optimizer updates are complete; use the report command."
            )
            self.tracker.finish()
            return self.checkpoints.last_path
        device = next(self.model.parameters()).device
        previous_elapsed = self.progress.elapsed_seconds
        clock = time.monotonic()
        last_save = interval_start = clock
        session_updates = interval_updates = 0
        interval_loss = 0.0

        def save(is_best=False):
            self.progress.elapsed_seconds = previous_elapsed + time.monotonic() - clock
            self.checkpoints.save(
                self.model,
                self.optimizer,
                self.ema,
                self.lr_policy,
                self.selection,
                self.progress,
                self.tracker.identity,
                is_best=is_best,
            )

        try:
            self.stop.install()
            while (
                self.progress.completed_updates < options.max_steps
                and not self.stop.requested
            ):
                lr = self.lr_policy(self.progress.completed_updates)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                batch = move_to_device(self.source.next_batch(), device)
                loss, norm = optimizer_update(
                    self.model,
                    self.task,
                    self.optimizer,
                    self.ema,
                    batch,
                    self.update_options,
                )
                # No code in the exception path saves partially failed optimizer state.
                self.progress.completed_updates += 1
                completed = self.progress.completed_updates
                session_updates += 1
                interval_updates += 1
                interval_loss += loss
                elapsed = time.monotonic() - clock
                stopping = (
                    self.stop.requested
                    or (
                        options.max_hours is not None
                        and elapsed >= options.max_hours * 3600
                    )
                    or (
                        options.stop_after_updates is not None
                        and session_updates >= options.stop_after_updates
                    )
                )
                validation_due = (
                    completed % options.validate_every == 0
                    or completed == options.max_steps
                )
                metrics = {}
                improved = False
                evaluated = validation_due and (
                    not stopping or options.validate_when_stopping
                )
                if evaluated:
                    old_mode = self.ema.model.training
                    try:
                        with torch.no_grad():
                            self.ema.model.eval()
                            metrics.update(self.evaluator(self.ema.model))
                    finally:
                        self.ema.model.train(old_mode)
                    improved = self.selection.consider(metrics)
                if (
                    completed == 1
                    or completed % options.log_every == 0
                    or evaluated
                    or stopping
                    or completed == options.max_steps
                ):
                    duration = max(time.monotonic() - interval_start, 1e-9)
                    metrics.update(
                        {
                            "train/loss": interval_loss / interval_updates,
                            "train/lr": lr,
                            "train/grad_norm": norm,
                            options.throughput_metric: interval_updates
                            * options.units_per_update
                            / duration,
                            "systemlog/elapsed_seconds": previous_elapsed
                            + time.monotonic()
                            - clock,
                        }
                    )
                    if device.type == "cuda":
                        metrics["systemlog/peak_gpu_gib"] = (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        )
                    self.tracker.log(completed, metrics)
                    print(
                        json.dumps({"optimizer_step": completed, **metrics}), flush=True
                    )
                    interval_loss, interval_updates = 0.0, 0
                    interval_start = time.monotonic()
                save_due = (
                    completed % options.checkpoint_every == 0
                    or validation_due
                    or stopping
                    or (
                        options.checkpoint_seconds is not None
                        and time.monotonic() - last_save >= options.checkpoint_seconds
                    )
                )
                if save_due:
                    save(improved)
                    last_save = time.monotonic()
                if stopping:
                    break
            save()
        except BaseException:
            self.tracker.finish(exit_code=1)
            raise
        else:
            self.tracker.finish()
        finally:
            self.stop.restore()
        return self.checkpoints.last_path

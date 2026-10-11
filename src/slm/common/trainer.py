from __future__ import annotations

import json
import signal
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .contracts import Progress, Task
from .ema import EMA


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
        *,
        model,
        task,
        source,
        optimizer,
        lr_policy,
        ema,
        evaluator,
        selection,
        tracker,
        checkpoint_manager,
        update_options,
        lifecycle_options,
        progress=None,
        stop_request=None,
    ):
        self.model, self.task, self.source = model, task, source
        self.optimizer, self.lr_policy, self.ema = optimizer, lr_policy, ema
        self.evaluator, self.selection, self.tracker = evaluator, selection, tracker
        self.checkpoints = checkpoint_manager
        self.update_options, self.options = update_options, lifecycle_options
        self.progress = progress or Progress()
        self.stop = stop_request or StopRequest()

    def run(self):
        device = next(self.model.parameters()).device
        options = self.options
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

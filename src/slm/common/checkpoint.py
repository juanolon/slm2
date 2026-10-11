from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import Progress


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Cannot restore CUDA RNG on a different device topology")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_save(state: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> dict:
    # Only load trusted checkpoints: optimizer/RNG states require pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "config",
        "data_fingerprint",
        "model",
        "optimizer",
        "ema",
        "lr_policy",
        "selection",
        "progress",
        "rng",
        "tracking",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(
            "Incomplete checkpoint or legacy notebook layout; start a new packaged run"
        )
    return state


class CheckpointManager:
    def __init__(self, run_dir, config, fingerprint):
        self.run_dir = Path(run_dir)
        self.config = dict(config)
        self.fingerprint = fingerprint

    @property
    def last_path(self):
        return self.run_dir / "last.pt"

    def validate(self, state):
        if state["config"] != self.config:
            raise ValueError("Checkpoint configuration differs; start a new run")
        if state["data_fingerprint"] != self.fingerprint:
            raise ValueError("Checkpoint dataset fingerprint differs")

    def save(
        self,
        model,
        optimizer,
        ema,
        lr_policy,
        selection,
        progress,
        tracking_identity,
        *,
        is_best=False,
    ):
        state = {
            "config": self.config,
            "data_fingerprint": self.fingerprint,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "lr_policy": lr_policy.state_dict(),
            "selection": selection.state_dict(),
            "progress": asdict(progress),
            "rng": capture_rng(),
            "tracking": dict(tracking_identity),
        }
        atomic_save(state, self.last_path)
        if is_best:
            atomic_save(state, self.run_dir / "best.pt")

    def restore(self, model, optimizer, ema, lr_policy, selection):
        state = load_checkpoint(self.last_path)
        self.validate(state)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        lr_policy.load_state_dict(state["lr_policy"])
        selection.load_state_dict(state["selection"])
        restore_rng(state["rng"])
        return Progress(**state["progress"])

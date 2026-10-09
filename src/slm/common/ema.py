from __future__ import annotations

import copy

import torch
from torch import nn


def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    assert 0 <= decay < 1
    with torch.no_grad():
        for ema_parameter, model_parameter in zip(
            ema.parameters(), model.parameters(), strict=True
        ):
            ema_parameter.mul_(decay).add_(model_parameter, alpha=1 - decay)
        for ema_buffer, model_buffer in zip(
            ema.buffers(), model.buffers(), strict=True
        ):
            ema_buffer.copy_(model_buffer)


def warmed_decay(decay: float, update_count: int) -> float:
    return min(decay, (1 + update_count) / (10 + update_count))


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    def update(self, model: nn.Module) -> None:
        self.updates += 1
        update_ema(self.model, model, warmed_decay(self.decay, self.updates))

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "updates": self.updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state) -> None:
        if state["decay"] != self.decay:
            raise ValueError("EMA policy differs from checkpoint")
        self.model.load_state_dict(state["model"])
        self.updates = int(state["updates"])

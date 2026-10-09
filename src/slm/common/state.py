"""Small component contracts. No diffusion or experiment mathematics live here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor, nn


@dataclass
class Progress:
    """Completed optimizer updates and cumulative training time for resuming."""

    completed_updates: int = 0
    elapsed_seconds: float = 0.0


class Stateful(Protocol):
    """Interface for components whose state is saved in checkpoints."""

    def state_dict(self) -> dict[str, Any]:
        """Return the component's checkpoint state."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the component from its saved state."""
        ...


class TrainingSource(Stateful, Protocol):
    """Supply training batches and save any sampling state needed to resume."""

    def next_batch(self) -> Any:
        """Sample and return the next effective training batch."""
        ...


class Task(Protocol):
    """Interface for batch preparation and loss computation, not optimizer updates.

    Prepare full-batch timesteps before splitting; defer corruption if needed.

    loss() returns a scalar *microbatch mean*. The trainer weights it by
    microbatch size / effective batch size. Preparation runs outside autocast;
    loss runs inside it. Tasks may corrupt in either phase to preserve RNG order.
    """

    def prepare(self, batch: Any) -> Any:
        """Prepare the full batch, sampling timesteps before microbatch splitting."""
        ...

    def batch_size(self, prepared: Any) -> int:
        """Return the number of examples in the effective batch."""
        ...

    def slice_batch(self, prepared: Any, start: int, stop: int) -> Any:
        """Select examples in [start, stop) from the prepared batch."""
        ...

    def loss(self, model: nn.Module, microbatch: Any) -> Tensor:
        """Return a differentiable scalar microbatch mean using the task's reduction."""
        ...


class Evaluator(Protocol):
    """Interface for model evaluation outside the training objective."""

    def __call__(self, model: nn.Module) -> Mapping[str, float]:
        """Evaluate the supplied model and return named metrics."""
        ...


class Selection(Stateful, Protocol):
    """Track the best validation result for checkpoint selection and resuming."""

    def consider(self, metrics: Mapping[str, float]) -> bool:
        """Update selection state; return whether strictly improved."""
        ...


class Tracker(Protocol):
    """Interface for logging metrics and artifacts and closing the tracking run."""

    identity: Mapping[str, Any]

    def log(self, optimizer_step: int, metrics: Mapping[str, Any]) -> None:
        """Record named metrics at a completed optimizer update."""
        ...

    def artifact(self, name: str, value: Any) -> None:
        """Record a named artifact, such as a table or plot."""
        ...

    def finish(self, exit_code: int = 0) -> None:
        """Close the tracking run, marking success (zero) or failure (nonzero)."""
        ...

from dataclasses import dataclass


def learning_rate(step: int, base_lr: float, warmup: int) -> float:
    assert step >= 0 and warmup >= 0
    return base_lr if warmup == 0 else base_lr * min(step / warmup, 1.0)


@dataclass(frozen=True)
class LinearWarmup:
    base_lr: float
    warmup: int

    def __call__(self, completed_updates: int) -> float:
        return learning_rate(completed_updates, self.base_lr, self.warmup)

    def state_dict(self) -> dict:
        return {"base_lr": self.base_lr, "warmup": self.warmup}

    def load_state_dict(self, state) -> None:
        if state != self.state_dict():
            raise ValueError("Learning-rate policy differs from checkpoint")

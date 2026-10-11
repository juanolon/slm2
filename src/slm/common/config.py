from pathlib import Path
from types import SimpleNamespace

import yaml

PARAMETERS = (
    "T",
    "hidden",
    "heads",
    "blocks",
    "cond",
    "dropout",
    "global_batch",
    "microbatch",
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "warmup",
    "max_steps",
    "ema",
    "clip",
    "seed",
    "objective",
    "validate_every",
    "amp",
)


def validate_config(config: SimpleNamespace) -> None:
    missing = set(PARAMETERS) - vars(config).keys()
    if missing:
        raise ValueError(f"Missing parameter: {sorted(missing)}")
    for key in (
        "T",
        "hidden",
        "heads",
        "blocks",
        "cond",
        "global_batch",
        "microbatch",
        "max_steps",
        "validate_every",
    ):
        value = vars(config)[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.global_batch % config.microbatch:
        raise ValueError(
                "Global batch size must be divisible by microbatch size"
        )
    if config.objective not in ("weighted", "simple"):
        raise ValueError("objective must be weighted or simple")
    if type(config.amp) is not bool:
        raise ValueError("amp must be boolean")
    if type(config.warmup) is not int or config.warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")


def load_config(experiment: str, task: str) -> SimpleNamespace:
    filename = Path(experiment)
    if filename.name != experiment or filename.suffix not in ("", ".yaml"):
        raise ValueError(
            "--experiment must be a filename in experiments/, with an optional .yaml suffix"
        )

    path = Path("experiments") / filename.with_suffix(".yaml")
    try:
        settings = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(settings, dict):
        raise TypeError(f"{path} must contain a settings mapping")
    if not isinstance(settings.get(task), dict):
        raise TypeError(f"{path} must contain a {task}: settings mapping")

    if "betas" in settings:
        settings["betas"] = tuple(settings["betas"])
    config = SimpleNamespace(**settings)
    validate_config(config)
    return config

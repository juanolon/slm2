from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

from slm.common.config import load_config
from slm.common.contracts import TaskRunner
from slm.defaults import DEFAULT_CONFIG
from slm.slm.samplers import SAMPLERS


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def add_general_arguments(parser):
    general = parser.add_argument_group("general settings")
    general.add_argument(
        "--experiment",
        required=True,
        help="YAML filename in experiments/, e.g. text8_weighted",
    )
    general.add_argument("--run-dir", type=Path, default=Path("runs"))
    general.add_argument("--data-dir", type=Path, default=Path("data"))
    general.add_argument("--device", default="cuda")
    general.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="tracking mode (default: online)",
    )
    general.add_argument(
        "--project",
        help="W&B project; reports append -report to this or the training project",
    )
    general.add_argument("--entity", help="W&B account or team")
    general.add_argument(
        "--name", help="W&B run name; defaults to the experiment filename stem"
    )


def add_training_arguments(parser):
    runtime = parser.add_argument_group("training runtime")
    runtime.add_argument("--resume", action="store_true", help="resume run-dir/last.pt")
    runtime.add_argument("--max-hours", type=positive_float)
    runtime.add_argument("--stop-after-updates", type=positive_int)
    runtime.add_argument("--checkpoint-every", type=positive_int, default=250)
    runtime.add_argument("--checkpoint-seconds", type=positive_float, default=600)


def add_tasks(command_parser):
    tasks = command_parser.add_subparsers(dest="task", required=True)
    text8 = tasks.add_parser("text8")
    add_general_arguments(text8)

    sudoku = tasks.add_parser("sudoku")
    add_general_arguments(sudoku)
    return text8, sudoku


def build_train_command(commands):
    parser = commands.add_parser("train", help="Train or resume an experiment")
    text8, sudoku = add_tasks(parser)
    add_training_arguments(text8)
    add_training_arguments(sudoku)


def build_report_command(commands):
    parser = commands.add_parser("report", help="Evaluate saved weights")
    text8, sudoku = add_tasks(parser)
    for task in (text8, sudoku):
        seeds = task.add_mutually_exclusive_group()
        seeds.add_argument("--sampling-seed", type=int, help="Run once using this seed")
        seeds.add_argument(
            "--sampling-seeds",
            type=positive_int,
            help="Generate N distinct random seeds and average N runs",
        )
        task.add_argument(
            "--sampler",
            choices=SAMPLERS,
            default="default",
            help="Sampling algorithm (default: default)",
        )
    text8.add_argument("--report", choices=("default",), default="default")
    sudoku.add_argument(
        "--report",
        choices=("default", "sweep", "heatmap"),
        default="default",
        help="Report implementation: basic, sampling sweep, or one-board heatmap",
    )
    sudoku.add_argument(
        "--legacy", action="store_true", help="Load a legacy notebook Sudoku checkpoint"
    )
    sudoku.add_argument(
        "--board-id",
        type=int,
        default=0,
        help="Heatmap: zero-based index in the selected test split",
    )
    sudoku.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
        help="Heatmap: checkpoint weights to inspect",
    )
    sudoku.add_argument(
        "--sampling-steps",
        type=positive_int,
        help="Default/heatmap sampling budget (defaults to report_steps or sample_steps)",
    )


def build_smoke_command(commands):
    parser = commands.add_parser("smoke", help="Run smoke test")
    tasks = parser.add_subparsers(dest="task", required=True)
    tasks.add_parser("text8")
    tasks.add_parser("sudoku")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SLM training")
    commands = parser.add_subparsers(dest="command", required=True)
    build_train_command(commands)
    build_report_command(commands)
    build_smoke_command(commands)

    args = parser.parse_args(argv)
    module_name = f"slm.{args.task}.run"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in (f"slm.{args.task}", module_name):
            raise
        parser.error(f"{args.task} is not implemented yet")
    runner: TaskRunner = module.Runner()
    try:
        args.config = (
            DEFAULT_CONFIG
            if args.command == "smoke"
            else load_config(args.experiment, args.task)
        )
        runner.validate_config(args.config)
        match args.command:
            case "train":
                runner.train(args)
            case "report":
                runner.report(args)
            case "smoke":
                runner.smoke(DEFAULT_CONFIG)
    except (TypeError, ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

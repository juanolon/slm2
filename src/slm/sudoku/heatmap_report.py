from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch

from slm.common.contracts import Report
from slm.common.reporting import mean_scores, sampling_settings
from slm.common.seeds import SAMPLE_SEED
from slm.slm.core import (
    one_hot_tokens,
    predict_log_probs,
    survival_ratio,
)
from slm.slm.samplers import get_sampler

from .data import decode_board
from .evaluation import sudoku_scores
from .report import _load_report, _start_report
from .task import clamp_clues, initial_candidates


def solve_sudoku(
    model,
    questions,
    steps,
    generator=None,
    *,
    sampler="default",
):
    """Mirror evaluation.solve_sudoku for one board, returning recorded frames too."""
    if questions.ndim != 2 or len(questions) != 1 or questions.shape[1] not in (16, 81):
        raise ValueError("Expected one 4x4 or 9x9 question, with shape (1, cells)")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    pruning = get_sampler(sampler)
    old_mode = model.training
    frames = []
    model.eval()
    try:
        with torch.inference_mode():
            candidates = initial_candidates(questions)
            initial = candidates[0].cpu().tolist()
            frames.append(
                {
                    "kind": "initial",
                    "step": 0,
                    "time": 1.0,
                    "earlier_time": None,
                    "reverse_survival_ratio": None,
                    "before": initial,
                    "sampled": initial,
                    "repaired": initial,
                    "after": initial,
                    "model_probability": None,
                    "retention_probability": None,
                    "draws": None,
                }
            )
            for index in range(steps, 0, -1):
                time = torch.full(
                    (len(questions),),
                    index / steps,
                    dtype=torch.float64,
                    device=questions.device,
                )
                earlier = torch.full_like(time, (index - 1) / steps)

                candidates, frame = pruning(
                    model, candidates, earlier, time, generator, record=True
                )
                candidates = clamp_clues(candidates, questions)
                frame.update(
                    kind="transition",
                    step=steps - index + 1,
                    time=time.item(),
                    earlier_time=earlier.item(),
                    reverse_survival_ratio=survival_ratio(
                        earlier, time, candidates.shape[-1]
                    ).item(),
                    after=candidates[0].cpu().tolist(),
                )
                frames.append(frame)
            log_probs = predict_log_probs(
                model, candidates, torch.zeros(len(questions), device=questions.device)
            )
            predicted = log_probs.argmax(-1)
            predicted = torch.where(questions >= 0, questions, predicted)
            final_mask = candidates[0].cpu().tolist()
            frames.append(
                {
                    "kind": "decode",
                    "step": steps + 1,
                    "time": 0.0,
                    "earlier_time": None,
                    "reverse_survival_ratio": None,
                    "before": final_mask,
                    "sampled": final_mask,
                    "repaired": final_mask,
                    "after": final_mask,
                    "model_probability": log_probs.double().exp()[0].cpu().tolist(),
                    "retention_probability": None,
                    "draws": None,
                }
            )
    finally:
        model.train(old_mode)
    return {"frames": frames, "prediction": predicted[0].cpu().tolist()}


def analyse_trajectory(trajectory, question, reference):
    frames = trajectory["frames"]
    cells = []
    for cell, digit in enumerate(reference):
        first_loss = next(
            (i for i, frame in enumerate(frames) if not frame["after"][cell][digit]),
            None,
        )
        wrong_singleton = any(
            sum(frame["after"][cell]) == 1 and not frame["after"][cell][digit]
            for frame in frames
        )
        cells.append(
            {
                "cell": cell,
                "clue": question[cell] >= 0,
                "first_loss": first_loss,
                "wrong_singleton": wrong_singleton,
                "multiple_until_decode": sum(frames[-1]["after"][cell]) > 1,
            }
        )
    return cells


def average_trajectories(runs, question, reference):
    """Mean probabilities and mask frequencies"""
    blanks = [c for c, digit in enumerate(question) if digit < 0]
    frames = []
    for index, template in enumerate(runs[0]["frames"]):
        samples = [run["frames"][index] for run in runs]
        frame = {
            key: template[key]
            for key in (
                "kind",
                "step",
                "time",
                "earlier_time",
                "reverse_survival_ratio",
            )
        }
        for field in ("model_probability", "retention_probability"):
            frame[field] = (
                torch.tensor([sample[field] for sample in samples], dtype=torch.float64)
                .mean(0)
                .tolist()
                if template[field] is not None
                else None
            )
        for field in ("before", "after"):
            frame[f"{field}_frequency"] = (
                torch.tensor([sample[field] for sample in samples], dtype=torch.float64)
                .mean(0)
                .tolist()
            )
        frame["mean_candidate_count"] = [sum(mask) for mask in frame["after_frequency"]]
        frame["reference_survival"] = [
            mask[digit] for mask, digit in zip(frame["after_frequency"], reference)
        ]
        frame["singleton_frequency"] = [
            sum(sum(sample["after"][c]) == 1 for sample in samples) / len(samples)
            for c in range(len(question))
        ]
        frame["reference_board_survival"] = sum(
            all(sample["after"][c][reference[c]] for c in blanks) for sample in samples
        ) / len(samples)
        frames.append(frame)
    prediction_frequency = (
        one_hot_tokens(
            torch.tensor([run["prediction"] for run in runs]),
            len(runs[0]["frames"][0]["after"][0]),
        )
        .double()
        .mean(0)
        .tolist()
    )
    return {"frames": frames, "prediction_frequency": prediction_frequency}


class HeatmapReport(Report):
    def run(self, args: Namespace) -> dict[str, Any]:
        board_id = getattr(args, "board_id", 0)
        if board_id < 0:
            raise ValueError("board-id must be a nonnegative test-split index")
        sampling = sampling_settings(args, SAMPLE_SEED)
        seed_count = sampling["sampling_seeds"]
        weights = getattr(args, "weights", "ema")
        config, model, test, completed, fingerprint, tracking = _load_report(
            args, weights=weights
        )
        if board_id >= len(test["questions"]):
            raise ValueError(
                f"board-id {board_id} is outside the selected test split ({len(test['questions'])} boards)"
            )
        steps = getattr(args, "sampling_steps", None)
        if steps is None:
            steps = args.config.sudoku.get("report_steps", config.sample_steps)
        question, answer = test["questions"][board_id], test["answers"][board_id]
        q, ref = question.tolist(), answer.tolist()
        device_question = question.to(args.device)
        seeds = sampling["sampling_seed_list"]
        runs = []
        for sampling_seed in seeds:
            generator = torch.Generator(device=device_question.device).manual_seed(
                sampling_seed
            )
            trajectory = solve_sudoku(
                model,
                device_question[None],
                steps,
                generator,
                sampler=sampling["sampler"],
            )
            prediction = torch.tensor(trajectory["prediction"], dtype=torch.long)
            runs.append(
                {
                    "seed": sampling_seed,
                    **trajectory,
                    "cells": analyse_trajectory(trajectory, q, ref),
                    "scores": {
                        key: value.item()
                        for key, value in sudoku_scores(
                            prediction[None], question[None], answer[None]
                        ).items()
                    },
                }
            )
        scores = mean_scores([run["scores"] for run in runs])
        metadata = {
            "board_id": board_id,
            "board_id_scope": "zero-based selected test-split index",
            "board_key": hashlib.sha256(decode_board(question).encode()).hexdigest(),
            "dataset": config.dataset,
            "rating": int(test["ratings"][board_id]),
            "checkpoint": str((Path(args.run_dir) / "best.pt").resolve()),
            "checkpoint_step": completed,
            "checkpoint_dataset_fingerprint": fingerprint,
            "weights": weights,
            **sampling,
            "sampling_steps": steps,
        }
        payload = {
            "metadata": metadata,
            "question": q,
            "reference": ref,
            "vocab": config.vocab,
            "scores": scores,
        }
        if seed_count == 1:
            payload.update(runs[0])
        else:
            payload.update(runs=runs, average=average_trajectories(runs, q, ref))
        report = {
            "metadata": metadata,
            "scores": scores,
            "html": "trajectory.html",
        }
        report_path, tracker = _start_report(
            args, {**vars(config), "sample_steps": steps, **metadata}, report, tracking
        )
        html_path = report_path / "trajectory.html"
        try:
            write_html(html_path, payload)
            tracker.log(
                completed,
                {f"report/heatmap/{key}": value for key, value in scores.items()},
            )
        except BaseException:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish()
        print(f"Self-contained heatmap report: {html_path.resolve()}")
        return {**report, "html_path": str(html_path)}


def write_html(path: Path, data):
    # Escape '<' so metadata cannot terminate the embedded JSON script element.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = (
        files("slm.sudoku").joinpath("trajectory.html").read_text(encoding="utf-8")
    )
    path.write_text(template.replace("__TRAJECTORY_DATA__", payload), encoding="utf-8")

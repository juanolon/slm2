from __future__ import annotations

import torch

from slm.slm.core import predict_log_probs, prune_step

from .data import valid_units
from .task import blank_mean, clamp_clues, initial_candidates


def solve_sudoku(model, questions, steps, generator=None):
    if steps < 1:
        raise ValueError("steps must be positive")
    old_mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            candidates = initial_candidates(questions)
            for index in range(steps, 0, -1):
                time = torch.full(
                    (len(questions),),
                    index / steps,
                    dtype=torch.float64,
                    device=questions.device,
                )
                earlier = torch.full_like(time, (index - 1) / steps)
                candidates = prune_step(model, candidates, earlier, time, generator)
                candidates = clamp_clues(candidates, questions)
            log_probs = predict_log_probs(
                model, candidates, torch.zeros(len(questions), device=questions.device)
            )
            predicted = log_probs.argmax(-1)
            return torch.where(questions >= 0, questions, predicted)
    finally:
        model.train(old_mode)


def sudoku_scores(predictions, questions, answers):
    assert predictions.shape == questions.shape == answers.shape
    units = valid_units(predictions)
    board_validity = units.all(-1)
    clues = ((questions < 0) | (predictions == questions)).all(-1)
    correct = predictions == answers
    blanks = questions < 0
    blank_accuracy = torch.where(
        blanks.any(-1),
        blank_mean(correct.float(), questions),
        torch.ones(len(questions), device=questions.device),
    )
    return {
        "exact_match": correct.all(-1).float(),
        "board_validity": board_validity.float(),
        "clue_preservation": clues.float(),
        "solve_rate": (board_validity & clues).float(),
        "blank_accuracy": blank_accuracy,
        "cell_accuracy": correct.float().mean(-1),
        "unit_validity": units.float().mean(-1),
    }


def evaluate_sudoku(model, split, config, device, limit=None, seed=1234):
    count = (
        len(split["questions"])
        if limit is None
        else min(limit, len(split["questions"]))
    )
    if count < 1:
        raise ValueError("No evaluation boards")
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {}
    for start in range(0, count, config.microbatch):
        questions = split["questions"][
            start : min(start + config.microbatch, count)
        ].to(device)
        answers = split["answers"][start : min(start + config.microbatch, count)].to(
            device
        )
        predictions = solve_sudoku(model, questions, config.sample_steps, generator)
        for key, values in sudoku_scores(predictions, questions, answers).items():
            totals[key] = totals.get(key, 0.0) + values.sum().item()
    return {key: value / count for key, value in totals.items()}


class SudokuEvaluator:
    def __init__(self, split, config):
        self.split, self.config = split, config

    def __call__(self, model):
        scores = evaluate_sudoku(
            model,
            self.split,
            self.config,
            next(model.parameters()).device,
            self.config.validation_boards,
            seed=1234,
        )
        return {f"validation/{key}": value for key, value in scores.items()}

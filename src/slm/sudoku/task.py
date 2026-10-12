from __future__ import annotations

import torch

from slm.slm.core import (
    one_hot_tokens,
    predict_log_probs,
    sample_forward,
    sample_steps,
    training_loss,
)


def initial_candidates(questions):
    assert questions.ndim == 2 and questions.shape[1] in (16, 81)
    vocab = 4 if questions.shape[1] == 16 else 9
    clues = one_hot_tokens(questions.clamp_min(0), vocab).bool()
    return torch.where((questions >= 0)[..., None], clues, torch.ones_like(clues))


def clamp_clues(candidate_mask, questions):
    assert candidate_mask.shape[:2] == questions.shape
    return torch.where(
        (questions >= 0)[..., None], initial_candidates(questions), candidate_mask
    )


def conditional_forward(answers, questions, time, generator=None):
    assert answers.shape == questions.shape
    vocab = 4 if questions.shape[1] == 16 else 9
    return clamp_clues(sample_forward(answers, time, vocab, generator), questions)


def blank_mean(cell_losses, questions):
    """Average over blanks within each board, then let the caller average boards."""
    assert cell_losses.shape == questions.shape
    blanks = questions < 0
    return torch.where(blanks, cell_losses, torch.zeros_like(cell_losses)).sum(
        -1
    ) / blanks.sum(-1).clamp_min(1)


class SudokuTask:
    def __init__(self, config):
        self.config = config

    def prepare(self, batch):
        questions, answers = batch
        steps = sample_steps(len(questions), self.config.T, device=questions.device)
        return questions, answers, steps

    def batch_size(self, prepared):
        return len(prepared[0])

    def slice_batch(self, prepared, start, stop):
        return tuple(item[start:stop] for item in prepared)

    def loss(self, model, microbatch):
        # Unlike text8, the notebook corrupts inside each microbatch's autocast context.
        questions, answers, steps = microbatch
        time = steps.float() / self.config.T
        candidates = conditional_forward(answers, questions, time)
        log_probs = predict_log_probs(model, candidates, time)
        cell_losses = training_loss(
            log_probs, answers, steps, self.config.T, self.config.objective
        )
        return blank_mean(cell_losses, questions).mean()

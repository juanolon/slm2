from __future__ import annotations

import math

import torch

from slm.slm.core import repair_empty

from .core import predict_log_probs, reverse_probs


def protected_prune_step(
    model, candidate_mask, earlier_time, time, generator=None, *, record=False
):
    """
    Normal prune step + protect highest probability candidate from being prune
    """
    if record and len(candidate_mask) != 1:
        raise ValueError("Recorded pruning requires a single board")
    with torch.no_grad():
        clean_probabilities = (
            predict_log_probs(model, candidate_mask, time.float()).double().exp()
        )
        clean_probabilities = clean_probabilities.masked_fill(~candidate_mask, 0.0)
        protected_digit = torch.multinomial(
            clean_probabilities.reshape(-1, clean_probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(*clean_probabilities.shape[:-1], 1)

        probs = reverse_probs(
            clean_probabilities,
            candidate_mask,
            earlier_time.double(),
            time.double(),
        )
        uniform_draws = torch.rand(
            probs.shape,
            dtype=torch.float64,
            device=probs.device,
            generator=generator,
        )

        sampled = uniform_draws < probs
        survivors = sampled & candidate_mask
        survivors.scatter_(-1, protected_digit, True)
        if not record:
            return survivors
        return survivors, {
            "before": candidate_mask[0].cpu().tolist(),
            "model_probability": clean_probabilities[0].cpu().tolist(),
            "retention_probability": probs[0].cpu().tolist(),
            "draws": uniform_draws[0].cpu().tolist(),
            "sampled": sampled[0].cpu().tolist(),
            "repaired": survivors[0].cpu().tolist(),
        }


def confidence_gated_prune_step(
    model,
    candidate_mask,
    earlier_time,
    time,
    generator=None,
    *,
    confidence_power=2.0,
    ensure_one=True,
    record=False,
):
    """
    select cells containing candidates with high probability
    still, protect highest probability candidate from being prune
    continue with pruning on the selected cells

    """
    if record and len(candidate_mask) != 1:
        raise ValueError("Recorded pruning requires a single board")
    if confidence_power <= 0:
        raise ValueError("confidence_power must be positive")

    with torch.no_grad():
        clean_probabilities = (
            predict_log_probs(model, candidate_mask, time.float()).double().exp()
        )
        clean_probabilities = clean_probabilities.masked_fill(~candidate_mask, 0.0)

        # choose cells containing more that one candidate
        unresolved_cells = candidate_mask.sum(dim=-1) > 1
        cell_confidence = clean_probabilities.max(dim=-1).values

        cell_selection_probability = cell_confidence.pow(confidence_power).masked_fill(
            ~unresolved_cells, 0.0
        )

        selection_draws = torch.rand(
            cell_selection_probability.shape,
            dtype=torch.float64,
            device=candidate_mask.device,
            generator=generator,
        )
        selected_cells = selection_draws < cell_selection_probability


        # If an unfinished board selected no cells, select its most
        # confident unresolved cell. This guarantees an update attempt.
        forced_selection = torch.zeros_like(selected_cells)
        if ensure_one:
            needs_selection = unresolved_cells.any(dim=-1) & ~selected_cells.any(dim=-1)
            best_cell = cell_confidence.masked_fill(
                ~unresolved_cells, -torch.inf
            ).argmax(dim=-1, keepdim=True)

            forced_selection.scatter_(-1, best_cell, needs_selection.unsqueeze(-1))
            selected_cells = selected_cells | forced_selection

        # protect high probability candidate
        protect_digit = torch.multinomial(
            clean_probabilities.reshape(-1, clean_probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(*clean_probabilities.shape[:-1], 1)

        # reverse transition probabilities
        probs = reverse_probs(
            clean_probabilities,
            candidate_mask,
            earlier_time.double(),
            time.double(),
        )

        uniform_draws = torch.rand(
            probs.shape,
            dtype=torch.float64,
            device=probs.device,
            generator=generator,
        )
        orig_reverse_candidates = (uniform_draws < probs) & candidate_mask

        proposed = orig_reverse_candidates & candidate_mask
        proposed.scatter_(-1, protect_digit, True)

        # apply new candidates to selected cells
        survivors = torch.where(
            selected_cells.unsqueeze(-1),
            proposed,
            candidate_mask,
        )

        if not record:
            return survivors

        return survivors, {
            "before": candidate_mask[0].cpu().tolist(),
            "model_probability": clean_probabilities[0].cpu().tolist(),
            "retention_probability": probs[0].cpu().tolist(),
            "draws": uniform_draws[0].cpu().tolist(),
            "sampled": orig_reverse_candidates[0].cpu().tolist(),
            "repaired": survivors[0].cpu().tolist(),
            "protected_digit_index": (protect_digit[0].squeeze(-1).cpu().tolist()),
            "cell_confidence": cell_confidence[0].cpu().tolist(),
            "cell_selection_probability": (
                cell_selection_probability[0].cpu().tolist()
            ),
            "cell_selection_draw": selection_draws[0].cpu().tolist(),
            "cell_selected": selected_cells[0].cpu().tolist(),
            "cell_selection_forced": (forced_selection[0].cpu().tolist()),
        }

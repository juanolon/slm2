from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .core import (
    expected_count,
    predict_log_probs,
    prune_step,
    repair_empty,
    reverse_probs,
)

if TYPE_CHECKING:
    from shape_extensions import IntVar

    B = IntVar("B")
    L = IntVar("L")
    K = IntVar("K")


def default_prune_step(
    model: nn.Module,
    candidate_mask: Tensor[[B, L, K]],
    earlier_time: Tensor[[B]],
    time: Tensor[[B]],
    generator: torch.Generator | None = None,
    *,
    record: bool = False,
):
    """Core pruning, with optional diagnostics using exactly the same RNG draws."""
    if not record:
        return prune_step(model, candidate_mask, earlier_time, time, generator)
    if len(candidate_mask) != 1:
        raise ValueError("Recorded pruning requires a single board")
    with torch.no_grad():
        clean_probabilities = (
            predict_log_probs(model, candidate_mask, time.float()).double().exp()
        )
        probs = reverse_probs(
            clean_probabilities, candidate_mask, earlier_time.double(), time.double()
        )
        uniform_draws = torch.rand(
            probs.shape, dtype=torch.float64, device=probs.device, generator=generator
        )
        sampled = uniform_draws < probs
        repaired = repair_empty(sampled, probs)
        return repaired, {
            "before": candidate_mask[0].cpu().tolist(),
            "model_probability": clean_probabilities[0].cpu().tolist(),
            "retention_probability": probs[0].cpu().tolist(),
            "draws": uniform_draws[0].cpu().tolist(),
            "sampled": sampled[0].cpu().tolist(),
            "repaired": repaired[0].cpu().tolist(),
        }


def protected_prune_step(
    model: nn.Module,
    candidate_mask: Tensor[[B, L, K]],
    earlier_time: Tensor[[B]],
    time: Tensor[[B]],
    generator: torch.Generator | None = None,
    *,
    record: bool = False,
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
            "protected_digit_index": protected_digit[0].squeeze(-1).cpu().tolist(),
        }


def confidence_gated_prune_step(
    model: nn.Module,
    candidate_mask: Tensor[[B, L, K]],
    earlier_time: Tensor[[B]],
    time: Tensor[[B]],
    generator: torch.Generator | None = None,
    *,
    confidence_power: float = 2.0,
    ensure_one: bool = True,
    record: bool = False,
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
        cell_confidence = clean_probabilities.amax(dim=-1)

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


def topk_prune_step(
    model: nn.Module,
    candidate_mask: Tensor[[B, L, K]],
    earlier_time: Tensor[[B]],
    time: Tensor[[B]],
    generator: torch.Generator | None = None,
    *,
    top_k: int = 5,
    protect_digit: bool = False,
    record: bool = False,
):
    """
    select the top-k most confident unresolved cells
    use a "hardcoded" retention of 0.5 for the selected cells
    """
    if record and len(candidate_mask) != 1:
        raise ValueError("Recorded pruning requires a single board")
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")

    with torch.no_grad():
        clean_probabilities = (
            predict_log_probs(model, candidate_mask, time.float()).double().exp()
        )
        clean_probabilities = clean_probabilities.masked_fill(~candidate_mask, 0.0)

        # choose cells containing more that one candidate
        unresolved_cells = candidate_mask.sum(dim=-1) > 1
        cell_confidence = clean_probabilities.amax(dim=-1)

        cell_selection_scores = cell_confidence.masked_fill(
            ~unresolved_cells, -torch.inf
        )

        k = min(top_k, candidate_mask.shape[-2])
        top_indices = cell_selection_scores.topk(k, dim=-1)[1]

        selected_cells = torch.zeros_like(unresolved_cells)
        selected_cells.scatter_(-1, top_indices, True)

        selected_cells = selected_cells & unresolved_cells

        protected_digit: Tensor[[B, L, 1]] | None = None
        if protect_digit:
            # protect high probability candidate
            protected_digit = torch.multinomial(
                clean_probabilities.reshape(-1, clean_probabilities.shape[-1]),
                num_samples=1,
                generator=generator,
            ).reshape(*clean_probabilities.shape[:-1], 1)

        baseline_retention = 0.5
        probs = (
            clean_probabilities + (1.0 - clean_probabilities) * baseline_retention
        ).masked_fill(~candidate_mask, 0.0)

        uniform_draws = torch.rand(
            probs.shape,
            dtype=torch.float64,
            device=probs.device,
            generator=generator,
        )
        orig_reverse_candidates = (uniform_draws < probs) & candidate_mask

        if protected_digit is not None:
            proposed = orig_reverse_candidates.clone()
            proposed.scatter_(-1, protected_digit, True)
        else:
            proposed = repair_empty(orig_reverse_candidates, probs)

        # apply new candidates to selected cells
        survivors = torch.where(
            selected_cells.unsqueeze(-1),
            proposed,
            candidate_mask,
        )

        if not record:
            return survivors

        frame = {
            "before": candidate_mask[0].cpu().tolist(),
            "model_probability": clean_probabilities[0].cpu().tolist(),
            "retention_probability": probs[0].cpu().tolist(),
            "draws": uniform_draws[0].cpu().tolist(),
            "sampled": orig_reverse_candidates[0].cpu().tolist(),
            "repaired": survivors[0].cpu().tolist(),
            "cell_confidence": cell_confidence[0].cpu().tolist(),
            "cell_selected": selected_cells[0].cpu().tolist(),
            "top_k": top_k,
            "baseline_retention": baseline_retention,
        }
        if protected_digit is not None:
            frame["protected_digit_index"] = (
                protected_digit[0].squeeze(-1).cpu().tolist()
            )
        return survivors, frame


class BudgetedPruner:
    """
    * excess of a cell is the current amount of candidates minus one
    * the remaining fraction describes how much of the excess should be kept, based on the expected count at the current time
    * the current step pruning budget is current excess - the target excess
    *
    * the remaining mass R is the sum of the model probabilities of the candidates currently in the cell
    * the score of a candidate represents the difference between R before and after one candidate is removed
    * we want to remove candidates that minimize the increase in the negative log of R
    * or we want to maximaze the prob that every cell's correct digit survives pruning (\times R)
    """

    def __init__(
        self,
        candidate_mask: Tensor[[B, L, K]],
        generator: torch.Generator | None = None,
    ) -> None:
        self.initial_excess: Tensor[tuple[int]] = (candidate_mask.sum(-1) - 1).sum(-1)
        self.rounding_offset: Tensor[tuple[int]] = torch.rand(
            (len(candidate_mask),),
            dtype=torch.float64,
            device=candidate_mask.device,
            generator=generator,
        )

    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        candidate_mask: Tensor[[B, L, K]],
        earlier_time: Tensor[[B]],
        time: Tensor[[B]],
        generator: torch.Generator | None = None,
        *,
        record: bool = False,
    ):
        batch_size, _, vocabulary_size = candidate_mask.shape
        if record and batch_size != 1:
            raise ValueError("Recorded pruning requires a single board")

        # alpha(s) = {K^s - 1} / {K - 1}
        remaining_fraction = (
            expected_count(earlier_time.double(), vocabulary_size) - 1
        ) / (vocabulary_size - 1)

        # stochastic rounding
        target_excess = torch.floor(
            self.initial_excess.double() * remaining_fraction + self.rounding_offset
        ).long()

        # current - target = number of candidates to remove
        removal_budget = (candidate_mask.sum(-1) - 1).sum(-1) - target_excess
        if (removal_budget < 0).any().item():
            raise ValueError(
                "The scheduled target_excess would require adding candidates"
            )

        clean_log_probabilities = predict_log_probs(
            model, candidate_mask, time.float()
        ).double()

        survivors = candidate_mask.clone()
        removals: list[tuple[int, int]] = []
        board_indices = torch.arange(batch_size, device=candidate_mask.device)
        for removal_index in range(int(removal_budget.max().item())):

            # R_i = log \sum_i p_ik for each cell k, ignoring removed candidates
            log_remaining_mass = torch.logsumexp(
                clean_log_probabilities.masked_fill(~survivors, -torch.inf), -1
            )
            # candidate must be kept and cell must have more than one candidate
            removable_candidates = survivors & (survivors.sum(-1) > 1).unsqueeze(-1)

            # score: log p_ik - log R_i
            log_relative_removed_mass = (
                clean_log_probabilities - log_remaining_mass.unsqueeze(-1)
            ).masked_fill(~removable_candidates, torch.inf)

            # minimum cost next removal
            chosen_indices = log_relative_removed_mass.flatten(1).argmin(-1)
            cell_indices, digit_indices = (
                chosen_indices // vocabulary_size,
                chosen_indices % vocabulary_size,
            )
            if record:
                removals.append((int(cell_indices[0]), int(digit_indices[0])))

            active_boards = removal_index < removal_budget
            survivors[
                board_indices[active_boards],
                cell_indices[active_boards],
                digit_indices[active_boards],
            ] = False

        if not record:
            return survivors

        return survivors, self._record(
            candidate_mask, clean_log_probabilities, removals
        )

    @staticmethod
    def _record(
        candidate_mask: Tensor[[B, L, K]],
        clean_log_probabilities: Tensor[[B, L, K]],
        removals: list[tuple[int, int]],
    ):
        """Build only the model-p, per-cell J and removal table used by the UI."""
        board_log_probabilities = clean_log_probabilities[0]
        log_initial_mass = torch.logsumexp(board_log_probabilities, -1)
        clean_probabilities = (
            board_log_probabilities - log_initial_mass[:, None]
        ).exp()
        remaining = candidate_mask[0].clone()
        cell_costs = torch.zeros_like(log_initial_mass)
        events = []
        for cell, digit in removals:
            log_mass_before = torch.logsumexp(
                board_log_probabilities[cell].masked_fill(~remaining[cell], -torch.inf),
                -1,
            )
            remaining[cell, digit] = False
            log_mass_after = torch.logsumexp(
                board_log_probabilities[cell].masked_fill(~remaining[cell], -torch.inf),
                -1,
            )
            incremental_cost = log_mass_before - log_mass_after
            cell_costs[cell] += incremental_cost
            events.append(
                {
                    "cell": cell,
                    "removed_digit_index": digit,
                    "model_probability": float(clean_probabilities[cell, digit]),
                    "incremental_cost": float(incremental_cost),
                    "objective_cost_after": float(cell_costs.sum()),
                }
            )
        return {
            "before": candidate_mask[0].cpu().tolist(),
            "model_probability": clean_probabilities.cpu().tolist(),
            "cell_objective_cost": cell_costs.cpu().tolist(),
            "removal_events": events,
        }

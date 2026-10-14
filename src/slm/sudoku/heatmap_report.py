from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from slm.common.contracts import Report
from slm.common.seeds import SAMPLE_SEED
from slm.slm.core import (
    one_hot_tokens,
    predict_log_probs,
    repair_empty,
    reverse_probs,
    survival_ratio,
)

from .data import decode_board
from .evaluation import sudoku_scores
from .report import _load_report, _start_report
from .task import clamp_clues, initial_candidates


def _changes(before, after):
    changes = []
    for cell in (before != after).any(-1)[0].nonzero().flatten().tolist():
        old, new = before[0, cell], after[0, cell]
        digits = lambda mask: (mask.nonzero().flatten() + 1).tolist()
        changes.append(
            {
                "cell": cell,
                "before": digits(old),
                "after": digits(new),
                "removed": digits(old & ~new),
                "added": digits(~old & new),
            }
        )
    return changes


def prune_step(model, candidate_mask, earlier_time, time, generator=None):
    """Mirror core.prune_step; additionally return this transition's diagnostics."""
    with torch.no_grad():
        # TODO: add temperature to predict_log_probs
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
        changes = _changes(sampled, repaired)
        return repaired, {
            "before": candidate_mask[0].cpu().tolist(),
            "model_probability": clean_probabilities[0].cpu().tolist(),
            "retention_probability": probs[0].cpu().tolist(),
            "draws": uniform_draws[0].cpu().tolist(),
            "sampled": sampled[0].cpu().tolist(),
            "repaired": repaired[0].cpu().tolist(),
            "events": [{"kind": "empty_set_repair", "changes": changes}]
            if changes
            else [],
        }


def solve_sudoku(model, questions, steps, generator=None, observer=None):
    """Mirror evaluation.solve_sudoku for one board, returning recorded frames too."""
    if questions.ndim != 2 or len(questions) != 1 or questions.shape[1] not in (16, 81):
        raise ValueError("Expected one 4x4 or 9x9 question, with shape (1, cells)")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    old_mode = model.training
    frames = []
    model.eval()
    try:
        with torch.inference_mode():
            candidates = initial_candidates(questions)
            initial_changes = _changes(torch.ones_like(candidates), candidates)
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
                    "events": [{"kind": "clue_enforcement", "changes": initial_changes}]
                    if initial_changes
                    else [],
                }
            )
            if observer is not None:
                observer(0, candidates)
            for index in range(steps, 0, -1):
                time = torch.full(
                    (len(questions),),
                    index / steps,
                    dtype=torch.float64,
                    device=questions.device,
                )
                earlier = torch.full_like(time, (index - 1) / steps)
                candidates, frame = prune_step(
                    model, candidates, earlier, time, generator
                )
                before_enforcement = candidates
                candidates = clamp_clues(candidates, questions)
                changes = _changes(before_enforcement, candidates)
                if changes:
                    frame["events"].append(
                        {"kind": "clue_enforcement", "changes": changes}
                    )
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
                if observer is not None:
                    observer(steps - index + 1, candidates)
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
                    "events": [],
                    "decoded": predicted[0].cpu().tolist(),
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
        wrong_singleton = next(
            (
                i
                for i, frame in enumerate(frames)
                if sum(frame["after"][cell]) == 1 and not frame["after"][cell][digit]
            ),
            None,
        )
        interventions = [
            i
            for i, frame in enumerate(frames)
            if any(
                change["cell"] == cell
                for event in frame["events"]
                for change in event["changes"]
            )
        ]
        cells.append(
            {
                "cell": cell,
                "clue": question[cell] >= 0,
                "reference": digit,
                "first_loss": first_loss,
                "wrong_singleton": wrong_singleton,
                "interventions": interventions,
                "multiple_until_decode": sum(frames[-1]["before"][cell]) > 1,
                "incorrect_final": trajectory["prediction"][cell] != digit,
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
        seed_count = getattr(args, "sampling_seeds", 1)
        if type(seed_count) is not int or seed_count < 1:
            raise ValueError("sampling-seeds must be a positive integer")
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
        seed = getattr(args, "sampling_seed", None)
        if seed is None:
            seed = SAMPLE_SEED
        question, answer = test["questions"][board_id], test["answers"][board_id]
        q, ref = question.tolist(), answer.tolist()
        device_question = question.to(args.device)
        seeds = [seed + index for index in range(seed_count)]
        runs = []
        for sampling_seed in seeds:
            generator = torch.Generator(device=device_question.device).manual_seed(
                sampling_seed
            )
            trajectory = solve_sudoku(model, device_question[None], steps, generator)
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
        scores = {
            key: sum(run["scores"][key] for run in runs) / seed_count
            for key in runs[0]["scores"]
        }
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
            "sampling_seed": seed,
            "sampling_seeds": seed_count,
            "sampling_seed_list": seeds,
            "sampling_scope": "same board, independent seed streams (not a replay of the batched sweep)",
            "sampling_steps": steps,
            "sampling_budget": steps,
            "interventions": {
                "clue_enforcement": True,
                "empty_set_repair": True,
                "constraint_filtering": False,
                "other": False,
            },
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
            "by_seed": [{"seed": run["seed"], "scores": run["scores"]} for run in runs],
        }
        report_path, tracker = _start_report(
            args, {**vars(config), "sample_steps": steps, **metadata}, report, tracking
        )
        html_path = report_path / "trajectory.html"
        try:
            write_html(html_path, payload)
            log_wandb(tracker, completed, payload)
        except BaseException:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish()
        print(f"Self-contained heatmap report: {html_path.resolve()}")
        return {**report, "html_path": str(html_path)}


def log_wandb(tracker, completed, data):
    import wandb

    runs = data.get("runs", [data])
    count = len(runs)

    for key, value in data["metadata"].items():
        tracker.run.summary[f"heatmap/{key}"] = value
    tracker.log(
        completed,
        {f"report/heatmap/{key}": value for key, value in data["scores"].items()},
    )
    steps = [frame["step"] for frame in runs[0]["frames"]]
    blanks = [i for i, digit in enumerate(data["question"]) if digit < 0]
    survival = [
        sum(
            run["frames"][column]["after"][i][data["reference"][i]]
            for run in runs
            for i in blanks
        )
        / (count * len(blanks))
        if blanks
        else 1.0
        for column in range(len(steps))
    ]
    counts = [
        sum(sum(run["frames"][column]["after"][i]) for run in runs for i in blanks)
        / (count * len(blanks))
        if blanks
        else 0.0
        for column in range(len(steps))
    ]
    for name, values in (
        ("reference_cell_survival", survival),
        ("mean_blank_candidate_count", counts),
    ):
        tracker.artifact(
            f"heatmap/{name}",
            wandb.plot.line_series(
                xs=steps,
                ys=[values],
                keys=[name],
                title=name,
                xname="sampling column (0=initial, N+1=decode)",
            ),
        )
    transitions = [
        frame for frame in runs[0]["frames"] if frame["kind"] == "transition"
    ]
    tracker.artifact(
        "heatmap/reverse_survival_ratio",
        wandb.plot.line_series(
            xs=[frame["step"] for frame in transitions],
            ys=[[frame["reverse_survival_ratio"] for frame in transitions]],
            keys=["reverse survival ratio"],
            title="Reverse survival schedule",
            xname="sampling transition",
        ),
    )
    kinds = sorted(
        {
            event["kind"]
            for run in runs
            for frame in run["frames"]
            for event in frame["events"]
        }
    )
    if kinds:
        tracker.artifact(
            "heatmap/interventions",
            wandb.plot.line_series(
                xs=steps,
                ys=[
                    [
                        sum(
                            len(event["changes"])
                            for run in runs
                            for event in run["frames"][column]["events"]
                            if event["kind"] == kind
                        )
                        / count
                        for column in range(len(steps))
                    ]
                    for kind in kinds
                ],
                keys=kinds,
                title="Mean cells modified by interventions per seed",
                xname="sampling column (0=initial, N+1=decode)",
            ),
        )
    side = data["vocab"]
    events = [
        [
            run["seed"],
            frame["step"],
            frame["time"],
            event["kind"],
            change["cell"],
            change["cell"] // side + 1,
            change["cell"] % side + 1,
            str(change["removed"]),
            str(change["added"]),
            str(change["before"]),
            str(change["after"]),
        ]
        for run in runs
        for frame in run["frames"]
        for event in frame["events"]
        for change in event["changes"]
    ]
    tracker.artifact(
        "heatmap/events",
        wandb.Table(
            columns=[
                "seed",
                "step",
                "time_before",
                "intervention",
                "cell",
                "row",
                "column",
                "removed",
                "added",
                "before",
                "after",
            ],
            data=events,
        ),
    )
    failures = [
        [
            run["seed"],
            cell["cell"],
            cell["cell"] // side + 1,
            cell["cell"] % side + 1,
            cell["reference"] + 1,
            run["prediction"][cell["cell"]] + 1,
            cell["first_loss"],
            cell["wrong_singleton"],
            str(cell["interventions"]),
            cell["multiple_until_decode"],
        ]
        for run in runs
        for cell in run["cells"]
        if cell["first_loss"] is not None
        or cell["wrong_singleton"] is not None
        or cell["interventions"]
        or cell["multiple_until_decode"]
        or cell["incorrect_final"]
    ]
    tracker.artifact(
        "heatmap/failures",
        wandb.Table(
            columns=[
                "seed",
                "cell",
                "row",
                "column",
                "reference",
                "prediction",
                "first_reference_loss",
                "first_wrong_singleton",
                "intervention_columns",
                "multiple_at_decode",
            ],
            data=failures,
        ),
    )
    question = decode_board(torch.tensor(data["question"], dtype=torch.long))
    reference = decode_board(torch.tensor(data["reference"], dtype=torch.long))
    tracker.artifact(
        "heatmap/board",
        wandb.Table(
            columns=["seed", "question", "reference", "prediction"],
            data=[
                [
                    run["seed"],
                    question,
                    reference,
                    decode_board(torch.tensor(run["prediction"], dtype=torch.long)),
                ]
                for run in runs
            ],
        ),
    )
    if count > 1:
        keys = list(data["scores"])
        tracker.artifact(
            "heatmap/seed_scores",
            wandb.Table(
                columns=["seed", *keys],
                data=[
                    [run["seed"], *[run["scores"][key] for key in keys]] for run in runs
                ],
            ),
        )


def write_html(path: Path, data):
    # Escape '<' so metadata cannot terminate the embedded JSON script element.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    path.write_text(HTML.replace("__TRAJECTORY_DATA__", payload), encoding="utf-8")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sudoku trajectory heatmap</title><style>
.scroll{position:relative}.time-cursor{position:absolute;top:0;bottom:20px;border:2px solid #f59e0b;box-sizing:border-box;pointer-events:none}
body{font:14px system-ui,sans-serif;margin:20px;color:#172033;background:#f8fafc}h1,h2{margin:12px 0}section{background:white;border:1px solid #d3dbe7;border-radius:6px;padding:14px;margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:12px}label{display:inline-block;margin:5px 15px 5px 0}select,input{padding:5px}.boards{display:flex;gap:30px;flex-wrap:wrap}.board{display:grid;gap:1px;background:#64748b;border:2px solid #172033}.board button{width:32px;height:32px;border:0;background:#fff;cursor:pointer;font:16px system-ui}.board .clue{font-weight:bold;background:#e2e8f0}.board .incorrect{color:#b91c1c}.board .selected{box-shadow:inset 0 0 0 3px #f59e0b}.scroll{overflow-x:auto;border:1px solid #cbd5e1;margin:8px 0;padding-bottom:20px}canvas{display:block}.legend{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.gradient{display:inline-block;width:130px;height:14px;background:linear-gradient(to right,#fff,#1455c5);border:1px solid #888}.masked{background:repeating-linear-gradient(45deg,#eee 0px,#eee 3px,#aaa 3px,#aaa 4px);padding:3px}.hint{color:#475569}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.metric{background:#eff6ff;border:1px solid #bfdbfe;padding:12px;border-radius:5px}.metric strong,.metric span{display:block}.metric span{font-size:17px;margin-top:5px;overflow-wrap:anywhere}.danger{color:#b91c1c}.good{color:#15803d}.candidate-board{display:grid;gap:2px;background:#334155;border:2px solid #334155;max-width:950px;min-width:450px}.candidate-cell{position:relative;background:#fff;min-height:90px;padding:7px;cursor:pointer}.candidate-cell.clue{background:#e2e8f0}.candidate-cell.peer{background:#f3e8ff}.candidate-cell.partial{box-shadow:inset 0 0 0 4px #d97706}.candidate-cell.lost{box-shadow:inset 0 0 0 4px #dc2626}.candidate-cell.selected{box-shadow:inset 0 0 0 4px #2563eb}.candidate-cell.selected.lost{box-shadow:inset 0 0 0 4px #dc2626}.candidate-cell.selected.partial{box-shadow:inset 0 0 0 4px #d97706}.candidate-cell.selected.lost::after,.candidate-cell.selected.partial::after{content:"";position:absolute;inset:0;border:4px dashed #2563eb;pointer-events:none}.candidate-cell.lost .cell-caption{color:#b91c1c}.candidate-digits{display:grid;gap:2px;min-height:54px;align-content:center}.candidate-digit{border:1px solid #cbd5e1;border-radius:3px;padding:1px;cursor:pointer;font-size:12px}.candidate-digit.reference{font-weight:700}.candidate-digit.decoded{outline:3px solid #f59e0b;outline-offset:1px}.candidate-cell.singleton .candidate-digit{font-size:30px}.cell-caption{font-size:11px;margin:3px 0;overflow-wrap:anywhere}.table-scroll{overflow:auto;max-height:480px;padding-bottom:20px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;text-align:left;border:1px solid #cbd5e1;white-space:nowrap}th{background:#e2e8f0;position:sticky;top:0}tr.selected{background:#fff0c2}tr.peer{background:#f3e8ff}tbody tr{cursor:pointer}#stepSlider{width:min(650px,70vw)}details{margin-top:10px}#frameSummary{font-weight:bold}
.expanded-panel{background:#fff;border:1px solid #d3dbe7;border-radius:6px;padding:14px;margin:14px 0}.panel-heading{display:flex;align-items:center;justify-content:space-between;gap:10px}.sticky-toggle{cursor:pointer;border:1px solid #94a3b8;border-radius:4px;padding:6px 10px;background:#f8fafc;font-size:18px}.sticky-toggle[aria-pressed="true"]{background:#dbeafe;border-color:#2563eb}.expanded-panel.is-sticky{position:sticky;top:8px;z-index:20;max-height:calc(100vh - 16px);overflow-y:auto;box-shadow:0 4px 14px #17203333;border-color:#2563eb}
</style></head><body>
<h1>Sudoku sampling trajectory</h1>
<section id="averageSection" hidden><h2 id="averageTitle">Average across seeds</h2><details id="averageInfo"><summary>Info and mean scores</summary><pre id="averageSummary"></pre></details><details><summary>Per-seed scores</summary><pre id="averageOutcomes"></pre></details>
<p>Colors are arithmetic mean probabilities across all seeds, including mask-forced zeros. Counts are mean candidate counts. The candidate board shows digits present in <b>any</b> seed, with exact survival counts underneath (surviving seeds / total seeds). It is not one sampled board. Cell borders: amber = reference present in some seeds, red = absent, blue = selected. Selected status cells have alternating blue/status-color segments. Bold digits mark the reference; there are no green borders. No majority-vote solution is used to calculate scores. A bold overview border means every seed is a singleton; a bold expanded border means that digit survives in every seed.</p>
<label>Mean probability <select id="averageMode"><option value="model_probability">Model digit probability</option><option value="retention_probability">Sampler retention probability</option></select></label><label><input id="averagePin" type="checkbox">Pin mean inspector</label>
<div id="averageTimelineScroll" class="scroll"><canvas id="averageTimeline"></canvas><div id="averageTimelineCursor" class="time-cursor"></div></div>
<h3>Mean reference-digit probability (blank cells)</h3><div id="averageOverviewScroll" class="scroll"><canvas id="averageOverview"></canvas><div id="averageOverviewCursor" class="time-cursor"></div></div>
<div id="averageExpandedPanel" class="expanded-panel"><div class="panel-heading"><h3 id="averageExpandedTitle">Mean candidate probabilities</h3><button id="averageSticky" class="sticky-toggle" type="button" aria-pressed="false" aria-label="Pin mean expanded view" title="Pin expanded view while scrolling">📌</button></div><div id="averageExpandedScroll" class="scroll"><canvas id="averageExpanded"></canvas><div id="averageExpandedCursor" class="time-cursor"></div></div></div><pre id="averageInspection"></pre>
<h3>Candidate survival frequencies through time</h3><label>Time <input id="averageSlider" type="range" min="0" step="1" value="0"></label><label>Candidate state <select id="averageStage"><option value="after_frequency">After transition</option><option value="before_frequency">Before transition</option></select></label><p id="averageFrameLabel"></p><p id="averageFrameSummary"></p><div class="scroll"><div id="averageBoard" class="candidate-board"></div></div></section>
<details id="individualTrajectory" open><summary><b>Individual seed trajectory</b> — actual candidates, draws and interventions</summary><label>Seed <select id="seedSelect"></select></label>
<section><details id="info"><summary><b>Info: identity and outcome</b></summary><pre id="metadata"></pre></details><div id="boards" class="boards"></div></section>
<section><h2>Probability and cell filters</h2>
<label>Probability <select id="mode"><option value="model_probability">Model digit probability (masked, normalized)</option><option value="retention_probability">Sampler retention probability (Bernoulli inclusion)</option></select></label>
<label><input id="clues" type="checkbox">Show fixed clues</label>
<label>Cells <select id="filter"><option value="all">All</option><option value="lost">Lost reference digit</option><option value="wrong">Incorrect singleton</option><option value="intervention">Required intervention</option><option value="multiple">Multiple candidates until decoding</option></select></label>
<p><b>Timing:</b> color = probability <b>before</b> pruning; borders/counts = candidates <b>after</b> sampling, repair and clue enforcement. Initial state is separate; final decoding chooses a digit without changing the candidate set.</p>
<div class="legend"><span>0 <span class="gradient"></span> 1 (fixed probability scale)</span><span class="masked">Already eliminated: zero forced by mask</span><span style="color:#dc2626">▲ first reference loss</span><span>▣ singleton / surviving digit border</span><span style="color:#b45309">★ intervention</span><span>— probability not applicable</span><span>✓ final selected digit; compact ✓/× = reference match/mismatch</span></div>
<p id="modeNote" class="hint"></p>
</section>
<section><h2>Timeline: timestep and reverse survival ratio</h2><p>Click a column to freeze the whole-board view at that transition. Each label shows actual step, pre-pruning timestep t, and schedule ratio s. The blue line underneath is s on a fixed 0–1 scale; s is a schedule factor, not the observed surviving-cell fraction. Stars appear only for actual forced changes. No empty intervention lanes are shown.</p><div id="timelineScroll" class="scroll"><canvas id="timeline"></canvas><div id="timelineCursor" class="time-cursor"></div></div><details id="stepEvents"><summary>Actual forced changes at the selected step</summary><div id="eventCells"></div><pre id="eventDetails"></pre></details></section>
<section><h2>Compact overview: reference-digit probability</h2><p>One row per originally blank cell, in board-position order. Numbers are candidate counts after the column; hover/click for before/after counts and exact probabilities. Hatching marks masked zeros after elimination.</p><div id="overviewScroll" class="scroll"><canvas id="overview"></canvas><div id="overviewCursor" class="time-cursor"></div></div><p id="empty"></p></section>
<section id="expandedPanel" class="expanded-panel"><div class="panel-heading"><h2 id="expandedTitle">Expanded candidate view</h2><button id="expandedSticky" class="sticky-toggle" type="button" aria-pressed="false" aria-label="Pin expanded view" title="Pin expanded view while scrolling">📌</button></div><p>One row per digit. <b>Bold borders</b> mean the digit survives the transition. The reference row is labeled ★. Click a Sudoku cell, overview row or digit box to link the views.</p><div id="expandedScroll" class="scroll"><canvas id="expanded"></canvas><div id="expandedCursor" class="time-cursor"></div></div></section>
<section><h2>Box inspection</h2><label><input id="pin" type="checkbox">Pin inspection (clicking a box pins it)</label><p class="hint">Hover previews a single box without moving the board's selected time. Click a box to pin it and synchronize the board below. Uncheck Pin to resume hover previews.</p><div id="inspection"></div><details><summary>Intermediate candidate sets and actual interventions</summary><pre id="inspectionDetails"></pre></details></section>
<section><h2>Whole-board candidates through time</h2><label>Time <input id="stepSlider" type="range" min="0" step="1" value="0"></label><button id="previousStep">Previous</button> <button id="nextStep">Next</button><p id="frameLabel"></p><label>Candidate state <select id="stage"><option value="after">After transition</option><option value="before">Before transition</option></select></label><p id="frameSummary"></p><p>Each cell shows all candidates at the selected state; the sets shrink and singletons become large digits. n = candidate count, −/+ = removed/added counts, ★ = actual forced change. Cell positions stay fixed. Colors still use probabilities <b>before</b> the selected transition. Cell borders: red = reference lost, blue = selected. A selected error cell has alternating red/blue segments. Bold digit = reference; gold = final decoded digit. Purple cells share the selected cell's row, column or box (structural peers, not all attention connections).</p><div class="scroll"><div id="candidateBoard" class="candidate-board"></div></div>
<h3>Cross-cell comparison at this time</h3><label>Probability target <select id="compareTarget"><option value="reference">Reference digit in each cell</option><option value="selected">Selected digit across all cells</option></select></label><label><input id="changedOnly" type="checkbox">Only changed/intervened cells</label><p id="compareScope"></p><div class="table-scroll"><table><thead><tr><th>Cell</th><th>Target</th><th>Before candidates</th><th>After candidates</th><th>Removed / added</th><th>Model p before</th><th>Retention q</th><th>Model p next call</th><th>Δ model p</th></tr></thead><tbody id="comparisonRows"></tbody></table></div></section>
</details>
<script id="trajectory" type="application/json">__TRAJECTORY_DATA__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('trajectory').textContent), RUNS=DATA.runs??[DATA], A=DATA.average, side=DATA.vocab;
let D={...DATA,...RUNS[0]}, F=D.frames, averageColumn=0;
const $=id=>document.getElementById(id), cellName=c=>'r'+(Math.floor(c/side)+1)+'c'+(c%side+1), digits=m=>m.flatMap((v,i)=>v?[i+1]:[]);
let selected=D.question.findIndex(v=>v<0), selectedColumn=0, selectedDigit=0, columns=[], rows=[], width=20;
if(selected<0)selected=0;selectedDigit=D.reference[selected];
const labelWidth=140, rowHeight=24;
columns=F.map((_,i)=>i);
$('metadata').textContent=JSON.stringify({...D.metadata,selected_seed:D.seed,scores:D.scores},null,2);
$('stepSlider').max=F.length-1;
$('averageSection').hidden=!A;$('individualTrajectory').open=!A;
RUNS.forEach((run,index)=>{const option=document.createElement('option');option.value=index;option.textContent='Seed '+run.seed;$('seedSelect').appendChild(option)});$('seedSelect').value='0';
if(A){$('averageSlider').max=A.frames.length-1;$('averageTitle').textContent='Average across '+RUNS.length+' seeds';$('averageSummary').textContent=JSON.stringify({board_id:DATA.metadata.board_id,board_key:DATA.metadata.board_key,seeds:DATA.metadata.sampling_seed_list,sampling_steps:DATA.metadata.sampling_steps,mean_scores:DATA.scores},null,2);$('averageOutcomes').textContent=JSON.stringify(RUNS.map(run=>({seed:run.seed,scores:run.scores})),null,2)}
function matches(c){const a=D.cells[c], f=$('filter').value;return f==='all'||f==='lost'&&a.first_loss!==null||f==='wrong'&&a.wrong_singleton!==null||f==='intervention'&&a.interventions.length>0||f==='multiple'&&a.multiple_until_decode}
function columnLabel(f){return f.kind==='initial'?'initial t=1 s=—':f.kind==='decode'?'decode t=0 s=—':'step '+f.step+' t='+f.time.toFixed(4)+' s='+f.reverse_survival_ratio.toFixed(4)}
function cellEvents(f,c,d=null){return f.events.flatMap(e=>e.changes.filter(a=>a.cell===c&&(d===null||a.added.includes(d+1)||a.removed.includes(d+1))).map(a=>({kind:e.kind,...a})))}
function mechanisms(f,c,d){
 if(f.kind==='initial')return D.question[c]>=0?['Initial clue enforcement']:['Initial unrestricted blank'];
 if(f.kind==='decode')return [f.decoded[c]===d?'Selected by final decoding (mask unchanged)':'Not selected by final decoding (mask unchanged)'];
 const result=[];
 if(!f.before[c][d])result.push('Unavailable before pruning: ordinary candidate masking forces probability to zero');
 else result.push(f.sampled[c][d]?'Retained by Bernoulli sampling':'Removed by Bernoulli sampling');
 if(f.sampled[c][d]!==f.repaired[c][d])result.push('Changed by empty-set repair');
 if(f.repaired[c][d]!==f.after[c][d])result.push('Changed by clue enforcement');
 return result;
}
function inspectionData(fi,c,d){
 const f=F[fi];
 return {cell:cellName(c),digit:d+1,reference:D.reference[c]+1,step:f.step,kind:f.kind,
  time_before:f.time,time_after:f.earlier_time,reverse_survival_ratio:f.reverse_survival_ratio,
  model_digit_probability:f.model_probability?.[c][d]??null,
  sampler_retention_probability:f.retention_probability?.[c][d]??null,
  uniform_draw:f.draws?.[c][d]??null,candidate_before:f.before[c][d],candidate_after:f.after[c][d],
  candidates_before:digits(f.before[c]),candidates_after:digits(f.after[c]),
  change_mechanism:mechanisms(f,c,d),final_digit:D.prediction[c]+1};
}
function metric(root,label,value){
 const card=document.createElement('div');card.className='metric';
 const title=document.createElement('strong');title.textContent=label;
 const text=document.createElement('span');text.textContent=value===null||value===undefined?'Not applicable':String(value);
 card.appendChild(title);card.appendChild(text);root.appendChild(card);
}
function inspect(fi,c,d){
 const f=F[fi],info=inspectionData(fi,c,d),root=$('inspection');root.replaceChildren();
 const title=document.createElement('h3');title.textContent=info.cell+' · digit '+info.digit+' (reference '+info.reference+') · '+columnLabel(f);root.appendChild(title);
 const cards=document.createElement('div');cards.className='metrics';
 metric(cards,'Model digit probability p',info.model_digit_probability);
 metric(cards,'Sampler retention probability q',info.sampler_retention_probability);
 metric(cards,'Uniform draw u (kept if u < q)',info.uniform_draw);
 metric(cards,'Candidate status',(info.candidate_before?'Available':'Excluded')+' → '+(info.candidate_after?'available':'excluded'));
 root.appendChild(cards);
 const sets=document.createElement('p');sets.textContent='Candidates ('+info.candidates_before.length+' → '+info.candidates_after.length+'): ['+info.candidates_before.join(', ')+'] → ['+info.candidates_after.join(', ')+']';root.appendChild(sets);
 const mechanism=document.createElement('strong');mechanism.className=info.candidate_before&&!info.candidate_after?'danger':'';mechanism.textContent=info.change_mechanism.join('; ');root.appendChild(mechanism);
 const refState=document.createElement('p');refState.className=f.after[c][D.reference[c]]?'good':'danger';
 refState.textContent=f.after[c][D.reference[c]]?'Reference digit still available.':'Reference digit lost; first elimination at step '+D.cells[c].first_loss+'.';root.appendChild(refState);
 $('inspectionDetails').textContent=JSON.stringify({time_before:info.time_before,time_after:info.time_after,
  reverse_survival_ratio:info.reverse_survival_ratio,after_sampling:digits(f.sampled[c]),
  after_repair:digits(f.repaired[c]),interventions:cellEvents(f,c),final_digit:info.final_digit},null,2);
}
function selectCell(c,d=D.reference[c]){selected=c;selectedDigit=d;if(D.question[c]>=0)$('clues').checked=true;if(!matches(c))$('filter').value='all';$('pin').checked=true;render();inspect(selectedColumn,c,d)}
function boardCell(c){selectCell(c,$('compareTarget').value==='selected'?selectedDigit:D.reference[c])}
function selectTime(fi){selectedColumn=Math.max(0,Math.min(F.length-1,fi));moveCursors();drawWholeBoard();showEvents(selectedColumn);inspect(selectedColumn,selected,selectedDigit)}
function drawBoards(){const root=$('boards');root.replaceChildren();[['Original puzzle',D.question],['Reference solution',D.reference],['Final prediction',D.prediction]].forEach(([title,values],boardIndex)=>{
 const wrap=document.createElement('div'), heading=document.createElement('h3');heading.textContent=title;wrap.appendChild(heading);const board=document.createElement('div');board.className='board';board.style.gridTemplateColumns='repeat('+side+',32px)';const box=Math.sqrt(side);
 values.forEach((v,c)=>{const b=document.createElement('button');b.textContent=v<0?'·':v+1;b.title=cellName(c)+' (cell '+c+')';if(D.question[c]>=0)b.classList.add('clue');if(boardIndex===2&&v!==D.reference[c])b.classList.add('incorrect');if(c===selected)b.classList.add('selected');if((c%side+1)%box===0&&c%side<side-1)b.style.borderRight='2px solid #172033';if((Math.floor(c/side)+1)%box===0&&Math.floor(c/side)<side-1)b.style.borderBottom='2px solid #172033';b.onclick=()=>boardCell(c);board.appendChild(b)});wrap.appendChild(board);root.appendChild(wrap)
})}
function prepare(canvas,height){canvas.width=labelWidth+width*columns.length;canvas.height=height;const ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font='11px system-ui';return ctx}
function marker(ctx,x,y,c,f,d,compact){const events=cellEvents(f,c,compact?null:d);if(events.length){ctx.fillStyle='#b45309';ctx.fillText('★',x+width-11,y+11)}if(D.cells[c].first_loss===f.step&&(compact||d===D.reference[c])){ctx.fillStyle='#dc2626';ctx.beginPath();ctx.moveTo(x+1,y+1);ctx.lineTo(x+9,y+1);ctx.lineTo(x+1,y+9);ctx.closePath();ctx.fill()}}
function paint(ctx,x,y,c,d,f,compact){const p=f[$('mode').value]?.[c][d];const masked=f.kind!=='initial'&&!f.before[c][d];
 if(p===undefined||p===null)ctx.fillStyle='#e2e8f0';else ctx.fillStyle='rgb('+Math.round(255-235*p)+','+Math.round(255-170*p)+','+Math.round(255-58*p)+')';ctx.fillRect(x,y,width,rowHeight);
 if(masked){ctx.fillStyle='#ededed';ctx.fillRect(x,y,width,rowHeight);ctx.strokeStyle='#a1a1aa';ctx.beginPath();for(let i=0;i<width+rowHeight;i+=6){ctx.moveTo(x+Math.max(0,i-rowHeight),y+Math.min(i,rowHeight));ctx.lineTo(x+Math.min(i,width),y+Math.max(0,i-width))}ctx.stroke()}
 const survives=compact?digits(f.after[c]).length===1:f.after[c][d];ctx.strokeStyle=survives?'#172033':'#cbd5e1';ctx.lineWidth=survives?2:0.5;ctx.strokeRect(x+1,y+1,width-2,rowHeight-2);
 if(compact&&width>=12){ctx.fillStyle=masked||p===undefined||p<0.65?'#172033':'white';ctx.fillText(digits(f.after[c]).length,x+width/2-3,y+17)}else if((p===undefined||p===null)&&width>=12){ctx.fillStyle='#64748b';ctx.fillText('—',x+3,y+17)}
 if(f.kind==='decode'&&width>=12&&(compact||f.decoded[c]===d)){ctx.fillStyle='#172033';ctx.fillText(f.decoded[c]===d?'✓':'×',x+2,y+22)}
 marker(ctx,x,y,c,f,d,compact);
}
function moveCursors(){['timelineCursor','overviewCursor','expandedCursor'].forEach(id=>{$(id).style.left=(labelWidth+selectedColumn*width)+'px';$(id).style.width=width+'px'})}
function drawOverview(){const ctx=prepare($('overview'),Math.max(30,rows.length*rowHeight));rows.forEach((c,r)=>{const y=r*rowHeight;ctx.fillStyle=c===selected?'#fff0c2':'#f1f5f9';ctx.fillRect(0,y,labelWidth,rowHeight);ctx.fillStyle='#172033';ctx.fillText(cellName(c)+' ref '+(D.reference[c]+1)+(D.question[c]>=0?' [clue]':''),5,y+17);columns.forEach((fi,i)=>paint(ctx,labelWidth+i*width,y,c,D.reference[c],F[fi],true));if(c===selected){ctx.strokeStyle='#f59e0b';ctx.lineWidth=2;ctx.strokeRect(1,y+1,$('overview').width-2,rowHeight-2)}});$('empty').textContent=rows.length?'':'No cells match this filter.'}
function highlightRow(ctx,canvas,y){ctx.strokeStyle='#f59e0b';ctx.lineWidth=3;ctx.strokeRect(1,y+1,canvas.width-2,rowHeight-2)}
function drawExpanded(){const ctx=prepare($('expanded'),side*rowHeight);$('expandedTitle').textContent='Expanded '+cellName(selected)+' — reference digit '+(D.reference[selected]+1);for(let d=0;d<side;d++){const y=d*rowHeight;ctx.fillStyle=d===selectedDigit?'#fff0c2':d===D.reference[selected]?'#fef3c7':'#f1f5f9';ctx.fillRect(0,y,labelWidth,rowHeight);ctx.fillStyle='#172033';ctx.fillText((d===D.reference[selected]?'★ reference ':'digit ')+(d+1),5,y+17);columns.forEach((fi,i)=>paint(ctx,labelWidth+i*width,y,selected,d,F[fi],false));if(d===selectedDigit)highlightRow(ctx,$('expanded'),y)}}
function drawTimeline(id='timeline',frames=F){
 const ctx=prepare($(id),240);ctx.fillStyle='#172033';ctx.fillText('reverse survival s',3,212);
 columns.forEach((fi,i)=>{const x=labelWidth+i*width,f=frames[fi];ctx.save();ctx.translate(x+width/2,178);ctx.rotate(-Math.PI/2);ctx.fillText(columnLabel(f),0,0);ctx.restore();if(f.events?.length){ctx.fillStyle='#b45309';ctx.fillText('★',x+1,237);ctx.fillStyle='#172033'}});
 ctx.strokeStyle='#2563eb';ctx.lineWidth=2;ctx.beginPath();let started=false;
 columns.forEach((fi,i)=>{const s=frames[fi].reverse_survival_ratio;if(s===null)return;const x=labelWidth+(i+.5)*width,y=225-35*s;if(!started){ctx.moveTo(x,y);started=true}else ctx.lineTo(x,y)});ctx.stroke();
}
function showEvents(fi){
 const f=F[fi];$('stepEvents').hidden=f.events.length===0;
 $('eventDetails').textContent=JSON.stringify(f.events,null,2);
 const root=$('eventCells');root.replaceChildren();
 [...new Set(f.events.flatMap(e=>e.changes.map(a=>a.cell)))].forEach(c=>{const b=document.createElement('button');b.textContent=cellName(c);b.onclick=()=>boardCell(c);root.appendChild(b)});
}
function peer(c){const box=Math.sqrt(side);return Math.floor(c/side)===Math.floor(selected/side)||c%side===selected%side||Math.floor(c/side/box)===Math.floor(selected/side/box)&&Math.floor(c%side/box)===Math.floor(selected%side/box)}
function targetDigit(c){return $('compareTarget').value==='selected'?selectedDigit:D.reference[c]}
function probabilityText(value){return value===null||value===undefined?'—':Number(value.toPrecision(6)).toString()}
function comparisonData(fi,c){
 const f=F[fi],d=targetDigit(c),p=f.model_probability?.[c][d]??null,next=F[fi+1]?.model_probability?.[c][d]??null;
 return {cell:c,target:d+1,before:digits(f.before[c]),after:digits(f.after[c]),
  removed:digits(f.before[c].map((v,i)=>v&&!f.after[c][i])),added:digits(f.after[c].map((v,i)=>v&&!f.before[c][i])),
  model_probability:p,retention_probability:f.retention_probability?.[c][d]??null,
  next_model_probability:next,delta:p===null||next===null?null:next-p};
}
function drawWholeBoard(){
 const f=F[selectedColumn],state=f[$('stage').value==='before'?'before':'after'];
 $('stepSlider').value=selectedColumn;$('frameLabel').textContent=columnLabel(f)+' · '+$('stage').value+' candidate state';
 const blanks=D.cells.filter(a=>!a.clue).map(a=>a.cell),n=blanks.length;
 const alive=blanks.filter(c=>f.after[c][D.reference[c]]).length,singletons=blanks.filter(c=>digits(f.after[c]).length===1).length;
 const changedCells=blanks.filter(c=>f.before[c].some((v,d)=>v!==f.after[c][d])).length;
 $('frameSummary').textContent=(f.kind==='initial'?'Initialized state: ':f.kind==='decode'?'Decoding (candidate mask unchanged): ':'After transition: ')+'reference present in '+alive+'/'+n+' blank cells; '+singletons+' singletons; '+changedCells+' blank cells changed.';
 const root=$('candidateBoard');root.replaceChildren();root.style.gridTemplateColumns='repeat('+side+',1fr)';root.style.minWidth=(side*90+2*(side-1))+'px';root.style.maxWidth=(side*106+2*(side-1))+'px';const box=Math.sqrt(side);
 state.forEach((mask,c)=>{
  const tile=document.createElement('div');tile.className='candidate-cell';
  if(D.question[c]>=0)tile.classList.add('clue');if(peer(c)&&c!==selected)tile.classList.add('peer');if(c===selected)tile.classList.add('selected');tile.classList.add(mask[D.reference[c]]?'reference-alive':'lost');
  if((c%side+1)%box===0&&c%side<side-1)tile.style.borderRight='3px solid #334155';if((Math.floor(c/side)+1)%box===0&&Math.floor(c/side)<side-1)tile.style.borderBottom='3px solid #334155';
  const caption=document.createElement('div');caption.className='cell-caption';caption.textContent=cellName(c)+(D.question[c]>=0?' · clue':' · n='+digits(mask).length);tile.appendChild(caption);
  const grid=document.createElement('div');grid.className='candidate-digits';const remaining=digits(mask);grid.style.gridTemplateColumns='repeat('+Math.max(1,Math.min(box,Math.ceil(Math.sqrt(remaining.length))))+',1fr)';
  if(remaining.length===1)tile.classList.add('singleton');
  remaining.forEach(digit=>{const d=digit-1,b=document.createElement('button');b.className='candidate-digit';b.textContent=digit;
   const p=f[$('mode').value]?.[c][d];if(p!==undefined){b.style.background='rgb('+Math.round(255-235*p)+','+Math.round(255-170*p)+','+Math.round(255-58*p)+')';b.style.color=p>.65?'white':'#172033'}
   if(d===D.reference[c])b.classList.add('reference');if(f.kind==='decode'&&d===D.prediction[c])b.classList.add('decoded');
   b.title=cellName(c)+' digit '+digit;b.onclick=e=>{e.stopPropagation();selectCell(c,d)};grid.appendChild(b)});
  tile.appendChild(grid);const delta=comparisonData(selectedColumn,c),footer=document.createElement('div');footer.className='cell-caption';footer.textContent='ref '+(D.reference[c]+1)+(delta.removed.length?' · −'+delta.removed.length:'')+(delta.added.length?' · +'+delta.added.length:'')+(cellEvents(f,c).length?' ★':'');tile.appendChild(footer);tile.onclick=()=>boardCell(c);root.appendChild(tile);
 });
 $('compareScope').textContent=$('compareTarget').value==='selected'?'Target: selected digit '+(selectedDigit+1)+' in every cell.':'Target: each cell’s own reference digit.';
 const tbody=$('comparisonRows');tbody.replaceChildren();
 D.cells.forEach(cell=>{const c=cell.cell,a=comparisonData(selectedColumn,c);if($('changedOnly').checked&&!a.removed.length&&!a.added.length&&!cellEvents(f,c).length)return;
  const row=document.createElement('tr');if(c===selected)row.className='selected';else if(peer(c))row.className='peer';
  const values=[cellName(c)+(cell.clue?' [clue]':''),a.target,a.before.join(','),a.after.join(','),'-['+a.removed.join(',')+'] / +['+a.added.join(',')+']',probabilityText(a.model_probability),probabilityText(a.retention_probability),probabilityText(a.next_model_probability),probabilityText(a.delta)];
  values.forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i>=5)td.title=String([a.model_probability,a.retention_probability,a.next_model_probability,a.delta][i-5]??'Not applicable');if(i===8&&a.delta!==null)td.className=a.delta<0?'danger':'good';row.appendChild(td)});row.onclick=()=>boardCell(c);tbody.appendChild(row);
 });
}
function chooseRun(index){if(!RUNS[index])return;D={...DATA,...RUNS[index]};F=D.frames;$('seedSelect').value=String(index);$('individualTrajectory').open=true;$('metadata').textContent=JSON.stringify({...D.metadata,selected_seed:D.seed,scores:D.scores},null,2);render();inspect(selectedColumn,selected,selectedDigit)}
function averageInspectionData(fi,c,d){
 const f=A.frames[fi],p=f.model_probability?.[c][d]??null,next=A.frames[fi+1]?.model_probability?.[c][d]??null;
 return {cell:cellName(c),digit:d+1,step:f.step,time_before:f.time,time_after:f.earlier_time,
  mean_model_probability:p,next_mean_model_probability:next,delta_mean_model_probability:p===null||next===null?null:next-p,
  mean_retention_probability:f.retention_probability?.[c][d]??null,
  candidate_frequency_before:f.before_frequency[c][d],candidate_frequency_after:f.after_frequency[c][d],
  mean_candidate_count_before:f.before_frequency[c].reduce((a,b)=>a+b,0),mean_candidate_count_after:f.mean_candidate_count[c],
  reference_survival_frequency:f.reference_survival[c],singleton_frequency:f.singleton_frequency[c],
  final_decoded_frequency:A.prediction_frequency[c][d]};
}
function inspectAverage(fi,c,d){$('averageInspection').textContent=JSON.stringify(averageInspectionData(fi,c,d),null,2)}
function paintAverage(ctx,x,y,c,d,f,compact){
 const p=f[$('averageMode').value]?.[c][d],available=f.before_frequency[c][d];
 ctx.fillStyle=p===undefined?'#e2e8f0':available===0?'#ededed':'rgb('+Math.round(255-235*p)+','+Math.round(255-170*p)+','+Math.round(255-58*p)+')';ctx.fillRect(x,y,width,rowHeight);
 const bold=compact?f.singleton_frequency[c]===1:f.after_frequency[c][d]===1;ctx.strokeStyle=bold?'#172033':'#cbd5e1';ctx.lineWidth=bold?2:.5;ctx.strokeRect(x+1,y+1,width-2,rowHeight-2);
 if(available===0&&p!==undefined){ctx.strokeStyle='#a1a1aa';ctx.beginPath();for(let i=0;i<width+rowHeight;i+=6){ctx.moveTo(x+Math.max(0,i-rowHeight),y+Math.min(i,rowHeight));ctx.lineTo(x+Math.min(i,width),y+Math.max(0,i-width))}ctx.stroke()}
 if(compact&&width>=12){ctx.fillStyle=p!==undefined&&p>.65&&available>0?'white':'#172033';ctx.fillText(f.mean_candidate_count[c].toFixed(1),x+2,y+17)}
}
function moveAverageCursors(){['averageTimelineCursor','averageOverviewCursor','averageExpandedCursor'].forEach(id=>{$(id).style.left=(labelWidth+averageColumn*width)+'px';$(id).style.width=width+'px'})}
function drawAverageBoard(){
 const f=A.frames[averageColumn],state=f[$('averageStage').value||'after_frequency'],root=$('averageBoard'),box=Math.sqrt(side);
 $('averageSlider').value=averageColumn;$('averageFrameLabel').textContent=columnLabel(f)+' · '+($('averageStage').value==='before_frequency'?'before':'after')+' candidate frequencies';
 const blanks=DATA.question.flatMap((v,c)=>v<0?[c]:[]);
 const n=blanks.length,meanCount=n?blanks.reduce((sum,c)=>sum+f.mean_candidate_count[c],0)/n:0,ref=n?blanks.reduce((sum,c)=>sum+f.reference_survival[c],0)/n:1;
 $('averageFrameSummary').textContent='After column: mean candidates per blank cell '+probabilityText(meanCount)+'; mean reference-cell survival '+probabilityText(ref)+'; reference intact across all blanks in '+probabilityText(f.reference_board_survival)+' of seeds.';
 root.replaceChildren();root.style.gridTemplateColumns='repeat('+side+',1fr)';root.style.minWidth=(side*90+2*(side-1))+'px';root.style.maxWidth=(side*106+2*(side-1))+'px';
 state.forEach((frequency,c)=>{
  const tile=document.createElement('div');tile.className='candidate-cell';if(DATA.question[c]>=0)tile.classList.add('clue');if(c===selected)tile.classList.add('selected');tile.classList.add(frequency[DATA.reference[c]]===1?'reference-alive':frequency[DATA.reference[c]]===0?'lost':'partial');
  if((c%side+1)%box===0&&c%side<side-1)tile.style.borderRight='3px solid #334155';if((Math.floor(c/side)+1)%box===0&&Math.floor(c/side)<side-1)tile.style.borderBottom='3px solid #334155';
  const caption=document.createElement('div');caption.className='cell-caption';caption.textContent=cellName(c)+' · mean n='+probabilityText(frequency.reduce((a,b)=>a+b,0));tile.appendChild(caption);
  const grid=document.createElement('div');grid.className='candidate-digits';grid.style.gridTemplateColumns='repeat('+box+',1fr)';
  frequency.forEach((value,d)=>{if(value===0)return;const b=document.createElement('button');b.className='candidate-digit';b.textContent=d+1;
   const small=document.createElement('small');small.style.display='block';small.style.fontSize='9px';small.textContent=Math.round(value*RUNS.length)+'/'+RUNS.length;b.appendChild(small);
   const p=f[$('averageMode').value]?.[c][d];if(p!==undefined){b.style.background='rgb('+Math.round(255-235*p)+','+Math.round(255-170*p)+','+Math.round(255-58*p)+')';b.style.color=p>.65?'white':'#172033'}
   if(d===DATA.reference[c])b.classList.add('reference');b.title='Candidate frequency '+value+'; mean probability '+(p??'not applicable');b.onclick=e=>{e.stopPropagation();selectAverageCell(c,d)};grid.appendChild(b)});
  tile.appendChild(grid);const footer=document.createElement('div');footer.className='cell-caption';footer.textContent='ref '+(DATA.reference[c]+1)+': '+Math.round(frequency[DATA.reference[c]]*RUNS.length)+'/'+RUNS.length;tile.appendChild(footer);tile.onclick=()=>selectAverageCell(c,DATA.reference[c]);root.appendChild(tile);
 });
}
function drawAverage(){
 if(!A)return;drawTimeline('averageTimeline',A.frames);
 const cells=DATA.question.flatMap((v,c)=>v<0?[c]:[]),ctx=prepare($('averageOverview'),Math.max(30,cells.length*rowHeight));
 cells.forEach((c,r)=>{const y=r*rowHeight;ctx.fillStyle=c===selected?'#fff0c2':'#f1f5f9';ctx.fillRect(0,y,labelWidth,rowHeight);ctx.fillStyle='#172033';ctx.fillText(cellName(c)+' ref '+(DATA.reference[c]+1),5,y+17);columns.forEach((fi,i)=>paintAverage(ctx,labelWidth+i*width,y,c,DATA.reference[c],A.frames[fi],true));if(c===selected)highlightRow(ctx,$('averageOverview'),y)});
 const expanded=prepare($('averageExpanded'),side*rowHeight);$('averageExpandedTitle').textContent='Mean candidate probabilities: '+cellName(selected);
 for(let d=0;d<side;d++){const y=d*rowHeight;expanded.fillStyle=d===selectedDigit?'#fff0c2':d===DATA.reference[selected]?'#fef3c7':'#f1f5f9';expanded.fillRect(0,y,labelWidth,rowHeight);expanded.fillStyle='#172033';expanded.fillText((d===DATA.reference[selected]?'★ ref ':'digit ')+(d+1),5,y+17);columns.forEach((fi,i)=>paintAverage(expanded,labelWidth+i*width,y,selected,d,A.frames[fi],false));if(d===selectedDigit)highlightRow(expanded,$('averageExpanded'),y)}
 moveAverageCursors();drawAverageBoard();
}
function selectAverageTime(fi){if(!A)return;averageColumn=Math.max(0,Math.min(A.frames.length-1,fi));moveAverageCursors();drawAverageBoard();inspectAverage(averageColumn,selected,selectedDigit)}
function selectAverageCell(c,d){selected=c;selectedDigit=d;$('averagePin').checked=true;render();inspectAverage(averageColumn,c,d)}
function render(){rows=D.cells.filter(a=>($('clues').checked||!a.clue)&&matches(a.cell)).map(a=>a.cell);width=Math.max(4,Math.min(24,Math.floor(24000/columns.length)));$('modeNote').textContent=$('mode').value==='model_probability'?'Color is the model probability of the reference digit in the overview, or the row’s digit in the expanded view. Dark blue = high predicted probability; gray hatching = excluded by the mask.':'Color is the chance that this candidate survives this one transition. Dark blue early is expected for nearly all digits because s is high; it does NOT indicate a selected answer or model confidence. Retention is undefined at initialization and decoding.';drawBoards();drawTimeline();drawOverview();drawExpanded();moveCursors();drawWholeBoard();showEvents(selectedColumn);drawAverage();if(A)inspectAverage(averageColumn,selected,selectedDigit)}
function hit(canvas,event){const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height;return {fi:columns[Math.floor((x-labelWidth)/width)],row:Math.floor(y/rowHeight)}}
['overview','expanded'].forEach(id=>{const canvas=$(id);canvas.onmousemove=e=>{if($('pin').checked)return;const h=hit(canvas,e),c=id==='overview'?rows[h.row]:selected,d=id==='overview'?D.reference[c]:h.row;if(h.fi!==undefined&&c!==undefined&&d>=0&&d<side)inspect(h.fi,c,d)};canvas.onclick=e=>{const h=hit(canvas,e);if(h.fi===undefined)return;const c=id==='overview'?rows[h.row]:selected,d=id==='overview'?D.reference[c]:h.row;if(c===undefined||d<0||d>=side)return;selectedColumn=h.fi;selectCell(c,d)}});
$('timeline').onclick=e=>{const h=hit($('timeline'),e);if(h.fi!==undefined)selectTime(h.fi)};
$('stepSlider').oninput=()=>selectTime(Number($('stepSlider').value));
$('previousStep').onclick=()=>selectTime(selectedColumn-1);$('nextStep').onclick=()=>selectTime(selectedColumn+1);
['mode','clues','filter'].forEach(id=>$(id).addEventListener('change',()=>{render();inspect(selectedColumn,selected,selectedDigit)}));
['stage','compareTarget','changedOnly'].forEach(id=>$(id).addEventListener('change',()=>{drawWholeBoard();inspect(selectedColumn,selected,selectedDigit)}));
let syncing=false;['timelineScroll','overviewScroll','expandedScroll'].forEach(id=>$(id).addEventListener('scroll',()=>{if(syncing)return;syncing=true;['timelineScroll','overviewScroll','expandedScroll'].filter(other=>other!==id).forEach(other=>$(other).scrollLeft=$(id).scrollLeft);syncing=false}));
const stickyPanels=[['expandedPanel','expandedSticky'],['averageExpandedPanel','averageSticky']];
function toggleSticky(panelId,buttonId){const active=$(buttonId).getAttribute('aria-pressed')!=='true';stickyPanels.forEach(([panel,button])=>{const on=active&&panel===panelId;$(panel).classList.toggle('is-sticky',on);$(button).setAttribute('aria-pressed',String(on));$(button).setAttribute('aria-label',on?'Unpin expanded view':'Pin expanded view');$(button).title=on?'Unpin expanded view':'Pin expanded view while scrolling'})}
stickyPanels.forEach(([panel,button])=>$(button).onclick=()=>toggleSticky(panel,button));
$('seedSelect').addEventListener('change',()=>chooseRun(Number($('seedSelect').value)));
$('averageMode').value='model_probability';$('averageStage').value='after_frequency';
$('averageMode').addEventListener('change',()=>{drawAverage();inspectAverage(averageColumn,selected,selectedDigit)});
$('averageStage').addEventListener('change',()=>{if(A)drawAverageBoard()});
$('averageSlider').oninput=()=>selectAverageTime(Number($('averageSlider').value));
$('averageTimeline').onclick=e=>{const h=hit($('averageTimeline'),e);if(h.fi!==undefined)selectAverageTime(h.fi)};
['averageOverview','averageExpanded'].forEach(id=>{const canvas=$(id);const box=e=>{const h=hit(canvas,e),cells=DATA.question.flatMap((v,c)=>v<0?[c]:[]),c=id==='averageOverview'?cells[h.row]:selected,d=id==='averageOverview'?DATA.reference[c]:h.row;return {...h,c,d}};canvas.onmousemove=e=>{if(!A||$('averagePin').checked)return;const h=box(e);if(h.fi!==undefined&&h.c!==undefined&&h.d>=0&&h.d<side)inspectAverage(h.fi,h.c,h.d)};canvas.onclick=e=>{if(!A)return;const h=box(e);if(h.fi===undefined||h.c===undefined||h.d<0||h.d>=side)return;averageColumn=h.fi;selectAverageCell(h.c,h.d)}});
let averageSyncing=false;['averageTimelineScroll','averageOverviewScroll','averageExpandedScroll'].forEach(id=>$(id).addEventListener('scroll',()=>{if(averageSyncing)return;averageSyncing=true;['averageTimelineScroll','averageOverviewScroll','averageExpandedScroll'].filter(other=>other!==id).forEach(other=>$(other).scrollLeft=$(id).scrollLeft);averageSyncing=false}));
render();inspect(0,selected,selectedDigit);
</script></body></html>"""

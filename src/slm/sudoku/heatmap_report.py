from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from slm.common.contracts import Report
from slm.common.seeds import SAMPLE_SEED
from slm.slm.core import predict_log_probs, repair_empty, reverse_probs

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


def sample_trajectory(model, question, steps, seed=SAMPLE_SEED):
    """Record one trajectory with identical RNG draws and rules to solve_sudoku."""
    if question.ndim != 1 or question.numel() not in (16, 81):
        raise ValueError("Expected one 4x4 or 9x9 question")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    questions = question[None]
    generator = torch.Generator(device=question.device).manual_seed(seed)
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
            for step in range(1, steps + 1):
                time = torch.full(
                    (1,),
                    (steps - step + 1) / steps,
                    dtype=torch.float64,
                    device=question.device,
                )
                earlier = torch.full_like(time, (steps - step) / steps)
                clean = (
                    predict_log_probs(model, candidates, time.float()).double().exp()
                )
                retention = reverse_probs(clean, candidates, earlier, time)
                draws = torch.rand(
                    retention.shape,
                    dtype=torch.float64,
                    device=question.device,
                    generator=generator,
                )
                sampled = draws < retention
                repaired = repair_empty(sampled, retention)
                after = clamp_clues(repaired, questions)
                events = []
                for kind, before_mask, after_mask in (
                    ("empty_set_repair", sampled, repaired),
                    ("clue_enforcement", repaired, after),
                ):
                    changes = _changes(before_mask, after_mask)
                    if changes:
                        events.append({"kind": kind, "changes": changes})
                frames.append(
                    {
                        "kind": "transition",
                        "step": step,
                        "time": time.item(),
                        "earlier_time": earlier.item(),
                        "before": candidates[0].cpu().tolist(),
                        "model_probability": clean[0].cpu().tolist(),
                        "retention_probability": retention[0].cpu().tolist(),
                        "draws": draws[0].cpu().tolist(),
                        "sampled": sampled[0].cpu().tolist(),
                        "repaired": repaired[0].cpu().tolist(),
                        "after": after[0].cpu().tolist(),
                        "events": events,
                    }
                )
                candidates = after
            log_probs = predict_log_probs(
                model, candidates, torch.zeros(1, device=question.device)
            )
            predicted = torch.where(questions >= 0, questions, log_probs.argmax(-1))
            final_mask = candidates[0].cpu().tolist()
            frames.append(
                {
                    "kind": "decode",
                    "step": steps + 1,
                    "time": 0.0,
                    "earlier_time": None,
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


class HeatmapReport(Report):
    def run(self, args: Namespace) -> dict[str, Any]:
        board_id = getattr(args, "board_id", 0)
        if board_id < 0:
            raise ValueError("board-id must be a nonnegative test-split index")
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
        trajectory = sample_trajectory(model, question.to(args.device), steps, seed)
        q, ref = question.tolist(), answer.tolist()
        cells = analyse_trajectory(trajectory, q, ref)
        prediction = torch.tensor(trajectory["prediction"], dtype=torch.long)
        scores = {
            key: value.item()
            for key, value in sudoku_scores(
                prediction[None], question[None], answer[None]
            ).items()
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
            **trajectory,
            "cells": cells,
            "scores": scores,
        }
        report = {"metadata": metadata, "scores": scores, "html": "trajectory.html"}
        report_path, tracker = _start_report(
            args, {**vars(config), **metadata}, report, tracking
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

    for key, value in data["metadata"].items():
        tracker.run.summary[f"heatmap/{key}"] = value
    tracker.log(
        completed,
        {f"report/heatmap/{key}": value for key, value in data["scores"].items()},
    )
    steps = [frame["step"] for frame in data["frames"]]
    blanks = [i for i, digit in enumerate(data["question"]) if digit < 0]
    survival = [
        sum(frame["after"][i][data["reference"][i]] for i in blanks) / len(blanks)
        if blanks
        else 1.0
        for frame in data["frames"]
    ]
    counts = [
        sum(sum(frame["after"][i]) for i in blanks) / len(blanks) if blanks else 0.0
        for frame in data["frames"]
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
    kinds = ("clue_enforcement", "constraint_filtering", "empty_set_repair", "other")
    tracker.artifact(
        "heatmap/interventions",
        wandb.plot.line_series(
            xs=steps,
            ys=[
                [
                    sum(
                        len(event["changes"])
                        for event in frame["events"]
                        if event["kind"] == kind
                    )
                    for frame in data["frames"]
                ]
                for kind in kinds
            ],
            keys=list(kinds),
            title="Cells modified by interventions",
            xname="sampling column (0=initial, N+1=decode)",
        ),
    )
    side = data["vocab"]
    events = [
        [
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
        for frame in data["frames"]
        for event in frame["events"]
        for change in event["changes"]
    ]
    tracker.artifact(
        "heatmap/events",
        wandb.Table(
            columns=[
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
            cell["cell"],
            cell["cell"] // side + 1,
            cell["cell"] % side + 1,
            cell["reference"] + 1,
            data["prediction"][cell["cell"]] + 1,
            cell["first_loss"],
            cell["wrong_singleton"],
            str(cell["interventions"]),
            cell["multiple_until_decode"],
        ]
        for cell in data["cells"]
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
    strings = [
        decode_board(torch.tensor(data[key], dtype=torch.long))
        for key in ("question", "reference", "prediction")
    ]
    tracker.artifact(
        "heatmap/board",
        wandb.Table(columns=["question", "reference", "prediction"], data=[strings]),
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
body{font:14px system-ui,sans-serif;margin:20px;color:#172033;background:#f8fafc}h1,h2{margin:12px 0}section{background:white;border:1px solid #d3dbe7;border-radius:6px;padding:14px;margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:12px}label{display:inline-block;margin:5px 15px 5px 0}select,input{padding:5px}.boards{display:flex;gap:30px;flex-wrap:wrap}.board{display:grid;gap:1px;background:#64748b;border:2px solid #172033}.board button{width:32px;height:32px;border:0;background:#fff;cursor:pointer;font:16px system-ui}.board .clue{font-weight:bold;background:#e2e8f0}.board .incorrect{color:#b91c1c}.board .selected{box-shadow:inset 0 0 0 3px #f59e0b}.scroll{overflow-x:auto;border:1px solid #cbd5e1;margin:8px 0}canvas{display:block}.legend{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.gradient{display:inline-block;width:130px;height:14px;background:linear-gradient(to right,#fff,#1455c5);border:1px solid #888}.masked{background:repeating-linear-gradient(45deg,#eee 0px,#eee 3px,#aaa 3px,#aaa 4px);padding:3px}.hint{color:#475569}.two{display:grid;grid-template-columns:1fr 1fr;gap:15px}@media(max-width:750px){.two{grid-template-columns:1fr}}
</style></head><body>
<h1>Sudoku sampling trajectory</h1>
<section><h2>Identity and outcome</h2><pre id="metadata"></pre><div id="boards" class="boards"></div><p class="hint">Red prediction digits disagree with the reference. For an ambiguous puzzle, another valid completion may still solve the board. Reference-loss diagnostics do not prove that all valid completions were eliminated.</p></section>
<section><h2>Views and failure filters</h2>
<label>Time view <select id="view"><option value="full">Full: every transition</option><option value="event">Event: changes, interventions and snapshots</option></select></label>
<label>Snapshot interval <input id="interval" type="number" min="1" value="10" style="width:65px"></label>
<label>Probability <select id="mode"><option value="model_probability">Model digit probability (masked, normalized)</option><option value="retention_probability">Sampler retention probability (Bernoulli inclusion)</option></select></label>
<label><input id="clues" type="checkbox">Show fixed clues</label>
<label>Cells <select id="filter"><option value="all">All</option><option value="lost">Lost reference digit</option><option value="wrong">Incorrect singleton</option><option value="intervention">Required intervention</option><option value="multiple">Multiple candidates until decoding</option></select></label>
<p class="hint" id="viewNote"></p><p><b>Timing:</b> color = probability <b>before</b> pruning; borders/counts = candidates <b>after</b> sampling, repair and clue enforcement. Markers describe changes during that column. Initial state is separate; final decoding chooses a digit without changing the candidate set.</p>
<div class="legend"><span>0% <span class="gradient"></span> 100% (fixed scale)</span><span class="masked">Already eliminated: zero forced by mask</span><span style="color:#dc2626">▲ first reference loss</span><span>▣ singleton / surviving digit border</span><span style="color:#b45309">★ intervention</span><span>— probability not applicable</span><span>✓ final selected digit; compact ✓/× = reference match/mismatch</span></div>
<p id="modeNote" class="hint"></p>
</section>
<section><h2>Event strip and actual step/timestep labels</h2><p>Click an intervention marker for affected cells/digits. Blue dots denote ordinary candidate changes, not enforcement. Initial clue enforcement is recorded at initialization. Constraint filtering and other rules are not implemented/enabled in this sampler; unchanged clue clamps are not change events.</p><div id="eventScroll" class="scroll"><canvas id="events"></canvas></div></section>
<section><h2>Compact overview: reference-digit probability</h2><p>One row per originally blank cell, in board-position order. Numbers are candidate counts after the column; hover/click for before/after counts and exact probabilities. Hatching marks masked zeros after elimination.</p><div id="overviewScroll" class="scroll"><canvas id="overview"></canvas></div><p id="empty"></p></section>
<section><h2 id="expandedTitle">Expanded candidate view</h2><p>One row per digit. <b>Bold borders</b> mean the digit survives the transition. The reference row is labeled ★. Click a Sudoku cell, overview row or digit box to link the views.</p><div id="expandedScroll" class="scroll"><canvas id="expanded"></canvas></div></section>
<div class="two"><section><h2>Box inspection</h2><pre id="inspection">Select or hover over a box.</pre></section><section><h2>Intervention details</h2><div id="eventCells"></div><pre id="eventDetails">Select an event marker.</pre></section></div>
<script id="trajectory" type="application/json">__TRAJECTORY_DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('trajectory').textContent), F=D.frames, side=D.vocab;
const $=id=>document.getElementById(id), cellName=c=>'r'+(Math.floor(c/side)+1)+'c'+(c%side+1), digits=m=>m.flatMap((v,i)=>v?[i+1]:[]);
let selected=D.question.findIndex(v=>v<0), selectedColumn=0, selectedDigit=0, columns=[], rows=[], width=20;
if(selected<0)selected=0;selectedDigit=D.reference[selected];
const labelWidth=140, rowHeight=24, kinds=['clue_enforcement','constraint_filtering','empty_set_repair','other'];
const eventColors=['#16a34a','#7c3aed','#e8790c','#be185d'];
$('metadata').textContent=JSON.stringify({...D.metadata,scores:D.scores},null,2);
$('interval').value=Math.max(1,Math.floor(D.metadata.sampling_budget/20));
function matches(c){const a=D.cells[c], f=$('filter').value;return f==='all'||f==='lost'&&a.first_loss!==null||f==='wrong'&&a.wrong_singleton!==null||f==='intervention'&&a.interventions.length>0||f==='multiple'&&a.multiple_until_decode}
function changed(f){return f.events.length>0||f.before.some((m,c)=>m.some((v,d)=>v!==f.after[c][d]||v!==f.sampled[c][d]))}
function columnLabel(f){return f.kind==='initial'?'initial t=1':f.kind==='decode'?'decode t=0':'s'+f.step+' t='+f.time.toFixed(5)}
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
function inspect(fi,c,d){const f=F[fi], p=f.model_probability?.[c][d], r=f.retention_probability?.[c][d];
 $('inspection').textContent=JSON.stringify({cell:cellName(c),digit:d+1,reference:D.reference[c]+1,column:fi,kind:f.kind,step:f.step,time_before:f.time,time_after:f.earlier_time,model_digit_probability:p??'not applicable',model_percent:p===undefined?'not applicable':p*100,sampler_retention_probability:r??'not applicable',retention_percent:r===undefined?'not applicable':r*100,uniform_draw:f.draws?.[c][d]??'not applicable',candidate_before:f.before[c][d],after_sampling:f.sampled[c][d],after_repair:f.repaired[c][d],candidate_after:f.after[c][d],candidate_count_before:digits(f.before[c]).length,candidate_count_after:digits(f.after[c]).length,candidates_before:digits(f.before[c]),candidates_after:digits(f.after[c]),change_mechanism:mechanisms(f,c,d),interventions:cellEvents(f,c),reference_first_eliminated_column:D.cells[c].first_loss,final_digit:D.prediction[c]+1},null,2);
}
function boardCell(c){selected=c;selectedDigit=D.reference[c];if(D.question[c]>=0)$('clues').checked=true;if(!matches(c))$('filter').value='all';render();inspect(selectedColumn,c,selectedDigit)}
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
function highlight(ctx,height){const i=columns.indexOf(selectedColumn);if(i>=0){ctx.strokeStyle='#f59e0b';ctx.lineWidth=2;ctx.strokeRect(labelWidth+i*width+1,1,width-2,height-2)}}
function drawOverview(){const ctx=prepare($('overview'),Math.max(30,rows.length*rowHeight));rows.forEach((c,r)=>{const y=r*rowHeight;ctx.fillStyle=c===selected?'#fff0c2':'#f1f5f9';ctx.fillRect(0,y,labelWidth,rowHeight);ctx.fillStyle='#172033';ctx.fillText(cellName(c)+' ref '+(D.reference[c]+1)+(D.question[c]>=0?' [clue]':''),5,y+17);columns.forEach((fi,i)=>paint(ctx,labelWidth+i*width,y,c,D.reference[c],F[fi],true));if(c===selected){ctx.strokeStyle='#f59e0b';ctx.lineWidth=2;ctx.strokeRect(1,y+1,$('overview').width-2,rowHeight-2)}});highlight(ctx,$('overview').height);$('empty').textContent=rows.length?'':'No cells match this filter.'}
function drawExpanded(){const ctx=prepare($('expanded'),side*rowHeight);$('expandedTitle').textContent='Expanded '+cellName(selected)+' — reference digit '+(D.reference[selected]+1);for(let d=0;d<side;d++){const y=d*rowHeight;ctx.fillStyle=d===D.reference[selected]?'#fef3c7':'#f1f5f9';ctx.fillRect(0,y,labelWidth,rowHeight);ctx.fillStyle='#172033';ctx.fillText((d===D.reference[selected]?'★ reference ':'digit ')+(d+1),5,y+17);columns.forEach((fi,i)=>paint(ctx,labelWidth+i*width,y,selected,d,F[fi],false))}highlight(ctx,$('expanded').height)}
function drawEvents(){const ctx=prepare($('events'),240);kinds.forEach((kind,r)=>{ctx.fillStyle=eventColors[r];ctx.fillText(kind.replaceAll('_',' '),3,160+r*18)});columns.forEach((fi,i)=>{const x=labelWidth+i*width,f=F[fi];ctx.save();ctx.translate(x+width/2,150);ctx.rotate(-Math.PI/2);ctx.fillStyle='#172033';ctx.fillText(columnLabel(f),0,0);ctx.restore();kinds.forEach((kind,r)=>{if(f.events.some(e=>e.kind===kind)){ctx.fillStyle=eventColors[r];ctx.fillRect(x+2,152+r*18,Math.max(2,width-4),14)}});if(f.kind==='transition'&&changed(f)){ctx.fillStyle='#2563eb';ctx.beginPath();ctx.arc(x+width/2,232,2,0,Math.PI*2);ctx.fill()}});highlight(ctx,240)}
function showEvents(fi){const f=F[fi];$('eventDetails').textContent=JSON.stringify({column:fi,label:columnLabel(f),events:f.events,ordinary_sampling_is_not_an_intervention:true},null,2);const root=$('eventCells');root.replaceChildren();const cells=[...new Set(f.events.flatMap(e=>e.changes.map(a=>a.cell)))];cells.forEach(c=>{const b=document.createElement('button');b.textContent=cellName(c);b.onclick=()=>boardCell(c);root.appendChild(b)})}
function render(){const event=$('view').value==='event',interval=Math.max(1,parseInt($('interval').value)||1);columns=F.map((f,i)=>i).filter(i=>!event||i===0||i===F.length-1||F[i].step%interval===0||changed(F[i]));rows=D.cells.filter(a=>($('clues').checked||!a.clue)&&matches(a.cell)).map(a=>a.cell);width=Math.max(4,Math.min(24,Math.floor(24000/columns.length)));$('viewNote').textContent=event?'Event columns are equally spaced visually, but do NOT represent equal time intervals. Labels retain actual steps and timesteps.':'Every sampling transition is shown, preceded by initial state and followed by final decoding.';$('modeNote').textContent=$('mode').value==='model_probability'?'Model digit probability: masked and normalized across currently available digits. Hatching distinguishes mask-forced zeros from predictions before elimination.':'Sampler retention probability: independent Bernoulli inclusion probability for this transition; it is NOT a normalized distribution across digits. Not applicable at initialization or final decoding.';drawBoards();drawEvents();drawOverview();drawExpanded()}
function hit(canvas,event){const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*canvas.width/rect.width,y=(event.clientY-rect.top)*canvas.height/rect.height;return {fi:columns[Math.floor((x-labelWidth)/width)],row:Math.floor(y/rowHeight)}}
['overview','expanded'].forEach(id=>{const canvas=$(id);canvas.onmousemove=e=>{const h=hit(canvas,e),c=id==='overview'?rows[h.row]:selected,d=id==='overview'?D.reference[c]:h.row;if(h.fi!==undefined&&c!==undefined&&d>=0&&d<side)inspect(h.fi,c,d)};canvas.onclick=e=>{const h=hit(canvas,e);if(h.fi===undefined)return;const c=id==='overview'?rows[h.row]:selected,d=id==='overview'?D.reference[c]:h.row;if(c===undefined||d<0||d>=side)return;selected=c;selectedDigit=d;selectedColumn=h.fi;render();inspect(h.fi,c,d);showEvents(h.fi)}});
$('events').onclick=e=>{const h=hit($('events'),e);if(h.fi===undefined)return;selectedColumn=h.fi;render();showEvents(h.fi);inspect(h.fi,selected,selectedDigit)};
['view','interval','mode','clues','filter'].forEach(id=>$(id).addEventListener('change',()=>{render();inspect(selectedColumn,selected,selectedDigit)}));
let syncing=false;['eventScroll','overviewScroll','expandedScroll'].forEach(id=>$(id).addEventListener('scroll',()=>{if(syncing)return;syncing=true;['eventScroll','overviewScroll','expandedScroll'].filter(other=>other!==id).forEach(other=>$(other).scrollLeft=$(id).scrollLeft);syncing=false}));
render();inspect(0,selected,selectedDigit);showEvents(0);
</script></body></html>"""

from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from tqdm.auto import tqdm

from slm.common.checkpoint import CheckpointManager, load_checkpoint
from slm.common.contracts import Report
from slm.common.reporting import (
    log_seed_distribution,
    phase_seeds,
    sampling_settings,
    score_statistics,
)
from slm.common.seeds import SAMPLE_SEED, TEST_SEED, VALIDATION_SEED
from slm.common.tracking import ExperimentTracker
from slm.slm.backbone import BFN_DIT
from slm.slm.core import seed_all

from .data import ALPHABET, load_text8
from .evaluation import evaluate_stream, generate

SAMPLE_NFE = 1000


def average_evaluations(results):
    if len(results) == 1:
        return results[0]
    count = len(results)
    mean = sum(result["bpc"] for result in results) / count
    combined = {
        **results[0],
        "bpc": mean,
        # Independent repeats include both window selection and MC randomness.
        "mc_se": math.sqrt(
            sum((result["bpc"] - mean) ** 2 for result in results)
            / (count * (count - 1))
        ),
    }
    if "window_seed" in combined:
        combined["window_seed_list"] = [result["window_seed"] for result in results]
    if "draw_scores" in combined:
        combined["draw_scores"] = [
            sum(result["draw_scores"][i] for result in results) / count
            for i in range(len(combined["draw_scores"]))
        ]
    return combined


class DefaultReport(Report):
    def run(self, args: Namespace) -> dict[str, Any]:
        from .run import model_config

        sampling = sampling_settings(args, TEST_SEED)
        validation_seeds = phase_seeds(args, sampling, VALIDATION_SEED)
        sample_seeds = phase_seeds(args, sampling, SAMPLE_SEED)
        config = model_config(args.config)
        settings = args.config.text8
        run_path = Path(args.run_dir)
        device = torch.device(args.device)
        _, valid, test, fingerprint = load_text8(args.data_dir)
        checkpoints = CheckpointManager(run_path, vars(config), fingerprint)
        state = load_checkpoint(run_path / "best.pt")
        checkpoints.validate(state)
        model = BFN_DIT(config).to(device).eval()
        model.load_state_dict(state["ema"]["model"])
        completed = state["progress"]["completed_updates"]
        tracking = json.loads((run_path / "manifest.json").read_text())["wandb"]
        draws, windows = settings["draws"], settings["windows"]

        by_seed = []
        seeds = sampling["sampling_seed_list"]
        for seed, validation_seed in tqdm(
            list(zip(seeds, validation_seeds)),
            desc="Text8 evaluation seeds",
            unit="seed",
            disable=len(seeds) < 2,
        ):
            by_seed.append(
                {
                    "seed": seed,
                    "validation_seed": validation_seed,
                    "validation": evaluate_stream(
                        model,
                        valid,
                        config,
                        windows or len(valid) // config.length,
                        draws,
                        seed=validation_seed,
                        progress=True,
                    ),
                    "test": evaluate_stream(
                        model,
                        test,
                        config,
                        windows or len(test) // config.length,
                        draws,
                        seed=seed,
                        progress=True,
                    ),
                }
            )
        validation = average_evaluations([run["validation"] for run in by_seed])
        testing = average_evaluations([run["test"] for run in by_seed])
        seed_scores = [
            {"test_bpc": run["test"]["bpc"], "validation_bpc": run["validation"]["bpc"]}
            for run in by_seed
        ]
        report = {
            "score_statistics": score_statistics(seed_scores),
            "checkpoint_step": completed,
            **sampling,
            "validation_seed_list": validation_seeds,
            "sample_seed_list": sample_seeds,
            "objective": config.objective,
            "config": vars(config),
            "data_fingerprint": fingerprint,
            "validation": validation,
            "test": testing,
        }
        if len(by_seed) > 1:
            report["by_seed"] = by_seed
        (run_path / "final_report.json").write_text(json.dumps(report, indent=2))
        report_path = run_path / "report" / uuid4().hex
        report_path.mkdir(parents=True)
        (report_path / "final_report.json").write_text(json.dumps(report, indent=2))
        tracker = ExperimentTracker(
            report_path,
            vars(args.config),
            name=args.name or tracking.get("name") or config.objective,
            mode=args.wandb_mode or tracking["mode"],
            project=getattr(args.config, "project", None)
            or f"{args.project or tracking['project']}-report",
            entity=args.entity if args.entity is not None else tracking.get("entity"),
            resume=False,
        )
        try:
            tracker.log(
                completed,
                {
                    "report/test_bpc": testing["bpc"],
                    "report/test_mc_se": testing["mc_se"],
                    "report/validation_bpc": validation["bpc"],
                },
            )
            tracker.run.summary["final_test_bpc"] = testing["bpc"]
            tracker.run.summary["final_test_mc_se"] = testing["mc_se"]
            tracker.run.summary["selected_checkpoint_step"] = completed
            log_seed_distribution(tracker, "report/bpc", seeds, seed_scores)
            texts, sample_rows = [], []
            for seed in tqdm(
                sample_seeds,
                desc="Text8 sample seeds",
                unit="seed",
                disable=len(sample_seeds) < 2,
            ):
                seed_all(seed)
                tokens = generate(
                    model,
                    4,
                    config.length,
                    config.vocab,
                    steps=SAMPLE_NFE - 1,
                    device=device,
                    sampler=sampling["sampler"],
                )
                for row in tokens.cpu().tolist():
                    text = "".join(ALPHABET[token] for token in row)
                    texts.append(
                        f"seed {seed}\n{text}" if len(sample_seeds) > 1 else text
                    )
                    sample_rows.append(
                        [seed, text] if len(sample_seeds) > 1 else [text]
                    )
            (run_path / "samples.txt").write_text("\n\n".join(texts))
            (report_path / "samples.txt").write_text("\n\n".join(texts))
            import wandb

            tracker.artifact(
                "samples/text8",
                wandb.Table(
                    columns=(["seed"] if len(sample_seeds) > 1 else []) + ["text"],
                    data=sample_rows,
                ),
            )
        except BaseException:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish()
        print(json.dumps(report, indent=2))
        return report

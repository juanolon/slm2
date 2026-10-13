from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from slm.common.checkpoint import CheckpointManager, load_checkpoint
from slm.common.contracts import Report
from slm.common.seeds import SAMPLE_SEED, TEST_SEED, VALIDATION_SEED
from slm.common.tracking import ExperimentTracker
from slm.slm.backbone import BFN_DIT
from slm.slm.core import seed_all

from .data import ALPHABET, load_text8
from .evaluation import evaluate_stream, generate

SAMPLE_NFE = 1000


class DefaultReport(Report):
    def run(self, args: Namespace) -> dict[str, Any]:
        from .run import model_config

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

        validation = evaluate_stream(
            model,
            valid,
            config,
            windows or len(valid) // config.length,
            draws,
            seed=VALIDATION_SEED,
        )
        testing = evaluate_stream(
            model,
            test,
            config,
            windows or len(test) // config.length,
            draws,
            seed=TEST_SEED,
        )
        report = {
            "checkpoint_step": completed,
            "objective": config.objective,
            "config": vars(config),
            "data_fingerprint": fingerprint,
            "validation": validation,
            "test": testing,
        }
        (run_path / "final_report.json").write_text(json.dumps(report, indent=2))
        report_path = run_path / "report" / uuid4().hex
        report_path.mkdir(parents=True)
        (report_path / "final_report.json").write_text(json.dumps(report, indent=2))
        tracker = ExperimentTracker(
            report_path,
            vars(config),
            name=args.name or tracking.get("name") or config.objective,
            mode=args.wandb_mode or tracking["mode"],
            project=f"{args.project or tracking['project']}-report",
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
            seed_all(SAMPLE_SEED)
            tokens = generate(
                model,
                4,
                config.length,
                config.vocab,
                steps=SAMPLE_NFE - 1,
                device=device,
            )
            texts = [
                "".join(ALPHABET[token] for token in row)
                for row in tokens.cpu().tolist()
            ]
            (run_path / "samples.txt").write_text("\n\n".join(texts))
            (report_path / "samples.txt").write_text("\n\n".join(texts))
            import wandb

            tracker.artifact(
                "samples/text8",
                wandb.Table(columns=["text"], data=[[text] for text in texts]),
            )
        except BaseException:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish()
        print(json.dumps(report, indent=2))
        return report

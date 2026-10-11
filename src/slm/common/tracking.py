from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class NullTracker:
    def __init__(self, *args, **kwargs):
        self.identity = {}

    def log(self, optimizer_step, metrics):
        pass

    def artifact(self, name, value):
        pass

    def finish(self, exit_code=0):
        pass


class ExperimentTracker:
    def __init__(
        self,
        run_dir: str | Path,
        config: Mapping[str, Any],
        *,
        name: str,
        mode: str,
        project: str,
        entity: str | None = None,
        resume: bool = False,
    ):
        self._initialize(run_dir, config, name, mode, project, entity, resume)

    def _initialize(self, run_dir, config, name, mode, project, entity, resume):
        import wandb

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        identity_path = self.run_dir / "wandb_identity.json"
        previous = (
            json.loads(identity_path.read_text()) if identity_path.exists() else None
        )
        if previous is not None and not resume:
            raise FileExistsError(
                "This directory already has tracking state; use --resume or a new directory"
            )
        experiment_id = previous["experiment_id"] if previous else uuid.uuid4().hex[:12]
        reuse_online = (
            resume and mode == "online" and previous and previous["mode"] == "online"
        )
        run_id = previous["id"] if reuse_online else uuid.uuid4().hex[:12]
        if reuse_online and (
            project != previous["project"]
            or (entity and entity != previous.get("entity"))
        ):
            raise ValueError(
                "Resuming an online run requires its original W&B project/entity"
            )
        self.run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            group=experiment_id,
            name=name,
            config=dict(config),
            dir=str(self.run_dir),
            mode=mode,
            resume="allow" if reuse_online else None,
            save_code=False,
        )
        self.closed = False
        try:
            self.run.define_metric("optimizer_step")
            for prefix in ("train/*", "validation/*", "systemlog/*", "report/*"):
                self.run.define_metric(prefix, step_metric="optimizer_step")
            self.identity = {
                "id": run_id,
                "project": project,
                "entity": getattr(self.run, "entity", None) or entity,
                "url": getattr(self.run, "url", None),
                "mode": mode,
                "experiment_id": experiment_id,
                "name": name,
            }
            identity_path.write_text(json.dumps(self.identity, indent=2))
        except BaseException:
            self.finish(exit_code=1)
            raise

    def log(self, optimizer_step: int, metrics: Mapping[str, Any]) -> None:
        record = {"optimizer_step": optimizer_step, **metrics}
        with (self.run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(
                json.dumps({**record, "run_segment": self.identity["id"]}) + "\n"
            )
        # No W&B step= argument: recovery can roll back past already-synced history.
        self.run.log(record)

    def artifact(self, name: str, value: Any) -> None:
        """Log a preconstructed chart/table; experiments supply scientific content."""
        self.run.log({name: value})

    def finish(self, exit_code: int = 0) -> None:
        if not self.closed:
            self.run.finish(exit_code=exit_code)
            self.closed = True

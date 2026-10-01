from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentRecord:
    run_id: str
    model_id: str
    task: str
    seed: int
    metrics: dict[str, float]
    parameters: dict[str, str | int | float | bool]
    dataset_fingerprint: str
    code_revision: str
    created_at: str


def new_experiment(
    *,
    run_id: str,
    model_id: str,
    task: str,
    seed: int,
    metrics: dict[str, float],
    parameters: dict[str, str | int | float | bool],
    dataset_fingerprint: str,
    code_revision: str,
) -> ExperimentRecord:
    if not run_id.strip() or not model_id.strip() or not task.strip():
        raise ValueError("run_id, model_id, and task must not be blank")

    return ExperimentRecord(
        run_id=run_id,
        model_id=model_id,
        task=task,
        seed=seed,
        metrics=dict(metrics),
        parameters=dict(parameters),
        dataset_fingerprint=dataset_fingerprint,
        code_revision=code_revision,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_experiment(record: ExperimentRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

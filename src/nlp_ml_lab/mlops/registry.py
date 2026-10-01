from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    model_name: str
    version: str
    artifact_path: str
    task: str
    dataset_fingerprint: str
    code_revision: str
    metrics: dict[str, float]
    stage: str = "candidate"


class ModelRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def register(self, record: ModelRecord) -> None:
        records = self._read()
        records.append(asdict(record))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def list(self) -> list[ModelRecord]:
        return [ModelRecord(**record) for record in self._read()]

    def promote(
        self,
        *,
        model_name: str,
        version: str,
        stage: str,
    ) -> ModelRecord:
        valid_stages = {"candidate", "staging", "production", "archived"}
        if stage not in valid_stages:
            raise ValueError(f"invalid stage: {stage}")

        records = self.list()
        target = next(
            (
                record
                for record in records
                if record.model_name == model_name and record.version == version
            ),
            None,
        )
        if target is None:
            raise KeyError(f"model not found: {model_name}:{version}")

        updated = [
            ModelRecord(**{
                **asdict(record),
                "stage": stage if record is target else (
                    "candidate" if record.model_name == model_name
                    and record.stage == "production"
                    and stage == "production"
                    else record.stage
                ),
            })
            for record in records
        ]
        self.path.write_text(
            json.dumps([asdict(record) for record in updated], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return next(
            record for record in updated
            if record.model_name == model_name and record.version == version
        )

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError("model registry file must contain a list")
        return payload

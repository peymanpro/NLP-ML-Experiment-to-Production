from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data: Path
    cache: Path
    processed_data: Path
    models: Path
    artifacts: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        return cls(
            root=root,
            raw_data=root / "data" / "raw",
            cache=root / "data" / "cache",
            processed_data=root / "data" / "processed",
            models=root / "models",
            artifacts=root / "artifacts",
            reports=root / "reports",
        )

    def create_runtime_directories(self) -> None:
        for path in (
            self.raw_data,
            self.cache,
            self.processed_data,
            self.models,
            self.artifacts,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)

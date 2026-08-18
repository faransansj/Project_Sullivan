"""Reproducibility metadata for research runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass
class ReproducibilityMetadata:
    git_sha: str
    config_hash: str
    split_manifest_hash: str
    normalization_artifact_hash: str
    dataset_version: str
    target_representation: str
    model_parameter_count: int
    seed: int
    checkpoint_selection_rule: str
    status: str
    git_dirty: bool
    resolved_config: dict
    schema_version: str = "1.0"
    training_started_at: Optional[str] = None
    training_ended_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"planned", "running", "completed", "failed", "aborted"}:
            raise ValueError(f"Invalid run status: {self.status}")
        if self.model_parameter_count < 0:
            raise ValueError("model_parameter_count cannot be negative")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

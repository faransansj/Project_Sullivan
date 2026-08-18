"""Versioned, speaker-disjoint dataset split manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class SampleAssignment:
    sample_id: str
    speaker_id: str
    utterance_id: str
    assignment: str

    def __post_init__(self) -> None:
        if self.assignment not in VALID_SPLITS:
            raise ValueError(f"Invalid split assignment: {self.assignment}")
        if not self.sample_id or not self.speaker_id or not self.utterance_id:
            raise ValueError("sample_id, speaker_id, and utterance_id are required")


@dataclass
class SplitManifest:
    dataset_version: str
    seed: int
    strategy: str
    git_sha: str
    samples: List[SampleAssignment]
    schema_version: str = "1.0"

    def validate(self) -> None:
        if not self.dataset_version or not self.strategy or not self.git_sha:
            raise ValueError("dataset_version, strategy, and git_sha are required")
        sample_ids = [sample.sample_id for sample in self.samples]
        duplicates = sorted(
            sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate sample IDs: {duplicates}")

        present_splits = {sample.assignment for sample in self.samples}
        missing_splits = sorted(VALID_SPLITS - present_splits)
        if missing_splits:
            raise ValueError(f"Manifest is missing required splits: {missing_splits}")

        utterance_splits = {}
        speaker_splits = {}
        for sample in self.samples:
            utterance_splits.setdefault(
                (sample.speaker_id, sample.utterance_id), set()
            ).add(sample.assignment)
            speaker_splits.setdefault(sample.speaker_id, set()).add(sample.assignment)
        utterance_overlap = {
            f"{speaker}/{utterance}": sorted(splits)
            for (speaker, utterance), splits in utterance_splits.items()
            if len(splits) > 1
        }
        if utterance_overlap:
            raise ValueError(f"Utterance overlap across splits: {utterance_overlap}")
        overlap = {
            speaker: sorted(splits) for speaker, splits in speaker_splits.items() if len(splits) > 1
        }
        if overlap:
            raise ValueError(f"Speaker overlap across splits: {overlap}")

    def to_dict(self) -> dict:
        self.validate()
        result = asdict(self)
        result["validation"] = {
            "duplicate_sample_ids": [],
            "utterance_overlap": [],
            "speaker_overlap": [],
            "split_counts": {
                split: sum(sample.assignment == split for sample in self.samples)
                for split in sorted(VALID_SPLITS)
            },
            "passed": True,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SplitManifest":
        manifest = cls(
            schema_version=data.get("schema_version", "1.0"),
            dataset_version=data["dataset_version"],
            seed=int(data["seed"]),
            strategy=data["strategy"],
            git_sha=data["git_sha"],
            samples=[SampleAssignment(**sample) for sample in data["samples"]],
        )
        manifest.validate()
        return manifest

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    samples: Iterable[SampleAssignment],
    *,
    dataset_version: str,
    seed: int,
    strategy: str,
    git_sha: str,
) -> SplitManifest:
    manifest = SplitManifest(dataset_version, seed, strategy, git_sha, list(samples))
    manifest.validate()
    return manifest

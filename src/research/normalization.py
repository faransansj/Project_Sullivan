"""Train-only normalization artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class NormalizationArtifact:
    feature_name: str
    mean: list
    std: list
    fit_split: str
    frame_count: int
    epsilon: float
    dtype: str
    config_hash: str
    manifest_hash: str
    dataset_version: str
    schema_version: str = "1.0"

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        feature_name: str,
        fit_split: str,
        config: dict,
        manifest_hash: str,
        dataset_version: str,
        mask: Optional[np.ndarray] = None,
        epsilon: float = 1e-8,
    ) -> "NormalizationArtifact":
        if fit_split != "train":
            raise ValueError("Normalization statistics may only be fit on the train split")
        if not manifest_hash or not dataset_version:
            raise ValueError("manifest_hash and dataset_version are required")
        values = np.asarray(values)
        if values.ndim < 2:
            raise ValueError("values must have a final feature dimension")
        flattened = values.reshape(-1, values.shape[-1])
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != values.shape[:-1]:
                raise ValueError("mask must match values excluding the feature dimension")
            flattened = flattened[mask.reshape(-1)]
        if not len(flattened):
            raise ValueError("Cannot fit normalization on zero valid frames")
        mean = flattened.mean(axis=0)
        std = flattened.std(axis=0)
        std = np.where(std < epsilon, 1.0, std)
        return cls(
            feature_name=feature_name,
            mean=mean.tolist(),
            std=std.tolist(),
            fit_split="train",
            frame_count=len(flattened),
            epsilon=epsilon,
            dtype=str(values.dtype),
            config_hash=config_sha256(config),
            manifest_hash=manifest_hash,
            dataset_version=dataset_version,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        return (values - np.asarray(self.mean)) / np.asarray(self.std)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        return values * np.asarray(self.std) + np.asarray(self.mean)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NormalizationArtifact":
        artifact = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        if artifact.fit_split != "train":
            raise ValueError("Invalid normalization artifact: fit_split is not train")
        return artifact

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

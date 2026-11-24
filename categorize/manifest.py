"""Reproducibility metadata stored next to a trained category model."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_FILENAME = "model_manifest.json"
MODEL_VERSION = "1.1.0"


class ModelManifest(BaseModel):
    model_version: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    hyperparameters: dict[str, object] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    label_names: list[str] = Field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(output_dir: Path, manifest: ModelManifest) -> Path:
    path = output_dir / MANIFEST_FILENAME
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(output_dir: Path) -> ModelManifest | None:
    path = output_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    return ModelManifest.model_validate_json(path.read_text(encoding="utf-8"))

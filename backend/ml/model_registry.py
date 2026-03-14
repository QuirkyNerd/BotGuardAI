from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ModelMetadata:
    model_version: str
    model_path: str
    training_date: str
    feature_schema: List[str]
    performance_metrics: Dict[str, Any]


def _load_registry(registry_path: Path) -> List[ModelMetadata]:
    """
    Load registry from JSON file.

    Supports two formats:
    - Dict format (current): {"models": {"v1": {"model_path": ..., ...}}}
    - List format (legacy):  [{"model_version": "v1", "model_path": ..., ...}]
    """
    if not registry_path.exists():
        logger.warning("Registry file not found at {}", registry_path)
        return []

    with registry_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    models: List[ModelMetadata] = []

    if isinstance(raw, list):
        # Legacy list format
        for entry in raw:
            models.append(ModelMetadata(**entry))
    elif isinstance(raw, dict) and "models" in raw:
        # Current dict format: {"models": {"v1": {...}}}
        for version, entry in raw["models"].items():
            models.append(
                ModelMetadata(
                    model_version=version,
                    model_path=entry["model_path"],
                    training_date=entry.get("training_date", datetime.utcnow().isoformat()),
                    feature_schema=entry.get("feature_schema", []),
                    performance_metrics=entry.get("performance_metrics", {}),
                )
            )
    else:
        logger.error("Unrecognised registry format in {}", registry_path)

    # Sort by training_date ascending so that models[-1] is the newest.
    models.sort(key=lambda m: m.training_date)
    return models


def _save_registry(registry_path: Path, models: List[ModelMetadata]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist back in the dict format.
    registry: Dict[str, Any] = {"models": {}}
    for m in models:
        registry["models"][m.model_version] = {
            "model_path": m.model_path,
            "training_date": m.training_date,
            "feature_schema": m.feature_schema,
            "performance_metrics": m.performance_metrics,
        }
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def resolve_model_path(
    registry_path: Path,
    requested_version: Optional[str] = None,
) -> Path:
    """
    Resolve which model artifact to load based on registry metadata.

    If requested_version is "latest" or None, the newest (last) model is returned.
    The model_path stored in the registry is relative to the project root (cwd).
    """
    models = _load_registry(registry_path)
    if not models:
        raise RuntimeError(f"No models found in registry at {registry_path}")

    if not requested_version or requested_version == "latest":
        chosen = models[-1]
    else:
        chosen = next(
            (m for m in models if m.model_version == requested_version),
            None,
        )
        if chosen is None:
            raise RuntimeError(
                f"Requested model_version={requested_version} not found in registry"
            )

    logger.info(
        "Using model_version={} from registry (trained_at={})",
        chosen.model_version,
        chosen.training_date,
    )

    # Resolve relative to the current working directory (project root).
    model_path = Path(chosen.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    return model_path.resolve()

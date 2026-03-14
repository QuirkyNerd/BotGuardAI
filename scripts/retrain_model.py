from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from backend.database.session import SessionLocal
from backend.ml.model_registry import register_model
from backend.services.feature_store import export_feature_dataset
from scripts.train_model import FEATURE_NAMES, generate_dataset


def load_dataset_from_feature_store() -> Tuple[np.ndarray, np.ndarray]:
    db = SessionLocal()
    try:
        records = export_feature_dataset(db)
    finally:
        db.close()

    if not records:
        logger.warning("No feature store records found; falling back to synthetic data.")
        return generate_dataset()

    X = np.array([r["features"] for r in records], dtype=float)
    # Simple heuristic label for demo: high interaction density -> bot
    interaction_idx = FEATURE_NAMES.index("interaction_density")
    y = np.array([1 if row[interaction_idx] < 8.0 else 0 for row in X])
    return X, y


def retrain_and_register(artifact_path: Path, registry_path: Path, model_version: str) -> None:
    X, y = load_dataset_from_feature_store()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report_str = classification_report(y_test, y_pred, output_dict=False)
    logger.info("Retrain classification report:\n{}", report_str)

    report_dict: Dict[str, object] = classification_report(
        y_test, y_pred, output_dict=True
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, artifact_path)
    logger.info("Saved retrained model to {}", artifact_path)

    register_model(
        registry_path=registry_path,
        model_version=model_version,
        model_path=artifact_path,
        feature_schema=FEATURE_NAMES,
        performance_metrics=report_dict,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_path = project_root / "backend" / "ml" / "artifacts" / "human_bot_model_retrained.pkl"
    model_path_str = os.getenv("MODEL_PATH", str(default_path))
    artifact_path = Path(model_path_str)

    registry_default = project_root / "backend" / "ml" / "artifacts" / "model_registry.json"
    registry_path_str = os.getenv("MODEL_REGISTRY_PATH", str(registry_default))
    registry_path = Path(registry_path_str)

    model_version = os.getenv("MODEL_VERSION", "v2")

    logger.info("Retraining RandomForest model for BotGuard AI (version={})...", model_version)
    retrain_and_register(artifact_path, registry_path, model_version)


if __name__ == "__main__":
    main()


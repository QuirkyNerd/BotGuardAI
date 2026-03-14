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

from backend.ml.model_registry import register_model


FEATURE_NAMES = [
    "avg_mouse_speed",
    "mouse_accel_variance",
    "click_interval_mean",
    "click_interval_std",
    "typing_latency_variance",
    "scroll_speed_mean",
    "scroll_accel_mean",
    "interaction_density",
    "avg_idle_duration",
]


def _generate_human_sample(n: int) -> np.ndarray:
    """
    Generate synthetic feature vectors approximating human-like behavior.
    """
    rng = np.random.default_rng()

    avg_mouse_speed = rng.normal(loc=400, scale=120, size=n)
    mouse_accel_variance = rng.normal(loc=2500, scale=800, size=n)
    click_interval_mean = rng.normal(loc=0.7, scale=0.2, size=n)
    click_interval_std = rng.normal(loc=0.4, scale=0.15, size=n)
    typing_latency_variance = rng.normal(loc=0.2, scale=0.08, size=n)
    scroll_speed_mean = rng.normal(loc=600, scale=200, size=n)
    scroll_accel_mean = rng.normal(loc=3000, scale=1000, size=n)
    interaction_density = rng.normal(loc=4.0, scale=1.0, size=n)
    avg_idle_duration = rng.normal(loc=3.0, scale=1.0, size=n)

    return np.vstack(
        [
            avg_mouse_speed,
            mouse_accel_variance,
            click_interval_mean,
            click_interval_std,
            typing_latency_variance,
            scroll_speed_mean,
            scroll_accel_mean,
            interaction_density,
            avg_idle_duration,
        ]
    ).T


def _generate_bot_sample(n: int) -> np.ndarray:
    """
    Generate synthetic feature vectors approximating scripted/bot behavior.
    """
    rng = np.random.default_rng()

    avg_mouse_speed = rng.normal(loc=1000, scale=150, size=n)
    mouse_accel_variance = rng.normal(loc=500, scale=200, size=n)
    click_interval_mean = rng.normal(loc=0.2, scale=0.05, size=n)
    click_interval_std = rng.normal(loc=0.05, scale=0.02, size=n)
    typing_latency_variance = rng.normal(loc=0.02, scale=0.01, size=n)
    scroll_speed_mean = rng.normal(loc=2000, scale=300, size=n)
    scroll_accel_mean = rng.normal(loc=800, scale=200, size=n)
    interaction_density = rng.normal(loc=15.0, scale=2.0, size=n)
    avg_idle_duration = rng.normal(loc=0.3, scale=0.1, size=n)

    return np.vstack(
        [
            avg_mouse_speed,
            mouse_accel_variance,
            click_interval_mean,
            click_interval_std,
            typing_latency_variance,
            scroll_speed_mean,
            scroll_accel_mean,
            interaction_density,
            avg_idle_duration,
        ]
    ).T


def generate_dataset(n_humans: int = 2000, n_bots: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a combined synthetic dataset of human and bot behaviors.
    """
    human_features = _generate_human_sample(n_humans)
    bot_features = _generate_bot_sample(n_bots)

    X = np.vstack([bot_features, human_features])
    # Class ordering is [0=bot, 1=human]
    y = np.array([0] * n_bots + [1] * n_humans)
    return X, y


def train_and_save_model(artifact_path: Path, registry_path: Path, model_version: str) -> None:
    X, y = generate_dataset()

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
    logger.info("Classification report:\n{}", report_str)

    report_dict: Dict[str, object] = classification_report(
        y_test, y_pred, output_dict=True
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, artifact_path)
    logger.info("Saved trained model to {}", artifact_path)

    register_model(
        registry_path=registry_path,
        model_version=model_version,
        model_path=artifact_path,
        feature_schema=FEATURE_NAMES,
        performance_metrics=report_dict,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_path = project_root / "backend" / "ml" / "artifacts" / "human_bot_model.pkl"
    model_path_str = os.getenv("MODEL_PATH", str(default_path))
    artifact_path = Path(model_path_str)

    registry_default = project_root / "backend" / "ml" / "artifacts" / "model_registry.json"
    registry_path_str = os.getenv("MODEL_REGISTRY_PATH", str(registry_default))
    registry_path = Path(registry_path_str)

    model_version = os.getenv("MODEL_VERSION", "v1")

    logger.info("Training RandomForest model for BotGuard AI (version={})...", model_version)
    train_and_save_model(artifact_path, registry_path, model_version)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()


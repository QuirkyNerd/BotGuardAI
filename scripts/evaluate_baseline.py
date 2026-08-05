from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from backend.ml.evaluation import (
    calculate_evaluation_metrics,
    export_evaluation_artifact,
    generate_telemetry_derived_dataset,
)
from scripts.train_model import generate_dataset as generate_legacy_dataset


def run_legacy_baseline_evaluation(seed: int = 42) -> dict:
    """
    Evaluate the original Random Forest model using the LEGACY Gaussian synthetic feature dataset.
    """
    logger.info("Evaluating LEGACY Feature-Space Synthetic Baseline (n=4000)...")
    X, y = generate_legacy_dataset(n_humans=2000, n_bots=2000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # Predict human probability P(Y=1)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = calculate_evaluation_metrics(y_test, y_prob, threshold=0.5)

    model_info = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "class_weight": "balanced",
        "random_state": seed,
    }

    artifact_path = project_root / "backend" / "ml" / "artifacts" / "legacy_baseline_metrics.json"
    export_evaluation_artifact(
        artifact_path=artifact_path,
        dataset_name="Legacy Feature-Space Synthetic Baseline",
        dataset_description="Gaussian random feature generation directly constructing 9-feature vectors (train_model.py methodology)",
        metrics=metrics,
        model_info=model_info,
        seed=seed,
    )

    return metrics


def run_telemetry_baseline_evaluation(seed: int = 42) -> dict:
    """
    Evaluate the Random Forest model using the RUNTIME-FAITHFUL TELEMETRY-DERIVED simulated dataset.
    Raw events -> production feature_engineering.py -> 9-feature vectors -> Random Forest.
    """
    logger.info("Evaluating RUNTIME-FAITHFUL Telemetry-Derived Baseline (n=2000 raw event streams)...")
    X, y, session_ids = generate_telemetry_derived_dataset(n_humans=1000, n_bots=1000, seed=seed)

    # Session-level stratified train/test split to prevent leakage across sessions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = calculate_evaluation_metrics(y_test, y_prob, threshold=0.5)

    model_info = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "class_weight": "balanced",
        "random_state": seed,
        "feature_pipeline": "backend.services.feature_engineering.compute_features_from_batches",
    }

    artifact_path = project_root / "backend" / "ml" / "artifacts" / "telemetry_baseline_metrics.json"
    export_evaluation_artifact(
        artifact_path=artifact_path,
        dataset_name="Runtime-Faithful Telemetry-Derived Baseline",
        dataset_description="Simulated raw event streams (mouse, click, key, scroll) passed through production feature engineering",
        metrics=metrics,
        model_info=model_info,
        seed=seed,
    )

    return metrics


def print_comparison(legacy: dict, telemetry: dict) -> None:
    """
    Print a side-by-side technical comparison table between the two baselines.
    """
    print("\n" + "=" * 85)
    print("      BOTGUARD AI — BASELINE EVALUATION COMPARISON (STEP 1 AUDIT & BASELINE)")
    print("=" * 85)
    print(f"{'Metric':<35} | {'Legacy Synthetic Baseline':<22} | {'Telemetry-Derived Baseline':<22}")
    print("-" * 85)

    lm_std = legacy["standard_metrics"]
    tm_std = telemetry["standard_metrics"]
    lm_sec = legacy["security_metrics"]
    tm_sec = telemetry["security_metrics"]
    lm_cal = legacy["probabilistic_calibration_metrics"]
    tm_cal = telemetry["probabilistic_calibration_metrics"]

    rows = [
        ("Accuracy", f"{lm_std['accuracy']:.4f}", f"{tm_std['accuracy']:.4f}"),
        ("Precision (Human)", f"{lm_std['precision_human']:.4f}", f"{tm_std['precision_human']:.4f}"),
        ("Recall (Human)", f"{lm_std['recall_human']:.4f}", f"{tm_std['recall_human']:.4f}"),
        ("F1-Score (Human)", f"{lm_std['f1_human']:.4f}", f"{tm_std['f1_human']:.4f}"),
        ("ROC-AUC", f"{lm_std['roc_auc']:.4f}", f"{tm_std['roc_auc']:.4f}"),
        ("PR-AUC", f"{lm_std['pr_auc']:.4f}", f"{tm_std['pr_auc']:.4f}"),
        ("Brier Score (Calibration)", f"{lm_cal['brier_score']:.4f}", f"{tm_cal['brier_score']:.4f}"),
        ("Log Loss", f"{lm_cal['log_loss']:.4f}", f"{tm_cal['log_loss']:.4f}"),
        ("False Acceptance Rate (FAR)", f"{lm_sec['far_percentage']:.2f}%", f"{tm_sec['far_percentage']:.2f}%"),
        ("False Rejection Rate (FRR)", f"{lm_sec['frr_percentage']:.2f}%", f"{tm_sec['frr_percentage']:.2f}%"),
    ]

    for label, val_l, val_t in rows:
        print(f"{label:<35} | {val_l:<22} | {val_t:<22}")

    print("-" * 85)
    print("Class Semantics: Label 1 = Human (Positive), Label 0 = Bot (Negative)")
    print("Security Definitions:")
    print("  - FAR (False Acceptance Rate): Bots misclassified as Human (FP / Total Bots)")
    print("  - FRR (False Rejection Rate): Humans misclassified as Bot (FN / Total Humans)")
    print("=" * 85 + "\n")


def main() -> None:
    seed = 42
    logger.info("Starting Baseline Evaluation Suite (Seed={})...", seed)
    legacy_metrics = run_legacy_baseline_evaluation(seed=seed)
    telemetry_metrics = run_telemetry_baseline_evaluation(seed=seed)
    print_comparison(legacy_metrics, telemetry_metrics)


if __name__ == "__main__":
    main()

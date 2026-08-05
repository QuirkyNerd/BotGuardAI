from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split


# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import CALIBRATED_MODEL_PATH, CHALLENGE_THRESHOLD, ALLOW_THRESHOLD
from backend.ml.calibration import train_and_evaluate_calibration
from backend.ml.evaluation import generate_telemetry_derived_dataset, CLASS_SEMANTICS


def run_threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate FAR, FRR, Allow %, Challenge %, Block % across threshold sweeps.
    """
    sweep_results = []
    thresholds = [round(t, 2) for t in np.arange(0.05, 1.00, 0.05)]

    for t in thresholds:
        # Decision logic:
        # prob >= ALLOW_THRESHOLD (0.85) -> ALLOW
        # CHALLENGE_THRESHOLD (0.60) <= prob < ALLOW_THRESHOLD -> CHALLENGE
        # prob < CHALLENGE_THRESHOLD -> BLOCK
        y_pred_human = y_prob >= t
        total = len(y_true)
        bots = y_true == 0
        humans = y_true == 1

        far = float(np.sum((y_prob >= t) & bots) / np.sum(bots)) if np.sum(bots) > 0 else 0.0
        frr = float(np.sum((y_prob < t) & humans) / np.sum(humans)) if np.sum(humans) > 0 else 0.0

        sweep_results.append({
            "threshold": t,
            "far_percentage": round(far * 100.0, 2),
            "frr_percentage": round(frr * 100.0, 2),
        })

    # Proportional breakdown at current provisional thresholds (0.60 challenge, 0.85 allow)
    allow_count = int(np.sum(y_prob >= ALLOW_THRESHOLD))
    challenge_count = int(np.sum((y_prob >= CHALLENGE_THRESHOLD) & (y_prob < ALLOW_THRESHOLD)))
    block_count = int(np.sum(y_prob < CHALLENGE_THRESHOLD))
    total_samples = len(y_prob)

    provisional_breakdown = {
        "status": "PROVISIONAL / SIMULATION-DERIVED",
        "allow_threshold": ALLOW_THRESHOLD,
        "challenge_threshold": CHALLENGE_THRESHOLD,
        "allow_percentage": round((allow_count / total_samples) * 100.0, 2),
        "challenge_percentage": round((challenge_count / total_samples) * 100.0, 2),
        "block_percentage": round((block_count / total_samples) * 100.0, 2),
        "allow_count": allow_count,
        "challenge_count": challenge_count,
        "block_count": block_count,
    }

    return {
        "provisional_decision_breakdown": provisional_breakdown,
        "threshold_sweep": sweep_results,
    }


def generate_reliability_plot(results: Dict[str, Any], output_path: Path) -> None:
    """
    Generate and save reliability / calibration curve plot using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6), dpi=300)

        # Reference perfectly calibrated line
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)

        colors = {"uncalibrated": "#ef4444", "sigmoid": "#3b82f6", "isotonic": "#10b981"}
        markers = {"uncalibrated": "o", "sigmoid": "s", "isotonic": "^"}

        for name, data in results.items():
            bins = data.get("reliability_curve_bins", [])
            if not bins:
                continue
            prob_pred = [b["predicted_prob"] for b in bins]
            actual_freq = [b["actual_freq"] for b in bins]
            brier = data["probabilistic_calibration_metrics"]["brier_score"]

            plt.plot(
                prob_pred,
                actual_freq,
                marker=markers.get(name, "o"),
                color=colors.get(name, "blue"),
                linewidth=2,
                label=f"{name.capitalize()} (Brier: {brier:.5f})",
            )

        plt.xlabel("Mean Predicted Probability P(Y=Human)")
        plt.ylabel("Empirical Frequency of Human Class")
        plt.title("BotGuard AI — Calibration & Reliability Curves")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        plt.savefig(output_path)
        plt.close()
        logger.info("Saved calibration curve plot to {}", output_path)
    except Exception as exc:
        logger.warning("Could not generate calibration plot artifact: {}", exc)



def main() -> None:
    seed = 42
    logger.info("Starting Probability Calibration Analysis (Seed={})...", seed)

    # 1. Generate Telemetry-Derived Dataset (N=2000)
    X, y, session_ids = generate_telemetry_derived_dataset(n_humans=1000, n_bots=1000, seed=seed)

    # 2. Perform 3-Way Leakage-Free Split: 60% Train, 20% Calibration, 20% Test
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_cal, y_train_cal, test_size=0.25, random_state=seed, stratify=y_train_cal
    )

    logger.info(
        "Data Split -> Train: {}, Calibration: {}, Test: {}",
        len(X_train),
        len(X_calib),
        len(X_test),
    )

    # 3. Train and Evaluate Calibration Methods
    models, results = train_and_evaluate_calibration(
        X_train, y_train, X_calib, y_calib, X_test, y_test, seed=seed
    )

    # 4. Perform Threshold Analysis for best model
    threshold_analysis = run_threshold_analysis(y_test, models["sigmoid"].predict_proba_batch(X_test))

    # 5. Export JSON Artifact
    artifact_path = project_root / "backend" / "ml" / "artifacts" / "calibration_metrics.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_limitation_notice": "Current evaluation datasets are highly separable simulated streams. Brier scores near zero reflect feature distinctness rather than real-world calibration completeness.",
        "class_semantics": CLASS_SEMANTICS,
        "sample_counts": {
            "train": len(X_train),
            "calibration": len(X_calib),
            "test": len(X_test),
        },
        "methods_evaluated": list(results.keys()),
        "calibration_results": results,
        "provisional_threshold_analysis": threshold_analysis,
    }

    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved calibration metrics artifact to {}", artifact_path)

    # 6. Save Plot Artifact
    plot_path = project_root / "backend" / "ml" / "artifacts" / "calibration_curve.png"
    generate_reliability_plot(results, plot_path)

    # 7. Select & Save Production Calibrated Model Artifact (Sigmoid chosen for stability & low sample risk)
    best_model_wrapper = models["sigmoid"]
    CALIBRATED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_wrapper, CALIBRATED_MODEL_PATH)
    logger.info("Saved calibrated model artifact (Sigmoid Platt Scaling) to {}", CALIBRATED_MODEL_PATH)

    # 8. Console Output Table
    print("\n" + "=" * 90)
    print("      BOTGUARD AI — PROBABILITY CALIBRATION EVALUATION (STEP 2)")
    print("=" * 90)
    print(f"{'Method':<20} | {'Brier Score':<14} | {'Log Loss':<12} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'FAR (%)':<8} | {'FRR (%)':<8}")
    print("-" * 90)

    for name, res in results.items():
        cal = res["probabilistic_calibration_metrics"]
        std = res["standard_metrics"]
        sec = res["security_metrics"]
        print(
            f"{name.capitalize():<20} | {cal['brier_score']:<14.6f} | {cal['log_loss']:<12.6f} | "
            f"{std['roc_auc']:<10.4f} | {std['pr_auc']:<10.4f} | {sec['far_percentage']:<8.2f} | {sec['frr_percentage']:<8.2f}"
        )

    print("-" * 90)
    print("SELECTED CALIBRATION METHOD: Sigmoid (Platt Scaling)")
    print(f"CALIBRATED ARTIFACT SAVED TO: {CALIBRATED_MODEL_PATH}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

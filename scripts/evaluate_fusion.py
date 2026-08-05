from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import (
    COMPOSITE_ALLOW_RISK_THRESHOLD,
    COMPOSITE_CHALLENGE_RISK_THRESHOLD,
    PROVISIONAL_ANOMALY_THRESHOLD,
)
from backend.ml.evaluation import _generate_raw_human_batch
from backend.ml.intelligence_engine import MultiEnginePrediction, load_intelligence_stack, run_multi_engine_prediction
from backend.models.schemas import RiskLevel
from backend.security.risk_engine import RiskEvaluationResult, compute_risk_score_v2
from backend.services.decision_engine import evaluate_session
from backend.services.feature_engineering import compute_features_from_batches
from backend.simulation.adversarial_simulator import generate_adversarial_bot_session


def generate_fusion_benchmark_datasets(n_per_level: int = 200, seed: int = 42) -> Dict[str, List[Any]]:
    """
    Generate evaluation session batches for Human Control and Levels 1 to 5.
    """
    logger.info("Generating fusion benchmark session batches (N={}/group)...", n_per_level)
    rng = np.random.default_rng(seed + 888)
    datasets: Dict[str, List[Any]] = {}

    # Human Control
    h_batches = [_generate_raw_human_batch(rng, f"fusion_ctrl_human_{i:03d}") for i in range(n_per_level)]
    datasets["Human Control"] = h_batches

    # Levels 1 to 5
    for lvl in range(1, 6):
        b_batches = [generate_adversarial_bot_session(level=lvl, session_id=f"fusion_adv_l{lvl}_{i:03d}", seed=seed + 300 + i) for i in range(n_per_level)]
        datasets[f"Level {lvl}"] = b_batches

    return datasets


def evaluate_ablation_configurations(datasets: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Run ablation across 5 configurations:
      1. RF Only
      2. RF + Isolation Forest
      3. RF + Temporal 1D CNN
      4. RF + Isolation Forest + Temporal 1D CNN
      5. Full Risk Engine 2.0 (Integrated Multi-Engine + Security Indicators)
    """
    ablation_results: Dict[str, Dict[str, Any]] = {}
    config_names = [
        "1. RF Only",
        "2. RF + Isolation Forest",
        "3. RF + Temporal 1D CNN",
        "4. ML Fusion (RF+Anom+Temp)",
        "5. Full Risk Engine 2.0",
    ]

    for cfg in config_names:
        cfg_out: Dict[str, Any] = {}
        for grp_name, batches in datasets.items():
            is_bot = grp_name != "Human Control"
            allows, challenges, blocks = 0, 0, 0
            risk_scores: List[float] = []

            for batch in batches:
                fv = compute_features_from_batches(batch.session_id, [batch])
                pred = run_multi_engine_prediction(fv.values, batch=batch)

                if cfg == "1. RF Only":
                    risk = pred.rf_risk
                elif cfg == "2. RF + Isolation Forest":
                    risk = 0.55 * pred.rf_risk + 0.45 * pred.anomaly_risk
                elif cfg == "3. RF + Temporal 1D CNN":
                    risk = 0.50 * pred.rf_risk + 0.50 * pred.temporal_risk
                elif cfg == "4. ML Fusion (RF+Anom+Temp)":
                    risk = 0.35 * pred.rf_risk + 0.30 * pred.anomaly_risk + 0.35 * pred.temporal_risk
                else:  # Full Risk Engine 2.0
                    sec_flags = {"suspicious": batch.metadata.webdriver if batch.metadata else False}
                    eval_res = compute_risk_score_v2(pred, fv.values, batch.metadata, sec_flags)
                    risk = eval_res.composite_risk_score

                risk_scores.append(risk)

                if risk < COMPOSITE_ALLOW_RISK_THRESHOLD:
                    allows += 1
                elif risk < COMPOSITE_CHALLENGE_RISK_THRESHOLD:
                    challenges += 1
                else:
                    blocks += 1

            total_n = len(batches)
            if is_bot:
                det_rate = (challenges + blocks) / total_n
                far = allows / total_n
                frr = 0.0
            else:
                det_rate = 1.0
                far = 0.0
                frr = (challenges + blocks) / total_n  # Human False Rejection/Challenge Rate

            cfg_out[grp_name] = {
                "sample_count": total_n,
                "allow_pct": round((allows / total_n) * 100.0, 2),
                "challenge_pct": round((challenges / total_n) * 100.0, 2),
                "block_pct": round((blocks / total_n) * 100.0, 2),
                "detection_rate": round(float(det_rate), 4),
                "false_acceptance_rate_far": round(float(far), 4),
                "human_false_rejection_rate_frr": round(float(frr), 4),
                "mean_risk_score": round(float(np.mean(risk_scores)), 2),
                "median_risk_score": round(float(np.median(risk_scores)), 2),
            }

        ablation_results[cfg] = cfg_out

    return ablation_results


def benchmark_end_to_end_latency(datasets: Dict[str, List[Any]], num_samples: int = 100) -> Dict[str, float]:
    """
    Measure end-to-end inference latency including feature engineering, RF, Isolation Forest, Temporal 1D CNN, and Risk Engine 2.0.
    """
    latencies_ms: List[float] = []
    all_batches = datasets["Level 5"]

    for i in range(num_samples):
        batch = all_batches[i % len(all_batches)]
        t0 = time.perf_counter()
        fv = compute_features_from_batches(batch.session_id, [batch])
        _ = evaluate_session(
            session_id=batch.session_id,
            features=fv.values,
            browser_metadata=batch.metadata,
            batch=batch,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    return {
        "end_to_end_mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "end_to_end_p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
    }


def generate_fusion_visualizations(
    ablation_res: Dict[str, Dict[str, Any]],
    plot_path: Path,
) -> None:
    """
    Generate ablation plot showing bot detection across configurations.
    """
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
    configs = list(ablation_res.keys())

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(levels))
    width = 0.15

    colors = ["#ef4444", "#f97316", "#f59e0b", "#10b981", "#3b82f6"]

    for idx, cfg in enumerate(configs):
        det_rates = [ablation_res[cfg][lvl]["detection_rate"] * 100.0 for lvl in levels]
        ax.bar(x + (idx - 2) * width, det_rates, width, label=cfg, color=colors[idx])

    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Multi-Engine Fusion Ablation: Bot Detection Across Attack Levels")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved risk engine fusion ablation plot to {}", plot_path)


def main() -> None:
    logger.info("Starting Multi-Engine Intelligence Integration & Risk Engine 2.0 Evaluation...")

    # 1. Load Multi-Engine Stack
    load_intelligence_stack()

    # 2. Generate Benchmark Datasets
    datasets = generate_fusion_benchmark_datasets(n_per_level=200, seed=42)

    # 3. Evaluate Ablation Configurations
    ablation_res = evaluate_ablation_configurations(datasets)

    # 4. Measure End-to-End Latency
    latency_res = benchmark_end_to_end_latency(datasets, num_samples=100)

    # 5. Save Artifacts
    artifact_path = project_root / "backend" / "ml" / "artifacts" / "fusion_benchmark_metrics.json"
    plot_path = project_root / "backend" / "ml" / "artifacts" / "risk_engine_fusion_ablation.png"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provisional_thresholds": {
            "composite_allow_risk_threshold": COMPOSITE_ALLOW_RISK_THRESHOLD,
            "composite_challenge_risk_threshold": COMPOSITE_CHALLENGE_RISK_THRESHOLD,
            "provisional_anomaly_threshold": PROVISIONAL_ANOMALY_THRESHOLD,
        },
        "end_to_end_latency": latency_res,
        "ablation_results": ablation_res,
    }

    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved fusion metrics JSON artifact to {}", artifact_path)
    generate_fusion_visualizations(ablation_res, plot_path)

    # 6. Console Output Summary Table (Full Risk Engine 2.0)
    print("\n" + "=" * 125)
    print("      BOTGUARD AI — RISK ENGINE 2.0 INTEGRATED BENCHMARK EVALUATION (STEP 6)")
    print("=" * 125)
    print(f"{'Attack Level / Group':<20} | {'Action Breakdown (ALLOW / CHALLENGE / BLOCK)':<45} | {'FAR (%)':<10} | {'Mean Composite Risk':<20}")
    print("-" * 125)

    full_cfg = ablation_res["5. Full Risk Engine 2.0"]
    for grp in ["Human Control", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]:
        res = full_cfg[grp]
        breakdown_str = f"{res['allow_pct']:.1f}% / {res['challenge_pct']:.1f}% / {res['block_pct']:.1f}%"
        far_str = f"{res['false_acceptance_rate_far']*100:.1f}%" if grp != "Human Control" else "N/A (Control)"
        print(f"{grp:<20} | {breakdown_str:<45} | {far_str:<10} | {res['mean_risk_score']:<20.1f}")

    print("-" * 125)
    print("ABLATION & LATENCY SUMMARY:")
    print(f"  - End-to-End Mean Inference Latency: {latency_res['end_to_end_mean_latency_ms']:.3f} ms (P95: {latency_res['end_to_end_p95_latency_ms']:.3f} ms)")
    print(f"  - Human Control Rejection/Challenge Rate (FRR): {full_cfg['Human Control']['human_false_rejection_rate_frr']*100:.1f}% (ALLOW Rate = {full_cfg['Human Control']['allow_pct']:.1f}%)")
    print(f"  - Level 4 Bot Detection Rate: {full_cfg['Level 4']['detection_rate']*100:.1f}% | Level 5 Bot Detection Rate: {full_cfg['Level 5']['detection_rate']*100:.1f}%")
    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()

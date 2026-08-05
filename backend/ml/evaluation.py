from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.models.schemas import (
    BehaviorBatch,
    BrowserMetadata,
    ClickEvent,
    FocusEvent,
    KeyPressEvent,
    MouseEvent,
    MousePosition,
    ScrollEvent,
)
from backend.services.feature_engineering import FEATURE_NAMES, compute_features_from_batches


# Class definitions:
# Class 1 = Human (Positive Class)
# Class 0 = Bot   (Negative Class)
CLASS_SEMANTICS = {
    0: "Bot (Negative Class)",
    1: "Human (Positive Class)",
    "security_definitions": {
        "False Acceptance Rate (FAR)": "Proportion of Bots (Class 0) incorrectly accepted as Human (Class 1) = FP / (FP + TN)",
        "False Rejection Rate (FRR)": "Proportion of Humans (Class 1) incorrectly rejected as Bot (Class 0) = FN / (TP + FN)",
    },
}


def calculate_evaluation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute standard machine learning and security evaluation metrics given
    ground-truth binary labels y_true and predicted probabilities y_prob.

    :param y_true: Ground truth binary labels (0 = Bot, 1 = Human)
    :param y_prob: Predicted human probability (P(Y=1))
    :param threshold: Decision threshold for binary classification (default: 0.5)
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Standard ML metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    # PR-AUC (Precision-Recall Curve AUC for positive class)
    try:
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
        pr_auc = float(auc(recalls, precisions))
    except Exception:
        pr_auc = 0.0

    # Calibration & Probabilistic Loss
    brier = float(brier_score_loss(y_true, y_prob))
    try:
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
        loss_log = float(log_loss(y_true, y_prob_clipped))
    except Exception:
        loss_log = float("nan")

    # Security Metrics:
    # False Acceptance Rate (FAR): FP / (FP + TN) -> Bots predicted as Human
    # False Rejection Rate (FRR): FN / (TP + FN) -> Humans predicted as Bot
    total_bots = int(fp + tn)
    total_humans = int(tp + fn)

    far = float(fp / total_bots) if total_bots > 0 else 0.0
    frr = float(fn / total_humans) if total_humans > 0 else 0.0

    return {
        "decision_threshold": threshold,
        "sample_counts": {
            "total": len(y_true),
            "bots": total_bots,
            "humans": total_humans,
        },
        "confusion_matrix": {
            "tn_bots_correct": int(tn),
            "fp_bots_as_human": int(fp),
            "fn_humans_as_bot": int(fn),
            "tp_humans_correct": int(tp),
            "raw_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "standard_metrics": {
            "accuracy": acc,
            "precision_human": prec,
            "recall_human": rec,
            "f1_human": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
        "security_metrics": {
            "false_acceptance_rate_far": far,
            "false_rejection_rate_frr": frr,
            "far_percentage": round(far * 100.0, 2),
            "frr_percentage": round(frr * 100.0, 2),
        },
        "probabilistic_calibration_metrics": {
            "brier_score": brier,
            "log_loss": loss_log,
        },
    }


def generate_telemetry_derived_dataset(
    n_humans: int = 1000,
    n_bots: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate a runtime-faithful simulated dataset by creating raw behavioral event
    streams (mouse movements, clicks, typing, scrolling, idle periods) and passing
    them through the ACTUAL production feature engineering pipeline:
    `backend.services.feature_engineering.compute_features_from_batches()`.

    This prevents train-serve feature definition skew.

    :return: Tuple of (X_features, y_labels, session_ids)
    """
    rng = np.random.default_rng(seed)
    feature_vectors: List[List[float]] = []
    labels: List[int] = []
    session_ids: List[str] = []

    # 1. Generate Simulated Human Telemetry Streams
    for i in range(n_humans):
        session_id = f"sim_human_{i:04d}_{seed}"
        batch = _generate_raw_human_batch(rng, session_id)

        # Pass through ACTUAL production feature engineering pipeline
        fv = compute_features_from_batches(session_id, [batch])
        feature_vectors.append(fv.values)
        labels.append(1)  # 1 = Human
        session_ids.append(session_id)

    # 2. Generate Simulated Bot Telemetry Streams (diverse bot patterns)
    for i in range(n_bots):
        session_id = f"sim_bot_{i:04d}_{seed}"
        bot_subtype = rng.choice(["headless", "rapid_click", "zero_typing", "noisy_bot"])
        batch = _generate_raw_bot_batch(rng, session_id, bot_subtype)

        # Pass through ACTUAL production feature engineering pipeline
        fv = compute_features_from_batches(session_id, [batch])
        feature_vectors.append(fv.values)
        labels.append(0)  # 0 = Bot
        session_ids.append(session_id)

    X = np.array(feature_vectors, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y, session_ids


def _generate_raw_human_batch(rng: np.random.Generator, session_id: str) -> BehaviorBatch:
    """
    Generate a simulated raw event batch with human-like behavioral variability.
    """
    start_time = 1000.0
    t = start_time
    moves: List[MouseEvent] = []
    clicks: List[ClickEvent] = []
    scrolls: List[ScrollEvent] = []
    key_presses: List[KeyPressEvent] = []
    focus_events: List[FocusEvent] = [FocusEvent(timestamp=t, focused=True)]

    # Human mouse trajectory (curved, variable speed & acceleration)
    num_moves = rng.integers(15, 45)
    curr_x, curr_y = rng.uniform(100, 500), rng.uniform(100, 500)
    for _ in range(num_moves):
        dt = rng.uniform(15.0, 60.0)  # ms between moves
        t += dt
        dx = rng.normal(loc=12.0, scale=8.0)
        dy = rng.normal(loc=8.0, scale=6.0)
        curr_x += dx
        curr_y += dy
        moves.append(MouseEvent(timestamp=t, position=MousePosition(x=float(curr_x), y=float(curr_y))))

    # Human clicks (variable timing deltas)
    num_clicks = rng.integers(2, 6)
    for _ in range(num_clicks):
        t += float(max(50.0, rng.normal(loc=650.0, scale=200.0)))
        clicks.append(ClickEvent(timestamp=t, button="left"))

    # Human typing (variable latency)
    num_keys = rng.integers(8, 25)
    for _ in range(num_keys):
        t += float(max(20.0, rng.normal(loc=220.0, scale=70.0)))
        key_presses.append(KeyPressEvent(timestamp=t, key=str(rng.choice(list("abcdefghijklmnopqrstuvwxyz")))))

    # Human scroll
    num_scrolls = rng.integers(3, 10)
    scroll_y = 0.0
    for _ in range(num_scrolls):
        t += rng.uniform(30.0, 100.0)
        scroll_y += float(rng.normal(loc=150.0, scale=40.0))
        scrolls.append(ScrollEvent(timestamp=t, delta_y=float(scroll_y)))

    # Human idle gap
    if rng.random() > 0.4:
        t += rng.uniform(1200.0, 3500.0)  # 1.2s - 3.5s natural pause

    metadata = BrowserMetadata(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        language="en-US",
        platform="Win32",
        screen_width=1920,
        screen_height=1080,
        webgl_fingerprint="ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0)",
        canvas_fingerprint="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        device_entropy=float(rng.uniform(50000, 150000)),
        webdriver=False,
        touch_points=0,
    )

    return BehaviorBatch(
        session_id=session_id,
        started_at=start_time,
        ended_at=t,
        mouse_moves=moves,
        scrolls=scrolls,
        clicks=clicks,
        key_presses=key_presses,
        focus_events=focus_events,
        metadata=metadata,
    )


def _generate_raw_bot_batch(rng: np.random.Generator, session_id: str, bot_subtype: str) -> BehaviorBatch:
    """
    Generate simulated raw event batch representing different bot automation profiles.
    """
    start_time = 1000.0
    t = start_time
    moves: List[MouseEvent] = []
    clicks: List[ClickEvent] = []
    scrolls: List[ScrollEvent] = []
    key_presses: List[KeyPressEvent] = []
    focus_events: List[FocusEvent] = [FocusEvent(timestamp=t, focused=True)]

    if bot_subtype == "headless":
        # Straight line constant step
        num_moves = 30
        x, y = 100.0, 100.0
        for _ in range(num_moves):
            t += 10.0  # constant 10ms
            x += 25.0
            y += 1.0
            moves.append(MouseEvent(timestamp=t, position=MousePosition(x=x, y=y)))

        for _ in range(5):
            t += 100.0
            clicks.append(ClickEvent(timestamp=t, button="left"))

        for _ in range(15):
            t += 5.0
            key_presses.append(KeyPressEvent(timestamp=t, key="a"))

        is_webdriver = True
        ua = "HeadlessChrome/128.0.0.0"

    elif bot_subtype == "rapid_click":
        for _ in range(25):
            t += 15.0  # 15ms interval
            clicks.append(ClickEvent(timestamp=t, button="left"))
        is_webdriver = True
        ua = "Puppeteer/1.0.0"

    elif bot_subtype == "zero_typing":
        for _ in range(30):
            t += 3.0  # 3ms typing latency
            key_presses.append(KeyPressEvent(timestamp=t, key="x"))
        is_webdriver = True
        ua = "Selenium/4.0"

    else:  # noisy_bot (bot trying to add basic random jitter)
        num_moves = 35
        x, y = 100.0, 100.0
        for _ in range(num_moves):
            t += float(max(1.0, rng.normal(loc=12.0, scale=1.0)))
            x += float(rng.normal(loc=20.0, scale=2.0))
            y += float(rng.normal(loc=5.0, scale=0.5))
            moves.append(MouseEvent(timestamp=t, position=MousePosition(x=float(x), y=float(y))))

        for _ in range(6):
            t += float(max(10.0, rng.normal(loc=150.0, scale=10.0)))
            clicks.append(ClickEvent(timestamp=t, button="left"))

        for _ in range(12):
            t += float(max(2.0, rng.normal(loc=30.0, scale=3.0)))
            key_presses.append(KeyPressEvent(timestamp=t, key="b"))

        is_webdriver = bool(rng.choice([True, False]))
        ua = "Mozilla/5.0 (X11; Linux x86_64) BotSimulator/2.0"

    metadata = BrowserMetadata(
        user_agent=ua,
        language="en-US",
        platform="Linux x86_64",
        screen_width=1920,
        screen_height=1080,
        webgl_fingerprint="bot-gl",
        canvas_fingerprint="bot-canvas",
        device_entropy=1234.0,
        webdriver=is_webdriver,
        touch_points=0,
    )

    return BehaviorBatch(
        session_id=session_id,
        started_at=start_time,
        ended_at=t,
        mouse_moves=moves,
        scrolls=scrolls,
        clicks=clicks,
        key_presses=key_presses,
        focus_events=focus_events,
        metadata=metadata,
    )


def export_evaluation_artifact(
    artifact_path: Path,
    dataset_name: str,
    dataset_description: str,
    metrics: Dict[str, Any],
    model_info: Dict[str, Any],
    seed: int = 42,
) -> None:
    """
    Save evaluation results to a clean JSON artifact.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_description": dataset_description,
        "class_semantics": CLASS_SEMANTICS,
        "random_seed": seed,
        "feature_schema": FEATURE_NAMES,
        "model_info": model_info,
        "evaluation_results": metrics,
    }

    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved evaluation artifact for '{}' to {}", dataset_name, artifact_path)

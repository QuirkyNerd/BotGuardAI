from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from loguru import logger

from backend.models.schemas import BehaviorBatch


FEATURE_NAMES: List[str] = [
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


@dataclass
class FeatureVector:
    """
    Container for numerical features that are passed to the ML model.
    """

    values: List[float]

    def as_model_input(self) -> List[float]:
        return self.values


def _compute_mouse_features(batches: List[BehaviorBatch]) -> Dict[str, float]:
    timestamps: List[float] = []
    speeds: List[float] = []
    accels: List[float] = []

    for batch in batches:
        moves = sorted(batch.mouse_moves, key=lambda e: e.timestamp)
        for i in range(1, len(moves)):
            prev = moves[i - 1]
            curr = moves[i]
            dt = (curr.timestamp - prev.timestamp) / 1000.0  # seconds
            if dt <= 0:
                continue
            dx = curr.position.x - prev.position.x
            dy = curr.position.y - prev.position.y
            dist = float(np.hypot(dx, dy))
            speed = dist / dt
            speeds.append(speed)
            timestamps.append(curr.timestamp)

    for i in range(1, len(speeds)):
        dv = speeds[i] - speeds[i - 1]
        # Use average dt between timestamps if available, else 1s.
        dt = (
            (timestamps[i] - timestamps[i - 1]) / 1000.0
            if i < len(timestamps)
            else 1.0
        )
        if dt <= 0:
            continue
        accels.append(dv / dt)

    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    accel_var = float(np.var(accels)) if accels else 0.0

    return {
        "avg_mouse_speed": avg_speed,
        "mouse_accel_variance": accel_var,
    }


def _compute_click_features(batches: List[BehaviorBatch]) -> Dict[str, float]:
    click_times: List[float] = []
    for batch in batches:
        click_times.extend([c.timestamp for c in batch.clicks])

    click_times = sorted(click_times)
    if len(click_times) < 2:
        return {"click_interval_mean": 0.0, "click_interval_std": 0.0}

    intervals = np.diff(click_times) / 1000.0  # seconds
    return {
        "click_interval_mean": float(np.mean(intervals)),
        "click_interval_std": float(np.std(intervals)),
    }


def _compute_typing_features(batches: List[BehaviorBatch]) -> Dict[str, float]:
    key_times: List[float] = []
    for batch in batches:
        key_times.extend([k.timestamp for k in batch.key_presses])

    key_times = sorted(key_times)
    if len(key_times) < 2:
        return {"typing_latency_variance": 0.0}

    latencies = np.diff(key_times) / 1000.0
    return {"typing_latency_variance": float(np.var(latencies))}


def _compute_scroll_features(batches: List[BehaviorBatch]) -> Dict[str, float]:
    scroll_times: List[float] = []
    speeds: List[float] = []
    accels: List[float] = []

    for batch in batches:
        scrolls = sorted(batch.scrolls, key=lambda e: e.timestamp)
        for i in range(1, len(scrolls)):
            prev = scrolls[i - 1]
            curr = scrolls[i]
            dt = (curr.timestamp - prev.timestamp) / 1000.0
            if dt <= 0:
                continue
            speed = (curr.delta_y - prev.delta_y) / dt
            speeds.append(speed)
            scroll_times.append(curr.timestamp)

    for i in range(1, len(speeds)):
        dv = speeds[i] - speeds[i - 1]
        dt = (
            (scroll_times[i] - scroll_times[i - 1]) / 1000.0
            if i < len(scroll_times)
            else 1.0
        )
        if dt <= 0:
            continue
        accels.append(dv / dt)

    return {
        "scroll_speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "scroll_accel_mean": float(np.mean(accels)) if accels else 0.0,
    }


def _compute_session_level_features(batches: List[BehaviorBatch]) -> Dict[str, float]:
    if not batches:
        return {"interaction_density": 0.0, "avg_idle_duration": 0.0}

    first_ts = min(b.started_at for b in batches)
    last_ts = max(b.ended_at for b in batches)
    duration_sec = max((last_ts - first_ts) / 1000.0, 1.0)

    total_events = 0
    all_timestamps: List[float] = []
    for batch in batches:
        total_events += (
            len(batch.mouse_moves)
            + len(batch.scrolls)
            + len(batch.clicks)
            + len(batch.key_presses)
            + len(batch.focus_events)
        )
        for coll in (
            batch.mouse_moves,
            batch.scrolls,
            batch.clicks,
            batch.key_presses,
            batch.focus_events,
        ):
            all_timestamps.extend(e.timestamp for e in coll)

    interaction_density = total_events / duration_sec if duration_sec > 0 else 0.0

    # Idle duration as large gaps between consecutive events
    if len(all_timestamps) < 2:
        avg_idle = duration_sec
    else:
        sorted_ts = sorted(all_timestamps)
        gaps = np.diff(sorted_ts) / 1000.0
        idle_gaps = gaps[gaps > 1.0] if len(gaps) else []
        avg_idle = float(np.mean(idle_gaps)) if len(idle_gaps) else 0.0

    return {
        "interaction_density": float(interaction_density),
        "avg_idle_duration": avg_idle,
    }


def compute_features_from_batches(session_id: str, batches: List[BehaviorBatch]) -> FeatureVector:
    """
    Compute a fixed-order numerical feature vector for a session based on
    all behavior batches that have been recorded for that session.
    """
    mouse_features = _compute_mouse_features(batches)
    click_features = _compute_click_features(batches)
    typing_features = _compute_typing_features(batches)
    scroll_features = _compute_scroll_features(batches)
    session_features = _compute_session_level_features(batches)

    feature_map: Dict[str, float] = {
        **mouse_features,
        **click_features,
        **typing_features,
        **scroll_features,
        **session_features,
    }

    values = [float(feature_map.get(name, 0.0)) for name in FEATURE_NAMES]

    logger.debug("Computed features for session {}: {}", session_id, feature_map)
    return FeatureVector(values=values)

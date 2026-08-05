from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.models.schemas import BehaviorBatch, MouseEvent, ClickEvent, KeyPressEvent, ScrollEvent


def extract_raw_event_sequence(batch: BehaviorBatch, max_len: int = 60) -> np.ndarray:
    """
    Extract a normalized chronologically sorted 7-dimensional raw event sequence matrix (max_len, 7).
    Privacy guarantee: No typed character content is inspected or stored; only keypress timing is used.

    Feature Channels (7):
      0: dt_norm (normalized timestamp delta in seconds)
      1: is_mouse (1.0 if mouse move, else 0.0)
      2: is_click (1.0 if click, else 0.0)
      3: is_key (1.0 if keypress, else 0.0)
      4: is_scroll (1.0 if scroll, else 0.0)
      5: spatial_delta_norm (normalized distance/scroll delta)
      6: velocity_norm (normalized instantaneous velocity)
    """
    events: List[Tuple[float, str, Tuple[float, float]]] = []

    for m in batch.mouse_moves:
        events.append((m.timestamp, "mouse", (m.position.x, m.position.y)))
    for c in batch.clicks:
        events.append((c.timestamp, "click", (0.0, 0.0)))
    for k in batch.key_presses:
        # PRIVACY PRINCIPLE: Ignore key string content entirely, keep timestamp only
        events.append((k.timestamp, "key", (0.0, 0.0)))
    for s in batch.scrolls:
        events.append((s.timestamp, "scroll", (0.0, s.delta_y)))

    # Sort all events chronologically by timestamp
    events.sort(key=lambda x: x[0])

    seq: List[List[float]] = []
    prev_time = batch.started_at
    prev_pos = (0.0, 0.0)

    for ts, ev_type, (x, y) in events:
        dt = max(0.0, (ts - prev_time) / 1000.0)
        dt_norm = min(dt / 3.0, 1.0)  # Normalized to [0, 1]

        is_mouse = 1.0 if ev_type == "mouse" else 0.0
        is_click = 1.0 if ev_type == "click" else 0.0
        is_key = 1.0 if ev_type == "key" else 0.0
        is_scroll = 1.0 if ev_type == "scroll" else 0.0

        if ev_type == "mouse":
            dist = math.hypot(x - prev_pos[0], y - prev_pos[1])
            prev_pos = (x, y)
        elif ev_type == "scroll":
            dist = abs(y)
        else:
            dist = 0.0

        spatial_norm = min(dist / 500.0, 1.0)
        vel = dist / (dt + 1e-4)
        vel_norm = min(vel / 2000.0, 1.0)

        seq.append([dt_norm, is_mouse, is_click, is_key, is_scroll, spatial_norm, vel_norm])
        prev_time = ts

    # Handle padding / truncation to max_len
    matrix = np.zeros((max_len, 7), dtype=np.float32)
    if len(seq) > 0:
        arr = np.array(seq[:max_len], dtype=np.float32)
        matrix[:len(arr), :] = arr

    return matrix


class Temporal1DCNN(nn.Module):
    """
    Lightweight 1D Convolutional Neural Network for raw event sequence classification.
    Input Shape: (Batch, Channels=7, SeqLen=60)
    Target: 1 = Human, 0 = Bot
    """

    def __init__(self, in_channels: int = 7, seq_len: int = 60) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (B, 7, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # Shape: (B, 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return torch.sigmoid(logits)


class TemporalGRU(nn.Module):
    """
    Lightweight Gated Recurrent Unit (GRU) for event sequence classification.
    Input Shape: (Batch, SeqLen=60, Features=7)
    Target: 1 = Human, 0 = Bot
    """

    def __init__(self, input_size: int = 7, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (B, T, 7)
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # Shape: (B, 32)
        logits = self.fc(last_hidden)
        return torch.sigmoid(logits)

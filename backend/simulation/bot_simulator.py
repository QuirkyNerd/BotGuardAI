from __future__ import annotations

import random
from typing import List

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


def _base_metadata(bot_name: str) -> BrowserMetadata:
    return BrowserMetadata(
        user_agent=f"{bot_name}-bot/1.0",
        language="en-US",
        platform="Linux x86_64",
        screen_width=1920,
        screen_height=1080,
        webgl_fingerprint="bot-gl",
        canvas_fingerprint="bot-canvas",
        device_entropy=1234.0,
        webdriver=True,
        touch_points=0,
    )


def simulate_headless_bot(session_id: str) -> BehaviorBatch:
    start = 0.0
    moves: List[MouseEvent] = []
    clicks: List[ClickEvent] = []
    scrolls: List[ScrollEvent] = []
    keys: List[KeyPressEvent] = []

    # Straight-line mouse movement with constant step and tiny jitter.
    t = start
    x, y = 100.0, 100.0
    for _ in range(50):
        t += 10.0
        x += 20.0
        y += 0.5
        moves.append(MouseEvent(timestamp=t, position=MousePosition(x=x, y=y)))

    # Perfectly regular clicks.
    for i in range(10):
        t += 50.0
        clicks.append(ClickEvent(timestamp=t, button="left"))

    # Fast scroll.
    scroll_y = 0.0
    for _ in range(20):
        t += 15.0
        scroll_y += 200.0
        scrolls.append(ScrollEvent(timestamp=t, delta_y=scroll_y))

    # Near-zero latency typing.
    for i in range(20):
        t += 5.0
        keys.append(KeyPressEvent(timestamp=t, key="a"))

    focus_events = [FocusEvent(timestamp=0.0, focused=True)]

    return BehaviorBatch(
        session_id=session_id,
        started_at=start,
        ended_at=t,
        mouse_moves=moves,
        scrolls=scrolls,
        clicks=clicks,
        key_presses=keys,
        focus_events=focus_events,
        metadata=_base_metadata("headless"),
    )


def simulate_rapid_click_bot(session_id: str) -> BehaviorBatch:
    start = 0.0
    clicks: List[ClickEvent] = []
    t = start
    for _ in range(40):
        t += 20.0
        clicks.append(ClickEvent(timestamp=t, button="left"))

    focus_events = [FocusEvent(timestamp=0.0, focused=True)]
    return BehaviorBatch(
        session_id=session_id,
        started_at=start,
        ended_at=t,
        mouse_moves=[],
        scrolls=[],
        clicks=clicks,
        key_presses=[],
        focus_events=focus_events,
        metadata=_base_metadata("rapid-click"),
    )


def simulate_zero_latency_typing_bot(session_id: str) -> BehaviorBatch:
    start = 0.0
    t = start
    keys: List[KeyPressEvent] = []
    for _ in range(60):
        t += 2.0
        keys.append(KeyPressEvent(timestamp=t, key="a"))

    focus_events = [FocusEvent(timestamp=0.0, focused=True)]
    return BehaviorBatch(
        session_id=session_id,
        started_at=start,
        ended_at=t,
        mouse_moves=[],
        scrolls=[],
        clicks=[],
        key_presses=keys,
        focus_events=focus_events,
        metadata=_base_metadata("zero-typing"),
    )


def simulate_selenium_bot(session_id: str) -> BehaviorBatch:
    # Similar to headless but with different UA.
    batch = simulate_headless_bot(session_id)
    batch.metadata.user_agent = "selenium-bot/1.0"
    return batch


def simulate_bot(session_id: str, bot_type: str) -> BehaviorBatch:
    if bot_type == "headless":
        return simulate_headless_bot(session_id)
    if bot_type == "selenium":
        return simulate_selenium_bot(session_id)
    if bot_type == "rapid_click":
        return simulate_rapid_click_bot(session_id)
    if bot_type == "zero_typing":
        return simulate_zero_latency_typing_bot(session_id)
    # default
    return simulate_headless_bot(session_id)

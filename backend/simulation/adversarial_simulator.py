from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

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


def cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], t: float) -> Tuple[float, float]:
    """
    Calculate point on a cubic Bézier curve at parameter t in [0, 1].
    """
    u = 1.0 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t

    x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
    y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
    return (x, y)


def generate_bezier_trajectory(
    rng: np.random.Generator,
    p0: Tuple[float, float],
    p3: Tuple[float, float],
    n_points: int = 30,
    with_overshoot: bool = True,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Generate points and time deltas for a human-like cursor movement using cubic Bézier curves,
    velocity acceleration/deceleration profiles, and optional target overshoot/correction.
    """
    # Generate control points P1 and P2 with random offsets to create arc curvature
    dx = p3[0] - p0[0]
    dy = p3[1] - p0[1]
    dist = math.hypot(dx, dy)

    # Offset control points perpendicular to direction
    perp_x = -dy / (dist + 1e-6)
    perp_y = dx / (dist + 1e-6)
    offset1 = rng.uniform(-0.3, 0.3) * dist
    offset2 = rng.uniform(-0.3, 0.3) * dist

    p1 = (p0[0] + 0.3 * dx + perp_x * offset1, p0[1] + 0.3 * dy + perp_y * offset1)
    p2 = (p0[0] + 0.7 * dx + perp_x * offset2, p0[1] + 0.7 * dy + perp_y * offset2)

    # If overshoot, target is slightly beyond p3
    actual_p3 = p3
    if with_overshoot and rng.random() > 0.6:
        overshoot_scale = rng.uniform(1.02, 1.08)
        actual_p3 = (p0[0] + dx * overshoot_scale, p0[1] + dy * overshoot_scale)

    points: List[Tuple[float, float]] = []
    dts: List[float] = []

    # Non-linear velocity mapping: sinusoidal acceleration & deceleration (slow start, fast middle, slow end)
    for i in range(n_points):
        # Sinusoidal re-parametrization for realistic human velocity curve
        t_linear = i / max(1, n_points - 1)
        t_curved = 0.5 * (1.0 - math.cos(math.pi * t_linear))

        pt = cubic_bezier(p0, p1, p2, actual_p3, t_curved)
        # Add micro-tremor jitter
        jitter_x = rng.normal(0, 0.4)
        jitter_y = rng.normal(0, 0.4)
        points.append((pt[0] + jitter_x, pt[1] + jitter_y))

        # Time step: slower at ends, faster in middle (15ms - 45ms)
        dt = rng.uniform(15.0, 35.0) * (1.2 - 0.4 * math.sin(math.pi * t_linear))
        dts.append(float(dt))

    # If overshot, append correction steps back to exact p3
    if actual_p3 != p3:
        n_corr = rng.integers(3, 7)
        for i in range(n_corr):
            tc = (i + 1) / n_corr
            cx = actual_p3[0] + tc * (p3[0] - actual_p3[0])
            cy = actual_p3[1] + tc * (p3[1] - actual_p3[1])
            points.append((cx + rng.normal(0, 0.2), cy + rng.normal(0, 0.2)))
            dts.append(float(rng.uniform(20.0, 45.0)))

    return points, dts


def generate_adversarial_bot_session(
    level: int,
    session_id: str,
    seed: int = 42,
) -> BehaviorBatch:
    """
    Generate a simulated raw event batch for a specific adversarial bot attack level (1 to 5).

    Level 1: Deterministic Automation (straight lines, constant velocity, fixed timing)
    Level 2: Randomized Automation (linear path with uniform noise, random timing delays)
    Level 3: Human-Like Cursor Automation (Bézier curves, velocity profile, overshoot/correction)
    Level 4: Human-Like Multi-Signal Automation (Bézier cursor + log-normal typing + scroll acceleration + idle gaps)
    Level 5: Combined Human-Mimicking Bot (full multi-modal mimicry designed to overlap human features)
    """
    rng = np.random.default_rng(seed)
    start_time = 1000.0
    t = start_time

    moves: List[MouseEvent] = []
    clicks: List[ClickEvent] = []
    scrolls: List[ScrollEvent] = []
    key_presses: List[KeyPressEvent] = []
    focus_events: List[FocusEvent] = [FocusEvent(timestamp=t, focused=True)]

    # ----------------------------------------------------
    # LEVEL 1 — DETERMINISTIC AUTOMATION
    # ----------------------------------------------------
    if level == 1:
        # Constant step straight line, fixed 10ms timing
        n_moves = 35
        x, y = 100.0, 100.0
        for _ in range(n_moves):
            t += 10.0  # Constant 10ms
            x += 20.0  # Constant 20px
            y += 0.5   # Constant 0.5px
            moves.append(MouseEvent(timestamp=t, position=MousePosition(x=x, y=y)))

        # Fixed clicks
        for _ in range(5):
            t += 100.0  # Constant 100ms
            clicks.append(ClickEvent(timestamp=t, button="left"))

        # Fixed zero-latency typing
        for _ in range(15):
            t += 5.0   # Constant 5ms
            key_presses.append(KeyPressEvent(timestamp=t, key="a"))

        metadata = BrowserMetadata(
            user_agent="HeadlessChrome/128.0.0.0",
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

    # ----------------------------------------------------
    # LEVEL 2 — RANDOMIZED AUTOMATION
    # ----------------------------------------------------
    elif level == 2:
        # Straight line with uniform random timing & position noise
        n_moves = 40
        x, y = 100.0, 100.0
        for _ in range(n_moves):
            t += float(rng.uniform(8.0, 25.0))
            x += float(rng.uniform(15.0, 25.0))
            y += float(rng.uniform(-2.0, 4.0))
            moves.append(MouseEvent(timestamp=t, position=MousePosition(x=x, y=y)))

        # Randomized click delays
        for _ in range(5):
            t += float(rng.uniform(80.0, 300.0))
            clicks.append(ClickEvent(timestamp=t, button="left"))

        # Randomized typing delays
        for _ in range(15):
            t += float(rng.uniform(15.0, 60.0))
            key_presses.append(KeyPressEvent(timestamp=t, key=str(rng.choice(list("abcdef")))))

        # Small scroll
        scroll_y = 0.0
        for _ in range(4):
            t += float(rng.uniform(20.0, 50.0))
            scroll_y += 100.0
            scrolls.append(ScrollEvent(timestamp=t, delta_y=scroll_y))

        metadata = BrowserMetadata(
            user_agent="Puppeteer/1.0.0",
            language="en-US",
            platform="Linux x86_64",
            screen_width=1920,
            screen_height=1080,
            webgl_fingerprint="bot-gl-2",
            canvas_fingerprint="bot-canvas-2",
            device_entropy=2345.0,
            webdriver=True,
            touch_points=0,
        )

    # ----------------------------------------------------
    # LEVEL 3 — HUMAN-LIKE CURSOR AUTOMATION
    # ----------------------------------------------------
    elif level == 3:
        # Cubic Bézier cursor trajectory with non-linear velocity and overshoot
        p0 = (100.0, 100.0)
        p3 = (600.0, 400.0)
        pts, dts = generate_bezier_trajectory(rng, p0, p3, n_points=35, with_overshoot=True)

        for (px, py), dt in zip(pts, dts):
            t += dt
            moves.append(MouseEvent(timestamp=t, position=MousePosition(x=px, y=py)))

        # Clicks at end of movement
        t += float(rng.uniform(200.0, 500.0))
        clicks.append(ClickEvent(timestamp=t, button="left"))

        # Simple typing
        for _ in range(10):
            t += float(rng.uniform(50.0, 150.0))
            key_presses.append(KeyPressEvent(timestamp=t, key="x"))

        metadata = BrowserMetadata(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Bot/3.0",
            language="en-US",
            platform="Win32",
            screen_width=1920,
            screen_height=1080,
            webgl_fingerprint="bot-gl-3",
            canvas_fingerprint="bot-canvas-3",
            device_entropy=3456.0,
            webdriver=True,
            touch_points=0,
        )

    # ----------------------------------------------------
    # LEVEL 4 — HUMAN-LIKE MULTI-SIGNAL AUTOMATION
    # ----------------------------------------------------
    elif level == 4:
        # Bézier cursor movement
        p0 = (float(rng.uniform(100, 300)), float(rng.uniform(100, 300)))
        p3 = (float(rng.uniform(500, 800)), float(rng.uniform(400, 700)))
        pts, dts = generate_bezier_trajectory(rng, p0, p3, n_points=40, with_overshoot=True)

        for (px, py), dt in zip(pts, dts):
            t += dt
            moves.append(MouseEvent(timestamp=t, position=MousePosition(x=px, y=py)))

        # Human-like log-normal typing latencies with pauses
        num_keys = rng.integers(12, 25)
        for i in range(num_keys):
            # Log-normal distribution: mu=5.0, sigma=0.4 (exp(5) ~ 148ms, std ~ 65ms)
            latency = float(rng.lognormal(mean=5.0, sigma=0.4))
            t += max(30.0, latency)
            # Insert word pause gap
            if i % 5 == 0:
                t += float(rng.uniform(400.0, 1200.0))
            key_presses.append(KeyPressEvent(timestamp=t, key=str(rng.choice(list("abcdefghijklmnopqrstuvwxyz")))))

        # Human-like variable scroll
        num_scrolls = rng.integers(5, 12)
        scroll_y = 0.0
        for _ in range(num_scrolls):
            t += float(rng.uniform(40.0, 120.0))
            scroll_y += float(rng.normal(120.0, 30.0))
            scrolls.append(ScrollEvent(timestamp=t, delta_y=scroll_y))

        # Clicks
        for _ in range(rng.integers(2, 4)):
            t += float(rng.normal(550.0, 150.0))
            clicks.append(ClickEvent(timestamp=t, button="left"))

        # Idle pause
        if rng.random() > 0.5:
            t += float(rng.uniform(1200.0, 2500.0))

        metadata = BrowserMetadata(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0.0.0",
            language="en-US",
            platform="Win32",
            screen_width=1920,
            screen_height=1080,
            webgl_fingerprint="webgl-render-4",
            canvas_fingerprint="canvas-fp-4",
            device_entropy=float(rng.uniform(40000, 100000)),
            webdriver=bool(rng.choice([True, False])),  # Spoofing non-webdriver
            touch_points=0,
        )

    # ----------------------------------------------------
    # LEVEL 5 — COMBINED HUMAN-MIMICKING BOT
    # ----------------------------------------------------
    else:  # Level 5
        # Multi-segment Bézier curves with natural pauses (mimicking human browsing)
        segments = rng.integers(2, 4)
        curr_p = (float(rng.uniform(100, 400)), float(rng.uniform(100, 400)))

        for seg in range(segments):
            next_p = (float(rng.uniform(200, 900)), float(rng.uniform(200, 700)))
            pts, dts = generate_bezier_trajectory(rng, curr_p, next_p, n_points=rng.integers(25, 45), with_overshoot=True)

            for (px, py), dt in zip(pts, dts):
                t += dt
                moves.append(MouseEvent(timestamp=t, position=MousePosition(x=px, y=py)))

            curr_p = next_p
            # Segment pause gap (sub-second or multi-second reflection)
            t += float(rng.uniform(300.0, 1800.0))

        # Human-mimicking log-normal typing cadence & latency variance
        num_keys = rng.integers(15, 30)
        for i in range(num_keys):
            latency = float(rng.lognormal(mean=5.3, sigma=0.45))  # exp(5.3) ~ 200ms
            t += max(40.0, latency)
            if i % 4 == 0:
                t += float(rng.uniform(500.0, 1500.0))
            key_presses.append(KeyPressEvent(timestamp=t, key=str(rng.choice(list("abcdefghijklmnopqrstuvwxyz")))))

        # Human-mimicking click timing & std
        num_clicks = rng.integers(3, 6)
        for _ in range(num_clicks):
            t += float(max(100.0, rng.normal(loc=620.0, scale=180.0)))
            clicks.append(ClickEvent(timestamp=t, button="left"))

        # Human-mimicking scroll dynamics
        num_scrolls = rng.integers(6, 15)
        scroll_y = 0.0
        for _ in range(num_scrolls):
            t += float(rng.uniform(40.0, 150.0))
            scroll_y += float(rng.normal(140.0, 45.0))
            scrolls.append(ScrollEvent(timestamp=t, delta_y=scroll_y))

        # Natural human idle periods
        t += float(rng.uniform(1500.0, 3500.0))

        # Fully realistic browser metadata (mimicking legit human user)
        metadata = BrowserMetadata(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            language="en-US",
            platform="MacIntel",
            screen_width=2560,
            screen_height=1440,
            webgl_fingerprint="ANGLE (Apple, Apple M2 Max, OpenGL 4.1)",
            canvas_fingerprint="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
            device_entropy=float(rng.uniform(80000, 180000)),
            webdriver=False,  # Stealth bot disabling navigator.webdriver
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

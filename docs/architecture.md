# Architecture – BotGuard AI

## Problem Statement

Traditional CAPTCHAs interrupt user flows and are increasingly vulnerable to automation. BotGuard AI provides a **passive, ML‑based human verification system** that continuously analyzes browser behavioral telemetry to distinguish human users from scripted bots **without explicit challenges**.

The system must:

- Collect rich behavioral signals in the browser
- Stream them to a backend API
- Perform feature engineering and multi-engine ML inference
- Produce a composite human confidence score and recommended action (ALLOW / CHALLENGE / BLOCK)
- Expose analytics for monitoring and tuning

---

## High-Level System Architecture

```
Browser Behavioral Telemetry
         │
         ▼
  Feature Engineering Service
         │
         ├──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
Aggregate Features (9-dim)       Raw Event Sequence (60×7)
         │                                  │
         ├──────────────┐                   │
         │              │                   │
         ▼              ▼                   ▼
Calibrated RF    Isolation Forest    Temporal 1D CNN
  (human prob)    (anomaly score)   (sequence class)
         │              │                   │
         └──────────────┴───────────────────┘
                        │
                        ▼
              Risk Engine 2.0
         (weighted multi-factor fusion)
                        │
                        ▼
           ALLOW / CHALLENGE / BLOCK
```

---

## Layer Descriptions

### Frontend (React + TypeScript)
- `BehaviorCollector` hook and service capture:
  - Mouse movement, velocity, acceleration
  - Click intervals and patterns
  - Scroll speed and acceleration
  - Typing rhythm and keypress timing intervals (no character content)
  - Focus/blur and idle periods
  - Session duration and browser metadata
- Aggregates telemetry into time-windowed batches and posts them periodically to the backend.
- Renders login simulation, verification status, confidence score, and analytics charts.

### API Layer (FastAPI)
- `POST /api/collect-behavior` – ingest raw telemetry batches (privacy-sanitized at boundary)
- `POST /api/verify-session` – feature engineering + multi-engine inference + risk decision
- `GET  /api/analytics` – aggregated verification statistics
- `GET  /api/session/{id}/heatmap` – mouse movement heatmap grid
- `GET  /api/explain/{id}` – Random Forest feature importances for session
- `POST /api/challenge/start` – issue behavioral challenge (slider / pattern / reaction / drag-drop)
- `POST /api/challenge/verify` – verify challenge response

### Feature Engineering Service
Converts raw telemetry into two representations:

**Aggregate Features (9-dimensional vector):**
- `avg_mouse_speed` — mean inter-event mouse velocity
- `mouse_accel_variance` — variance of mouse acceleration
- `click_interval_mean / _std` — timing statistics for click events
- `typing_latency_variance` — variance of inter-keypress intervals (no key content)
- `scroll_speed_mean / scroll_accel_mean` — scroll dynamics
- `interaction_density` — events per second
- `avg_idle_duration` — mean idle gap length

**Raw Event Sequence (60×7 matrix):**
Ordered event type one-hot + normalized timestamp + delta + position — for the Temporal CNN.

### Multi-Engine Intelligence Stack

| Engine | Input | Algorithm | Robustness |
|---|---|---|---|
| Calibrated Random Forest | 9-dim aggregate features | `RandomForestClassifier` + Platt scaling | Levels 1–3 |
| Isolation Forest Anomaly Detector | 9-dim aggregate features | Density-based outlier detection on human manifold | Rescues Level 4–5 |
| Temporal 1D CNN | 60×7 raw event sequence | Lightweight 1D convolutional classifier | 100% across Levels 1–5 (simulated) |

### Risk Engine 2.0

Multi-factor fusion across all three engines:

```
composite_risk = 0.35 × RF_risk + 0.30 × anomaly_risk + 0.35 × temporal_risk
              + non-ML indicator penalties (webdriver_detected, headless fingerprint, etc.)
```

High-confidence security escalation overrides apply when multiple signals agree.

**Decision thresholds:**
| composite_risk | Action |
|---|---|
| < 35.0 | `ALLOW` |
| 35.0 – 65.0 | `CHALLENGE` |
| ≥ 65.0 | `BLOCK` |

**Simulated benchmark results (not real-world traffic):**
- Human Control: 98.5% ALLOW, 1.5% false-rejection rate
- Bot Levels 1–5: 100% BLOCK across all adversarial tiers

### Persistence Layer (SQLAlchemy + PostgreSQL/SQLite)

Tables: `sessions`, `telemetry_batches`, `feature_vectors`, `verification_results`, `security_events`, `challenges`.

Managed by Alembic migrations (`alembic upgrade head`). SQLite for local development; PostgreSQL for production.

**Privacy enforcement:** keyboard telemetry is sanitized at the persistence boundary — typed characters, key values, and input text are never stored.

---

## Codebase Layout

```text
backend/
  main.py                      # FastAPI app, lifespan, CORS, middleware
  config.py                    # Centralized path and threshold configuration
  api/
    routes.py                  # All API endpoints
  challenge_engine/
    service.py                 # Challenge generation and verification logic
  database/
    base.py                    # SQLAlchemy declarative base
    session.py                 # Engine builder, SessionLocal, get_db()
    repository.py              # Data Access Layer (all ORM operations)
  ml/
    model.py                   # Calibrated RF loader and inference helper
    model_registry.py          # Model versioning registry utilities
    calibration.py             # CalibratedModelWrapper (Platt scaling)
    evaluation.py              # ML metric computation utilities
    anomaly_detector.py        # Isolation Forest detector
    temporal_model.py          # Temporal 1D CNN architecture
    intelligence_engine.py     # Multi-engine orchestrator
    artifacts/
      human_bot_model_calibrated.pkl  # Production: Calibrated RF model
      anomaly_detector.pkl            # Production: Isolation Forest
      temporal_model.pt               # Production: Temporal 1D CNN weights
      human_bot_model.pkl             # Legacy: Uncalibrated base RF model
      model_registry.json             # Model versioning metadata
  models/
    db_models.py               # SQLAlchemy ORM entity models
    schemas.py                 # Pydantic request/response schemas
  security/
    risk_engine.py             # Risk Engine 2.0 multi-factor fusion
    security_middleware.py     # Request security context middleware
  services/
    decision_engine.py         # evaluate_session() entrypoint
    feature_engineering.py     # Telemetry → feature vector
    feature_store.py           # Feature vector persistence helper
    logging_service.py         # Evaluation result logging
    metrics.py                 # Prometheus metrics
  simulation/
    bot_simulator.py           # Simple bot behavior generators (testing)
    adversarial_simulator.py   # Progressive 5-level adversarial bot profiles

frontend/
  src/
    App.tsx
    components/          # LoginPage, VerificationStatus, BehaviorDashboard, charts
    hooks/
      useBehaviorCollector.ts
    services/
      apiClient.ts

scripts/
  train_model.py               # Train + register base RF model
  retrain_model.py             # Incremental RF retraining utility
  evaluate_baseline.py         # Step 1: Baseline ML evaluation
  evaluate_calibration.py      # Step 2: Calibration analysis
  run_bot_benchmark.py         # Step 3: Adversarial benchmark (Levels 1–5)
  evaluate_anomaly_detection.py # Step 4: Isolation Forest experiment
  train_temporal_experiment.py # Step 5: Temporal CNN training experiment
  evaluate_fusion.py           # Step 6: Multi-engine fusion ablation

tests/
  test_adversarial_benchmark.py
  test_anomaly_detector.py
  test_calibration.py
  test_database_persistence.py
  test_risk_engine_fusion.py
  test_temporal_model.py

alembic/
  env.py                       # Alembic migration environment
  versions/
    001_initial_production_schema.py

docs/
  architecture.md              # This document
  assets/                      # Selected evaluation visualizations
```

---

## Security Considerations

- **Privacy:** Only behavioral timing and spatial metadata are collected. No typed characters, key values, passwords or form content are ever transmitted or persisted. Enforced by sanitization at the backend persistence boundary.
- **Server-side model:** All feature engineering and ML inference is server-side. Clients never see model weights or decision thresholds.
- **Security events:** High-risk indicators (webdriver detected, multi-engine anomaly confirmation) are recorded to the `security_events` table with severity classification.

---

## Deployment

- **Local:** `uvicorn backend.main:app --reload` + `npm run dev`
- **Docker:** `docker compose up` — orchestrates PostgreSQL, backend, and frontend
- **Migrations:** `alembic upgrade head` before starting (runs automatically if `create_all` fallback is present)

> **Note on benchmark results:** All performance metrics cited in this documentation (98.5% ALLOW, 100% BLOCK) are from controlled simulated adversarial evaluation, not real-world production traffic. Real-world performance will depend on the actual human/bot distribution and traffic patterns.

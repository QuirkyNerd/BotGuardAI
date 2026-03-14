BotGuard AI – ML-Based Passive Human Verification
=================================================

BotGuard AI is a production-quality prototype of a **passive, ML‑based human verification system** that serves as an alternative to traditional CAPTCHAs. It detects whether a user is human or an automated bot using **behavioral telemetry** collected from the browser, analyzed by a **FastAPI + scikit-learn** backend.

## Features

- **Passive human verification**
  - Mouse movement trajectory, speed, and acceleration
  - Scroll behavior
  - Click timing and intervals
  - Typing rhythm and latency
  - Focus/blur events and idle times
  - Session duration and interaction density
  - Basic browser fingerprint metadata
- **ML-based detection**
  - RandomForest classifier trained on synthetic human vs bot behavior
  - Feature engineering service to convert raw telemetry into numerical features
  - Decision engine producing:
    - `human_probability`
    - `risk_level`
    - `recommended_action`
- **Modern frontend**
  - React + TypeScript + Vite
  - TailwindCSS UI
  - Chart.js‑based analytics and probability visualizations
  - Real‑time verification status indicator and dashboard
- **Backend**
  - FastAPI with Pydantic schemas
  - Layered architecture (API, feature engineering, ML, decision engine, logging)
  - Model artifact loaded from disk

## Project Structure

```text
botguard-ai/
  backend/
    api/
    services/
    ml/
      artifacts/
    models/
  frontend/
    src/
      components/
      hooks/
      services/
  scripts/
    train_model.py
  docs/
    architecture.md
  requirements.txt
  package.json
  .env.example
  README.md
```

## Getting Started

### 1. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r ../requirements.txt
```

#### Train the model

```bash
cd ..
python scripts/train_model.py
```

This will create the model artifact at `backend/ml/artifacts/human_bot_model.pkl`.

#### Run the backend

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### 2. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at the URL printed by Vite (typically `http://localhost:5173`).

## Environment Configuration

Copy `.env.example` to `.env` in the project root and adjust values as needed.

Backend uses:

- `MODEL_PATH` – path to the serialized scikit-learn model (default: `backend/ml/artifacts/human_bot_model.pkl`)
- `LOG_LEVEL` – log level (e.g. `INFO`, `DEBUG`)
- `ALLOWED_ORIGINS` – comma‑separated list of frontend origins for CORS

## Documentation

- High-level architecture and design: see `docs/architecture.md`.

## Security & Production Notes

This repository is intended as a **hackathon‑grade prototype** but follows **enterprise‑style structure**:

- Layered, modular architecture suitable for service extraction
- Explicit typing and validation at API boundaries
- Centralized feature engineering and decision logic
- Clear extension points for:
  - Real data stores (Postgres, Redis)
  - Message queues and streaming
  - Advanced device/browser fingerprinting
  - External monitoring and SIEM integration

Before production use, see `docs/architecture.md` for security considerations and recommended hardening steps.


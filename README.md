# BotGuard AI

## Behavioral Bot Detection and Human Verification Platform

BotGuard AI is an intelligent behavioral verification system that distinguishes human users from automated bots using machine learning and behavioral telemetry analysis. Unlike traditional CAPTCHA-based solutions, BotGuard AI relies on passive monitoring of user interaction patterns such as mouse movement, scrolling behavior, and typing dynamics to compute a probabilistic human confidence score.

The system provides a scalable architecture combining behavioral feature engineering, machine learning inference, and risk-based decision logic to protect web applications from automated abuse while maintaining a seamless user experience.

---

# Live Deployment

### Frontend Application

https://bot-guard-ai.vercel.app

### Backend API (Swagger Documentation)

https://botguardai.onrender.com/docs

---

# Project Description

## Problem Statement

Automated bots increasingly exploit web applications for activities such as:

* Credential stuffing
* Web scraping
* Spam generation
* Fraudulent transactions

Traditional defenses such as CAPTCHAs introduce significant friction for legitimate users and are frequently bypassed by modern automation frameworks.

The challenge is to design a non-intrusive verification system that accurately differentiates humans from bots using behavioral interaction patterns rather than explicit challenges.

---

# Proposed Solution

BotGuard AI addresses this problem using a behavioral biometrics approach. Instead of interrupting users with CAPTCHAs, the system collects interaction telemetry from the browser and analyzes behavioral patterns using machine learning models.

The platform evaluates signals such as:

* Mouse movement speed and acceleration
* Click timing and interaction intervals
* Scrolling dynamics
* Typing latency patterns
* Interaction density and idle duration

These signals are transformed into behavioral feature vectors and passed through a trained machine learning model that estimates the probability of the session belonging to a human user.

A risk-based decision engine then determines whether the session should be:

* Allowed
* Challenged
* Blocked

This approach enables continuous and invisible verification while maintaining usability.

---

# System Architecture

The platform follows a modular architecture consisting of the following components.

### Frontend Telemetry Collector

Captures real-time behavioral events including:

* Mouse movements
* Clicks
* Keystrokes
* Scroll activity
* Browser metadata

### Behavioral Data Ingestion API

A FastAPI backend service that receives telemetry batches and manages behavioral data processing.

### Feature Engineering Pipeline

Transforms raw behavioral telemetry into structured behavioral features suitable for machine learning inference.

### Machine Learning Inference Engine

A trained classification model evaluates behavioral features and produces a human probability score.

### Risk Evaluation Engine

Calculates a composite risk score and determines recommended security actions.

### Analytics and Monitoring Dashboard

Provides visual insights into:

* Session classification outcomes
* Human probability distributions
* Risk analytics

---

# Workflow

```
User Interaction (Mouse / Keyboard / Scroll)
           ↓
Frontend Telemetry Collector
           ↓
Behavior Data API
           ↓
Feature Engineering
           ↓
Machine Learning Model
           ↓
Risk Scoring Engine
           ↓
Decision Output
           ↓
Frontend Visualization Dashboard
```

This pipeline enables real-time behavioral verification without interrupting legitimate users.

---

# Technologies Used

## Backend

* Python
* FastAPI
* SQLAlchemy
* Uvicorn

## Machine Learning

* Scikit-learn
* NumPy
* Pandas
* Behavioral feature engineering pipeline

## Frontend

* React
* Vite
* JavaScript behavioral event tracking

## Infrastructure

* Render (Backend Hosting)
* Vercel (Frontend Hosting)
* GitHub (Version Control)

---

# Installation and Local Setup

## Prerequisites

* Python 3.11+
* Node.js 18+
* Git

---

# Clone the Repository

```
git clone https://github.com/QuirkyNerd/BotGuardAI.git
cd BotGuardAI
```

---

# Backend Setup

Navigate to the backend directory and create a virtual environment.

```
cd backend
python -m venv venv
```

Activate the environment.

Windows

```
venv\Scripts\activate
```

Linux / macOS

```
source venv/bin/activate
```

Install dependencies.

```
pip install -r requirements.txt
```

Start the FastAPI server.

```
uvicorn backend.main:app --reload
```

Backend API will run at:

http://127.0.0.1:8000

Swagger API documentation:

http://127.0.0.1:8000/docs

---

# Frontend Setup

Navigate to the frontend directory.

```
cd frontend
npm install
```

Start the development server.

```
npm run dev
```

Frontend will run at:

http://localhost:5173

---

# Example API Endpoints

## Collect Behavioral Data

POST /api/collect-behavior

Accepts telemetry batches containing:

* Mouse events
* Keyboard events
* Scroll interactions

---

## Verify User Session

POST /api/verify-session

Returns:

* Human probability score
* Risk level
* Recommended security action

---

## Analytics

GET /api/analytics

Provides session-level metrics and classification statistics.

---

## Explainability

GET /api/explain/{session_id}

Returns the most influential behavioral features used in classification.

---

# Expected Impact

BotGuard AI demonstrates how behavioral biometrics and machine learning can provide a modern alternative to traditional CAPTCHA systems.

The platform enables:

* Passive bot detection without user friction
* Continuous behavioral verification
* Risk-based access control
* Protection against automated abuse

Potential applications include:

* Authentication security
* Fraud prevention
* Web scraping protection
* Account takeover detection
* E-commerce bot mitigation

---

# Future Enhancements

Potential improvements include:

* Deep learning behavioral models
* Real-time anomaly detection
* Integration with Web Application Firewalls
* Federated behavioral datasets
* Adaptive risk scoring mechanisms

---

# License

This project is released under the MIT License.

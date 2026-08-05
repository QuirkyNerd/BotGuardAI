FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install PyTorch CPU (not in requirements.txt to keep it flexible)
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

COPY backend /app/backend
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

# Multi-Engine Intelligence Stack artifact paths
ENV CALIBRATED_MODEL_PATH=backend/ml/artifacts/human_bot_model_calibrated.pkl
ENV ANOMALY_DETECTOR_PATH=backend/ml/artifacts/anomaly_detector.pkl
ENV TEMPORAL_MODEL_PATH=backend/ml/artifacts/temporal_model.pt
ENV MODEL_REGISTRY_PATH=backend/ml/artifacts/model_registry.json

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

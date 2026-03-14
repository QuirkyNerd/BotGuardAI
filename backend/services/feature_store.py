from __future__ import annotations

from typing import Iterable, List

from loguru import logger
from sqlalchemy.orm import Session

from backend.models.db_models import FeatureVectorEntity, SessionEntity
from backend.services.feature_engineering import FEATURE_NAMES, FeatureVector


def persist_feature_vector(
    db: Session,
    session_id: str,
    model_version: str,
    features: FeatureVector,
) -> FeatureVectorEntity:
    """
    Persist a computed feature vector in the feature store.
    """
    session = (
        db.query(SessionEntity)
        .filter(SessionEntity.session_id == session_id)
        .one_or_none()
    )
    if session is None:
        session = SessionEntity(session_id=session_id)
        db.add(session)
        db.flush()

    entity = FeatureVectorEntity(
        session_id=session.session_id,
        model_version=model_version,
        feature_schema=list(FEATURE_NAMES),
        values=list(features.values),
    )
    db.add(entity)
    db.flush()

    logger.debug(
        "Persisted feature vector for session {} with model_version={}",
        session_id,
        model_version,
    )
    return entity


def export_feature_dataset(db: Session) -> List[dict]:
    """
    Export feature vectors and (if available) labels for offline ML training.

    Returns a list of dictionaries with keys:
    - session_id
    - model_version
    - features (list[float])
    - label (optional, if attached via downstream systems)
    """
    vectors: Iterable[FeatureVectorEntity] = db.query(FeatureVectorEntity).all()
    dataset: List[dict] = []
    for fv in vectors:
        record = {
            "session_id": fv.session_id,
            "model_version": fv.model_version,
            "features": fv.values,
        }
        # Placeholder for future label integration.
        dataset.append(record)

    logger.info("Exported {} feature vectors from feature store", len(dataset))
    return dataset

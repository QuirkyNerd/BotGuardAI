from __future__ import annotations

from typing import Iterable, List

from loguru import logger
from sqlalchemy.orm import Session

from backend.database.repository import FeatureRepository, SessionRepository
from backend.models.db_models import FeatureVectorEntity
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
    session = SessionRepository.get_or_create(db, session_id=session_id)
    entity = FeatureRepository.save_feature_vector(
        db=db,
        session_id=session.session_id,
        model_version=model_version,
        feature_schema=list(FEATURE_NAMES),
        values=list(features.values),
    )
    logger.debug(
        "Persisted feature vector for session {} with model_version={}",
        session_id,
        model_version,
    )
    return entity


def export_feature_dataset(db: Session) -> List[dict]:
    """
    Export feature vectors and (if available) labels for offline ML training.
    """
    vectors: Iterable[FeatureVectorEntity] = db.query(FeatureVectorEntity).all()
    dataset: List[dict] = []
    for fv in vectors:
        record = {
            "session_id": fv.session_id,
            "model_version": fv.model_version,
            "features": fv.values,
        }
        dataset.append(record)

    logger.info("Exported {} feature vectors from feature store", len(dataset))
    return dataset

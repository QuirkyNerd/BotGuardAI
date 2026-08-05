from __future__ import annotations

import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateTable

from backend.database.base import Base
from backend.database.repository import (
    ChallengeRepository,
    SecurityEventRepository,
    SessionRepository,
    TelemetryRepository,
    VerificationRepository,
    sanitize_telemetry_payload,
)
from backend.models.db_models import (
    ChallengeEntity,
    FeatureVectorEntity,
    SecurityEventEntity,
    SessionEntity,
    TelemetryBatchEntity,
    VerificationResultEntity,
)
from backend.models.schemas import BehaviorBatch, RiskLevel, VerifyResponse


class TestDatabasePersistence(unittest.TestCase):

    def setUp(self) -> None:
        # Create isolated in-memory SQLite engine per test
        self.engine = create_engine("sqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, class_=Session, autoflush=False, autocommit=False)
        self.db = self.SessionLocal()

    def tearDown(self) -> None:
        self.db.close()

    def test_session_repository_get_or_create_and_update(self) -> None:
        session = SessionRepository.get_or_create(
            self.db,
            session_id="test_session_001",
            user_agent="TestAgent/1.0",
            ip_address="127.0.0.1",
            browser_metadata={"webdriver": False, "screen_width": 1920},
        )
        self.db.commit()

        fetched = self.db.query(SessionEntity).filter(SessionEntity.session_id == "test_session_001").one()
        self.assertEqual(fetched.session_id, "test_session_001")
        self.assertEqual(fetched.user_agent, "TestAgent/1.0")

        # Update summary
        SessionRepository.update_summary(self.db, "test_session_001", 0.95, "LOW")
        self.db.commit()

        updated = self.db.query(SessionEntity).filter(SessionEntity.session_id == "test_session_001").one()
        self.assertEqual(updated.last_human_probability, 0.95)
        self.assertEqual(updated.last_risk_level, "LOW")

    def test_keyboard_privacy_sanitization(self) -> None:
        raw_payload = {
            "session_id": "test_privacy",
            "key_presses": [
                {"timestamp": 100.0, "key": "SecretPassword123", "char": "S", "text": "SecretPassword123", "dt": 0.05},
                {"timestamp": 150.0, "key": "a", "char": "a", "dt": 0.05},
            ],
        }

        sanitized = sanitize_telemetry_payload(raw_payload)

        for kp in sanitized["key_presses"]:
            self.assertNotIn("key", kp)
            self.assertNotIn("char", kp)
            self.assertNotIn("text", kp)
            self.assertIn("timestamp", kp)
            self.assertIn("dt", kp)

    def test_verification_result_persistence_v2(self) -> None:
        SessionRepository.get_or_create(self.db, "test_verify_session")

        eval_resp = VerifyResponse(
            session_id="test_verify_session",
            human_probability=0.88,
            risk_level=RiskLevel.LOW,
            recommended_action="allow",
            risk_score=12.0,
            anomaly_score=15.5,
            temporal_human_probability=0.92,
            risk_components={"behavioral_ml": 12.0, "anomaly": 15.5, "temporal": 8.0},
            triggered_indicators=["normal_human_behavior"],
        )

        vr_entity = VerificationRepository.save_verification_result(self.db, eval_resp, "v2_test")
        self.db.commit()

        fetched = VerificationRepository.get_latest_verification_result(self.db, "test_verify_session")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.human_probability, 0.88)
        self.assertEqual(fetched.anomaly_score, 15.5)
        self.assertEqual(fetched.temporal_human_probability, 0.92)
        self.assertEqual(fetched.risk_components["behavioral_ml"], 12.0)

    def test_security_event_persistence(self) -> None:
        SessionRepository.get_or_create(self.db, "test_security_session")

        indicators = ["webdriver_detected", "escalation_override_multi_engine_anomaly"]
        events = SecurityEventRepository.record_events(self.db, "test_security_session", indicators, 85.0)
        self.db.commit()

        self.assertEqual(len(events), 2)
        sec_rows = self.db.query(SecurityEventEntity).filter(SecurityEventEntity.session_id == "test_security_session").all()
        self.assertEqual(len(sec_rows), 2)
        types = [e.event_type for e in sec_rows]
        self.assertIn("webdriver_detected", types)

    def test_challenge_persistence_and_status(self) -> None:
        SessionRepository.get_or_create(self.db, "test_challenge_session")

        ch = ChallengeRepository.create_challenge(
            self.db,
            session_id="test_challenge_session",
            challenge_type="slider",
            payload={"target": 50},
            solution={"target": 50},
        )
        self.db.commit()

        fetched = ChallengeRepository.get_challenge(self.db, ch.id, "test_challenge_session")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.status, "pending")

    def test_analytics_query_compatibility(self) -> None:
        SessionRepository.get_or_create(self.db, "sess_1")
        eval1 = VerifyResponse(
            session_id="sess_1", human_probability=0.90, risk_level=RiskLevel.LOW,
            recommended_action="allow", risk_score=10.0,
        )
        VerificationRepository.save_verification_result(self.db, eval1, "v1")
        self.db.commit()

        analytics = VerificationRepository.read_analytics(self.db)
        self.assertEqual(analytics.total_sessions, 1)
        self.assertEqual(analytics.average_human_probability, 0.90)

    def test_transaction_rollback_on_error(self) -> None:
        SessionRepository.get_or_create(self.db, "rollback_session")
        self.db.commit()

        try:
            SessionRepository.get_or_create(self.db, "rollback_session_2")
            # Force an error
            raise ValueError("Forced test error")
        except ValueError:
            self.db.rollback()

        count = self.db.query(SessionEntity).count()
        self.assertEqual(count, 1)

    def test_postgresql_dialect_schema_compatibility(self) -> None:
        """
        Validate PostgreSQL dialect/schema compatibility by compiling DDL statements.
        """
        for table in Base.metadata.tables.values():
            ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
            self.assertTrue(len(ddl) > 0)


if __name__ == "__main__":
    unittest.main()

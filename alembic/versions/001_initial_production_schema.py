"""001_initial_production_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-08-05 12:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    existing_tables = inspector.get_table_names()

    # 1. sessions table
    if 'sessions' not in existing_tables:
        op.create_table(
            'sessions',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('session_id', sa.String(length=128), unique=True, nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('ip_address', sa.String(length=64), nullable=True),
            sa.Column('user_agent', sa.Text(), nullable=True),
            sa.Column('browser_metadata', sa.JSON(), nullable=True),
            sa.Column('last_human_probability', sa.Float(), nullable=True),
            sa.Column('last_risk_level', sa.String(length=16), nullable=True),
        )
        op.create_index('ix_sessions_id', 'sessions', ['id'], unique=False)
        op.create_index('ix_sessions_session_id', 'sessions', ['session_id'], unique=True)

    # 2. telemetry_batches table
    if 'telemetry_batches' not in existing_tables:
        op.create_table(
            'telemetry_batches',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('session_id', sa.String(length=128), sa.ForeignKey('sessions.session_id'), nullable=False),
            sa.Column('started_at_ms', sa.Float(), nullable=False),
            sa.Column('ended_at_ms', sa.Float(), nullable=False),
            sa.Column('event_counts', sa.JSON(), nullable=False),
            sa.Column('payload', sa.JSON(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index('ix_telemetry_batches_id', 'telemetry_batches', ['id'], unique=False)
        op.create_index('ix_telemetry_batches_session_id', 'telemetry_batches', ['session_id'], unique=False)

    # 3. feature_vectors table
    if 'feature_vectors' not in existing_tables:
        op.create_table(
            'feature_vectors',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('session_id', sa.String(length=128), sa.ForeignKey('sessions.session_id'), nullable=False),
            sa.Column('model_version', sa.String(length=64), nullable=False),
            sa.Column('feature_schema', sa.JSON(), nullable=False),
            sa.Column('values', sa.JSON(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index('ix_feature_vectors_id', 'feature_vectors', ['id'], unique=False)
        op.create_index('ix_feature_vectors_session_id', 'feature_vectors', ['session_id'], unique=False)
        op.create_index('ix_feature_vectors_model_version', 'feature_vectors', ['model_version'], unique=False)
        op.create_index('ix_feature_vectors_created_at', 'feature_vectors', ['created_at'], unique=False)

    # 4. verification_results table
    if 'verification_results' not in existing_tables:
        op.create_table(
            'verification_results',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('session_id', sa.String(length=128), sa.ForeignKey('sessions.session_id'), nullable=False),
            sa.Column('model_version', sa.String(length=64), nullable=False),
            sa.Column('human_probability', sa.Float(), nullable=False),
            sa.Column('risk_level', sa.String(length=16), nullable=False),
            sa.Column('recommended_action', sa.String(length=32), nullable=False),
            sa.Column('risk_score', sa.Float(), nullable=True),
            sa.Column('anomaly_score', sa.Float(), nullable=True),
            sa.Column('temporal_human_probability', sa.Float(), nullable=True),
            sa.Column('risk_components', sa.JSON(), nullable=True),
            sa.Column('triggered_indicators', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index('ix_verification_results_id', 'verification_results', ['id'], unique=False)
        op.create_index('ix_verification_results_session_id', 'verification_results', ['session_id'], unique=False)
        op.create_index('ix_verification_results_model_version', 'verification_results', ['model_version'], unique=False)
        op.create_index('ix_verification_results_created_at', 'verification_results', ['created_at'], unique=False)
    else:
        columns = [c['name'] for c in inspector.get_columns('verification_results')]
        if 'anomaly_score' not in columns:
            op.add_column('verification_results', sa.Column('anomaly_score', sa.Float(), nullable=True))
        if 'temporal_human_probability' not in columns:
            op.add_column('verification_results', sa.Column('temporal_human_probability', sa.Float(), nullable=True))
        if 'risk_components' not in columns:
            op.add_column('verification_results', sa.Column('risk_components', sa.JSON(), nullable=True))
        if 'triggered_indicators' not in columns:
            op.add_column('verification_results', sa.Column('triggered_indicators', sa.JSON(), nullable=True))

    # 5. security_events table
    if 'security_events' not in existing_tables:
        op.create_table(
            'security_events',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('session_id', sa.String(length=128), sa.ForeignKey('sessions.session_id'), nullable=False),
            sa.Column('event_type', sa.String(length=64), nullable=False),
            sa.Column('severity', sa.String(length=16), nullable=False),
            sa.Column('details', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index('ix_security_events_id', 'security_events', ['id'], unique=False)
        op.create_index('ix_security_events_session_id', 'security_events', ['session_id'], unique=False)
        op.create_index('ix_security_events_event_type', 'security_events', ['event_type'], unique=False)
        op.create_index('ix_security_events_severity', 'security_events', ['severity'], unique=False)
        op.create_index('ix_security_events_created_at', 'security_events', ['created_at'], unique=False)

    # 6. challenges table
    if 'challenges' not in existing_tables:
        op.create_table(
            'challenges',
            sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
            sa.Column('session_id', sa.String(length=128), sa.ForeignKey('sessions.session_id'), nullable=False),
            sa.Column('challenge_type', sa.String(length=32), nullable=False),
            sa.Column('status', sa.String(length=16), nullable=False, server_default='pending'),
            sa.Column('payload', sa.JSON(), nullable=False),
            sa.Column('solution', sa.JSON(), nullable=False),
            sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('success', sa.Boolean(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('solved_at', sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index('ix_challenges_id', 'challenges', ['id'], unique=False)
        op.create_index('ix_challenges_session_id', 'challenges', ['session_id'], unique=False)
        op.create_index('ix_challenges_challenge_type', 'challenges', ['challenge_type'], unique=False)
        op.create_index('ix_challenges_created_at', 'challenges', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_table('challenges')
    op.drop_table('security_events')
    op.drop_table('verification_results')
    op.drop_table('feature_vectors')
    op.drop_table('telemetry_batches')
    op.drop_table('sessions')

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base model that accepts both camelCase (frontend) and snake_case field names."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class EventType(str, Enum):
    MOUSE_MOVE = "mouse_move"
    CLICK = "click"
    SCROLL = "scroll"
    KEY_PRESS = "key_press"
    FOCUS = "focus"
    BLUR = "blur"


class MousePosition(BaseModel):
    x: float = Field(..., description="Mouse X coordinate in pixels")
    y: float = Field(..., description="Mouse Y coordinate in pixels")


class MouseEvent(BaseModel):
    timestamp: float = Field(..., description="Event timestamp in milliseconds since epoch")
    position: MousePosition


class ScrollEvent(BaseModel):
    timestamp: float
    delta_y: float = Field(..., description="Scroll delta in pixels")


class ClickEvent(BaseModel):
    timestamp: float
    button: str = Field(..., description="Mouse button (e.g. left, right, middle)")


class KeyPressEvent(BaseModel):
    timestamp: float
    key: str = Field(..., description="Key identifier")


class FocusEvent(BaseModel):
    timestamp: float
    focused: bool = Field(..., description="True when window focused, False when blurred")


class BrowserMetadata(CamelModel):
    user_agent: str
    language: Optional[str] = None
    platform: Optional[str] = None
    screen_width: Optional[int] = None
    screen_height: Optional[int] = None
    webgl_fingerprint: Optional[str] = None
    canvas_fingerprint: Optional[str] = None
    device_entropy: Optional[float] = None
    webdriver: Optional[bool] = None
    touch_points: Optional[int] = None


class BehaviorBatch(BaseModel):
    """
    A time-windowed batch of behavioral events associated with a session.
    """

    session_id: str = Field(..., description="Logical session identifier")
    started_at: float = Field(..., description="Batch start time (ms since epoch)")
    ended_at: float = Field(..., description="Batch end time (ms since epoch)")

    mouse_moves: List[MouseEvent] = Field(default_factory=list)
    scrolls: List[ScrollEvent] = Field(default_factory=list)
    clicks: List[ClickEvent] = Field(default_factory=list)
    key_presses: List[KeyPressEvent] = Field(default_factory=list)
    focus_events: List[FocusEvent] = Field(default_factory=list)

    metadata: Optional[BrowserMetadata] = None


class VerifyRequest(BaseModel):
    """
    Request for verifying a session, optionally including the most recent batch.
    """

    session_id: str
    latest_batch: Optional[BehaviorBatch] = None


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerifyResponse(BaseModel):
    session_id: str
    human_probability: float
    risk_level: RiskLevel
    recommended_action: str
    risk_score: float


class ChallengeType(str, Enum):
    SLIDER = "slider"
    DRAG_DROP = "drag_drop"
    PATTERN = "pattern"
    REACTION = "reaction"


class ChallengeStartRequest(BaseModel):
    session_id: str
    preferred_type: Optional[ChallengeType] = None


class ChallengeStartResponse(BaseModel):
    challenge_id: int
    session_id: str
    challenge_type: ChallengeType
    payload: dict


class ChallengeVerifyRequest(BaseModel):
    session_id: str
    challenge_id: int
    response_payload: dict


class ChallengeVerifyResponse(BaseModel):
    challenge_id: int
    session_id: str
    success: bool
    updated_risk_score: Optional[float] = None


class AnalyticsBucket(BaseModel):
    label: str
    count: int


class AnalyticsResponse(BaseModel):
    total_sessions: int
    average_human_probability: float
    risk_distribution: List[AnalyticsBucket]


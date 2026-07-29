"""Privacy-safe domain contracts for the elder companion MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


PROTOCOL_VERSION = "1.0"

# Keep the wire contract string-based so older clients remain compatible.
SUPPORTED_EXPRESSIONS = frozenset({"neutral", "warm", "caring", "uncertain", "offline"})
SUPPORTED_POSES = frozenset({"idle", "quiet_presence", "listening", "gentle_wave"})
SUPPORTED_MOTION_LEVELS = frozenset({"none", "low", "medium"})


def normalize_expression(value: str) -> str:
    value = str(value or "neutral").strip()
    return value if value in SUPPORTED_EXPRESSIONS else "neutral"


def normalize_pose(value: str) -> str:
    value = str(value or "idle").strip()
    return value if value in SUPPORTED_POSES else "idle"


def normalize_motion_level(value: str) -> str:
    value = str(value or "low").strip()
    return value if value in SUPPORTED_MOTION_LEVELS else "low"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceMode(str, Enum):
    REAL = "real"
    SIMULATED = "simulated"
    MANUAL = "manual"
    DERIVED = "derived"


class KnowledgeStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    STALE = "stale"
    CONFLICTING = "conflicting"
    UNAVAILABLE = "unavailable"


@dataclass
class Observation:
    kind: str
    value: str
    source_id: str
    source_mode: SourceMode
    confidence: float
    observed_at: str = field(default_factory=utc_now)
    ttl_seconds: int = 300
    quality: str = "valid"
    evidence_summary: str = ""
    observation_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "kind": self.kind,
            "value": self.value,
            "source": {"id": self.source_id, "mode": self.source_mode.value},
            "confidence": self.confidence,
            "observed_at": self.observed_at,
            "ttl_seconds": self.ttl_seconds,
            "quality": self.quality,
            "evidence_summary": self.evidence_summary,
            "contains_raw_media": False,
        }


@dataclass
class StateDimension:
    label: str = "unknown"
    status: KnowledgeStatus = KnowledgeStatus.UNKNOWN
    confidence: float = 0.0
    source_mode: SourceMode | None = None
    observed_at: str | None = None
    evidence_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.status.value,
            "confidence": self.confidence,
            "source_mode": self.source_mode.value if self.source_mode else None,
            "observed_at": self.observed_at,
            "evidence_summary": self.evidence_summary,
        }


@dataclass
class ElderState:
    revision: int = 0
    generated_at: str = field(default_factory=utc_now)
    affect: StateDimension = field(default_factory=StateDimension)
    activity: StateDimension = field(default_factory=StateDimension)
    meal: StateDimension = field(default_factory=StateDimension)
    presence: StateDimension = field(default_factory=StateDimension)
    simulation_present: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "generated_at": self.generated_at,
            "simulation_present": self.simulation_present,
            "dimensions": {
                "affect": self.affect.to_dict(),
                "activity": self.activity.to_dict(),
                "meal": self.meal.to_dict(),
                "presence": self.presence.to_dict(),
            },
        }


@dataclass
class CompanionDirective:
    expression: str = "neutral"
    pose: str = "idle"
    motion_level: str = "low"
    message: str = "我在听。"
    priority: str = "idle"
    reason: str = "initial"
    valid_until: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": normalize_expression(self.expression),
            "pose": normalize_pose(self.pose),
            "motion_level": normalize_motion_level(self.motion_level),
            "message": self.message,
            "priority": self.priority,
            "reason": self.reason,
            "valid_until": self.valid_until,
        }

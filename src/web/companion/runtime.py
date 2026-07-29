"""Long-lived semantic state runtime used by the FastAPI WebSocket server."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from .models import (
    PROTOCOL_VERSION,
    CompanionDirective,
    ElderState,
    KnowledgeStatus,
    Observation,
    SourceMode,
    StateDimension,
    utc_now,
)
from .policy import directive_for

Broadcast = Callable[[dict[str, Any]], Awaitable[None]]


class ElderStateRuntime:
    """Own the HMI companion state without accepting raw camera or audio data.

    This MVP accepts only semantic observations from simulation and explicit
    user reports. Device adapters can later call ``submit_observation`` after
    converting their low-level readings into the same privacy-safe contract.
    """

    def __init__(self, broadcast: Broadcast) -> None:
        self._broadcast = broadcast
        self._state = ElderState()
        self._directive = directive_for(self._state)
        self._stream_id = str(uuid4())
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._started = False
        self._capabilities = {
            "conversation_affect": self._capability("available", SourceMode.DERIVED),
            "manual_elder_report": self._capability("available", SourceMode.MANUAL),
            "meal_activity": self._capability("simulated", SourceMode.SIMULATED),
            "visual_affect": self._capability("unavailable", None),
            "posture_activity": self._capability("unavailable", None),
            "fall_candidate": self._capability("unavailable", None),
            "wearable_health": self._capability("unavailable", None),
        }

    @staticmethod
    def _capability(availability: str, mode: SourceMode | None) -> dict[str, Any]:
        return {
            "availability": availability,
            "source_mode": mode.value if mode else None,
            "last_observation_at": None,
            "contains_raw_media": False,
        }

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def _envelope(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        return {
            "action": action,
            "data": data,
            "meta": {
                "protocol_version": PROTOCOL_VERSION,
                "stream_id": self._stream_id,
                "sequence": self._sequence,
                "message_id": str(uuid4()),
                "sent_at": utc_now(),
                "state_revision": self._state.revision,
            },
        }

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self._envelope(
                "companion.snapshot",
                {
                    "elder_state": self._state.to_dict(),
                    "companion_directive": self._directive.to_dict(),
                    "capabilities": self._capabilities,
                    "simulation": {"active": self._state.simulation_present, "label": "模拟数据"},
                },
            )

    async def submit_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Accept explicit low-risk reports; no report triggers device actions."""
        kind = str(report.get("kind", "")).strip()
        value = str(report.get("value", "")).strip()
        allowed = {
            "affect": {"positive", "neutral", "low"},
            "meal": {"started", "completed"},
        }
        if kind not in allowed or value not in allowed[kind]:
            return self._envelope(
                "error",
                {"message": "不支持的陪伴状态上报", "allowed": allowed},
            )
        observation = Observation(
            kind=f"elder.{kind}",
            value=value,
            source_id="hmi.manual_report",
            source_mode=SourceMode.MANUAL,
            confidence=1.0,
            evidence_summary="由老人通过 HMI 手动确认",
        )
        await self.submit_observation(observation)
        return self._envelope("elder.report.accepted", {"observation": observation.to_dict()})

    async def run_scenario(self, name: str) -> dict[str, Any]:
        """Run one deterministic, clearly simulated scenario for local demos."""
        scenarios = {
            "neutral": ("affect", "neutral", "模拟：平静在家"),
            "positive": ("affect", "positive", "模拟：积极情绪线索"),
            "low_mood": ("affect", "low", "模拟：可能心情低落"),
            "meal": ("meal", "started", "模拟：用餐活动候选"),
            "meal_complete": ("meal", "completed", "模拟：手动确认用餐完成"),
            "uncertain": ("affect", "unknown", "模拟：情绪线索不确定"),
        }
        if name not in scenarios:
            return self._envelope("error", {"message": "未知模拟场景", "available": list(scenarios)})
        kind, value, summary = scenarios[name]
        observation = Observation(
            kind=f"elder.{kind}",
            value=value,
            source_id="companion.simulator",
            source_mode=SourceMode.SIMULATED,
            confidence=0.82 if value != "unknown" else 0.25,
            evidence_summary=summary,
        )
        await self.submit_observation(observation)
        return self._envelope("simulation.started", {"scenario": name, "source_mode": "simulated"})

    async def submit_observation(self, observation: Observation) -> None:
        """Fuse a semantic observation and broadcast state/directive deltas."""
        async with self._lock:
            changed = self._apply(observation)
            if not changed:
                return
            self._state.revision += 1
            self._state.generated_at = utc_now()
            self._directive = directive_for(self._state)
            state_payload = self._envelope("elder.state.changed", self._state.to_dict())
            directive_payload = self._envelope("companion.directive.changed", self._directive.to_dict())
        await self._broadcast(state_payload)
        await self._broadcast(directive_payload)

    def _apply(self, observation: Observation) -> bool:
        dimension: StateDimension | None = None
        if observation.kind == "elder.affect":
            dimension = self._state.affect
            if observation.value == "unknown":
                replacement = StateDimension(
                    label="unknown",
                    status=KnowledgeStatus.UNKNOWN,
                    confidence=observation.confidence,
                    source_mode=observation.source_mode,
                    observed_at=observation.observed_at,
                    evidence_summary=observation.evidence_summary,
                )
            else:
                replacement = StateDimension(
                    label=observation.value,
                    status=KnowledgeStatus.KNOWN,
                    confidence=observation.confidence,
                    source_mode=observation.source_mode,
                    observed_at=observation.observed_at,
                    evidence_summary=observation.evidence_summary,
                )
            self._state.affect = replacement
        elif observation.kind == "elder.meal":
            label = "in_progress" if observation.value == "started" else "completed"
            self._state.meal = StateDimension(
                label=label,
                status=KnowledgeStatus.KNOWN,
                confidence=observation.confidence,
                source_mode=observation.source_mode,
                observed_at=observation.observed_at,
                evidence_summary=observation.evidence_summary,
            )
            self._state.activity = StateDimension(
                label="meal_activity_candidate" if label == "in_progress" else "unknown",
                status=KnowledgeStatus.KNOWN if label == "in_progress" else KnowledgeStatus.UNKNOWN,
                confidence=observation.confidence if label == "in_progress" else 0.0,
                source_mode=observation.source_mode,
                observed_at=observation.observed_at,
                evidence_summary=observation.evidence_summary,
            )
        else:
            return False
        self._state.simulation_present = observation.source_mode is SourceMode.SIMULATED
        return True

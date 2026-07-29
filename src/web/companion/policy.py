"""Deterministic, low-interruption policy for the on-screen companion."""

from __future__ import annotations

from .models import CompanionDirective, ElderState, KnowledgeStatus


def directive_for(state: ElderState) -> CompanionDirective:
    """Map qualified elder state to a calm companion response.

    This policy deliberately never mirrors possible distress as a distressed
    character and never states an inferred meal as a completed meal.
    """
    meal = state.meal
    affect = state.affect

    if meal.status is KnowledgeStatus.KNOWN and meal.label in {"candidate", "likely", "in_progress"}:
        return CompanionDirective(
            expression="warm",
            pose="quiet_presence",
            motion_level="none",
            message="慢慢吃，我在这里。",
            priority="supportive",
            reason="elder.meal_activity",
        )

    if affect.status is KnowledgeStatus.KNOWN and affect.label in {"low", "distressed_candidate"}:
        return CompanionDirective(
            expression="caring",
            pose="listening",
            motion_level="low",
            message="如果想说说话，我在听。",
            priority="supportive",
            reason="elder.possible_low_mood",
        )

    if affect.status is KnowledgeStatus.KNOWN and affect.label == "positive":
        return CompanionDirective(
            expression="warm",
            pose="gentle_wave",
            motion_level="low",
            message="看到您心情不错，我也很开心。",
            priority="supportive",
            reason="elder.positive_affect",
        )

    if affect.status in {KnowledgeStatus.CONFLICTING, KnowledgeStatus.STALE}:
        return CompanionDirective(
            expression="uncertain",
            pose="idle",
            motion_level="none",
            message="我在这里，需要时叫我。",
            priority="idle",
            reason="elder.state_uncertain",
        )

    return CompanionDirective()

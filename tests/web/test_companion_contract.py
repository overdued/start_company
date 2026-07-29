from src.web.companion.models import CompanionDirective
from src.web.companion.policy import directive_for
from src.web.companion.models import ElderState, KnowledgeStatus, StateDimension


def test_directive_policy_exposes_renderable_pose_and_motion():
    state = ElderState(
        affect=StateDimension(label="positive", status=KnowledgeStatus.KNOWN),
    )
    directive = directive_for(state).to_dict()
    assert directive["pose"] == "gentle_wave"
    assert directive["motion_level"] == "low"


def test_unknown_directive_values_fall_back_safely():
    directive = CompanionDirective(expression="danger", pose="raw_camera", motion_level="turbo")
    assert directive.to_dict()["expression"] == "neutral"
    assert directive.to_dict()["pose"] == "idle"
    assert directive.to_dict()["motion_level"] == "low"

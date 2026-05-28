from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

@dataclass
class ActionIntent:
    action_name: str
    action_type: str = "unknown"
    target_resource: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    environment: str = "unknown"
    actor: str = "agent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def normalize_action_intent(payload: Dict[str, Any]) -> ActionIntent:
    payload = payload or {}
    return ActionIntent(
        action_name=str(payload.get("action_name", "") or "").strip(),
        action_type=str(payload.get("action_type", "unknown") or "unknown").strip(),
        target_resource=str(payload.get("target_resource", "") or "").strip(),
        params=dict(payload.get("params", {}) or {}),
        environment=str(payload.get("environment", "unknown") or "unknown").strip().lower(),
        actor=str(payload.get("actor", "agent") or "agent").strip(),
    )

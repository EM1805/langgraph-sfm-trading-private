from __future__ import annotations
from typing import Dict, List

def derive_action_effects(intent: Dict[str, object], action_spec: Dict[str, object] | None = None) -> List[str]:
    action_spec = action_spec or {}
    direct_effects = list(action_spec.get("direct_effects", []) or [])
    params = dict(intent.get("params", {}) or {})
    if bool(params.get("bypass_approval", False)) and "bypass_control" not in direct_effects:
        direct_effects.append("bypass_control")
    if bool(params.get("suppress_review", False)) and "suppress_review" not in direct_effects:
        direct_effects.append("suppress_review")
    if bool(params.get("delete", False)) and "delete_resource" not in direct_effects:
        direct_effects.append("delete_resource")
    return sorted(set(str(x) for x in direct_effects if str(x).strip()))

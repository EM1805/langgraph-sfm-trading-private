from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActionIntent:
    action_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    environment: Optional[str] = None
    actor_role: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"action_name": self.action_name, "params": dict(self.params or {})}
        if self.environment is not None:
            out["environment"] = self.environment
        if self.actor_role is not None:
            out["actor_role"] = self.actor_role
        return out


@dataclass
class HistoricalEvent:
    action_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)
    event_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "action_name": self.action_name,
            "params": dict(self.params or {}),
            "outcome": dict(self.outcome or {}),
        }
        if self.event_time is not None:
            out["event_time"] = self.event_time
        return out


@dataclass
class PathTreatmentSpec:
    path_id: str
    treatment_var: str
    treated_value: Any
    control_value: Any
    outcome_var: str
    required_confounders: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    colliders: List[str] = field(default_factory=list)
    negative_controls: List[str] = field(default_factory=list)
    hard_match_keys: List[str] = field(default_factory=list)
    soft_balance_keys: List[str] = field(default_factory=list)

    def to_path_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "treatment_var": self.treatment_var,
            "treated_value": self.treated_value,
            "control_value": self.control_value,
            "outcome_var": self.outcome_var,
            "required_confounders": list(self.required_confounders or []),
            "mediators": list(self.mediators or []),
            "colliders": list(self.colliders or []),
            "negative_controls": list(self.negative_controls or []),
            "hard_match_keys": list(self.hard_match_keys or []),
            "soft_balance_keys": list(self.soft_balance_keys or []),
        }

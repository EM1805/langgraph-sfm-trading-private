from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from amantia.contracts import ActionPackage, DecisionPackage, normalize_action_package
from amantia.gate import DecisionGate


_DECISION_BASE_SCORE = {
    "allow": 50,
    "warn": 35,
    "ask_clarification": 30,
    "abstain": 10,
    "veto": -100,
}

_AMBIGUITY_VALUES = {"high", "very_high", "critical", "unknown"}
_HIGH_RISK_VALUES = {"high", "critical"}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "unknown") -> str:
    return _clean_str(value, default).lower()


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return list(value)
    return [value]


def _default_action_type(action_name: str) -> str:
    name = action_name.lower()
    if name in {"use_tool", "search_information", "call_tool", "execute_tool"}:
        return "tool_call"
    if name in {"delete_resource", "delete_files", "write_file", "send_email", "modify_resource"}:
        return "state_change"
    if name in {"ask_clarification", "answer_directly", "abstain", "veto_sensitive_action"}:
        return "communication"
    return "unknown"


def _default_target_resource(action_name: str) -> str:
    name = action_name.lower()
    if name in {"ask_clarification", "answer_directly", "abstain", "veto_sensitive_action"}:
        return "conversation"
    if name in {"use_tool", "search_information", "call_tool", "execute_tool"}:
        return "tool"
    return ""


def _candidate_name(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        return _clean_str(
            candidate.get("action_name")
            or candidate.get("candidate_action")
            or candidate.get("selected_action")
            or candidate.get("name")
        )
    return _clean_str(candidate)


def _merge_context(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(override or {}))
    return merged


@dataclass
class BrainRun:
    """Full result of one online Operational Brain pass."""

    selected: DecisionPackage
    evaluated_actions: List[DecisionPackage] = field(default_factory=list)
    input_package: Dict[str, Any] = field(default_factory=dict)
    mode: str = "online"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "selected": self.selected.to_dict(),
            "evaluated_actions": [item.to_dict() for item in self.evaluated_actions],
            "input_package": dict(self.input_package or {}),
            "notes": list(self.notes or []),
        }


class OperationalBrain:
    """Online orchestrator for Amantia.

    The Operational Brain is intentionally small in this step. It does not run
    heavy Discovery/Estimation. It receives candidate actions from an LLM/agent,
    normalizes them into ActionPackage objects, calls the DecisionGate for each
    candidate, and selects the safest useful next action for the final response.
    """

    def __init__(self, *, decision_gate: Optional[DecisionGate] = None, **gate_kwargs: Any) -> None:
        self.decision_gate = decision_gate or DecisionGate(**gate_kwargs)

    def build_action_packages(self, payload: Any) -> List[ActionPackage]:
        """Normalize a user/LLM payload into one ActionPackage per candidate."""
        if isinstance(payload, ActionPackage):
            return [payload]

        raw = dict(payload or {}) if isinstance(payload, Mapping) else {}
        context = _as_dict(raw.get("context"))
        trusted_context = _as_dict(raw.get("trusted_runtime_context") or raw.get("trusted_context"))
        untrusted_context = _as_dict(raw.get("untrusted_llm_context") or raw.get("llm_context"))
        params = _as_dict(raw.get("params"))
        candidates = _as_list(raw.get("candidate_actions") or raw.get("actions"))

        if not candidates:
            single = raw.get("candidate_action") or raw.get("action_name") or raw.get("selected_action")
            candidates = [single] if single else [{}]

        packages: List[ActionPackage] = []
        candidate_names = [_candidate_name(c) for c in candidates if _candidate_name(c)]

        for candidate in candidates:
            if isinstance(candidate, Mapping):
                candidate_payload = dict(raw)
                candidate_payload.update(dict(candidate))
                candidate_payload["context"] = _merge_context(context, _as_dict(candidate.get("context")))
                candidate_payload["trusted_runtime_context"] = _merge_context(
                    trusted_context,
                    _as_dict(candidate.get("trusted_runtime_context") or candidate.get("trusted_context")),
                )
                candidate_payload["untrusted_llm_context"] = _merge_context(
                    untrusted_context,
                    _as_dict(candidate.get("untrusted_llm_context") or candidate.get("llm_context")),
                )
                candidate_payload["params"] = _merge_context(params, _as_dict(candidate.get("params")))
            else:
                name = _candidate_name(candidate)
                candidate_payload = dict(raw)
                candidate_payload.update(
                    {
                        "candidate_action": name,
                        "action_name": name,
                        "action_type": raw.get("action_type") or _default_action_type(name),
                        "target_resource": raw.get("target_resource") or _default_target_resource(name),
                        "context": dict(context),
                        "trusted_runtime_context": dict(trusted_context),
                        "untrusted_llm_context": dict(untrusted_context),
                        "params": dict(params),
                    }
                )

            action_name = _candidate_name(candidate_payload)
            if action_name and not candidate_payload.get("candidate_action"):
                candidate_payload["candidate_action"] = action_name
            if action_name and not candidate_payload.get("action_type"):
                candidate_payload["action_type"] = _default_action_type(action_name)
            if action_name and not candidate_payload.get("target_resource"):
                candidate_payload["target_resource"] = _default_target_resource(action_name)

            candidate_payload.setdefault("candidate_actions", candidate_names)
            packages.append(normalize_action_package(candidate_payload))

        return packages

    def evaluate_candidates(self, payload: Any) -> List[DecisionPackage]:
        packages = self.build_action_packages(payload)
        return [self.decision_gate.evaluate(pkg) for pkg in packages]

    def select_decision(self, decisions: Sequence[DecisionPackage], payload: Any = None) -> DecisionPackage:
        if not decisions:
            empty_action = ActionPackage(
                action_name="ask_clarification",
                candidate_action="ask_clarification",
                user_message=_clean_str(_as_dict(payload).get("user_message")) if isinstance(payload, Mapping) else "",
            )
            result = self.decision_gate.evaluate(empty_action)
            result.decision = "ask_clarification"
            result.reason = "No candidate action was supplied."
            result.llm_instruction = "Ask the user or LLM planner to provide a candidate action before proceeding."
            return result

        raw = dict(payload or {}) if isinstance(payload, Mapping) else {}
        context = _as_dict(raw.get("context"))
        trusted_context = _as_dict(raw.get("trusted_runtime_context") or raw.get("trusted_context"))
        untrusted_context = _as_dict(raw.get("untrusted_llm_context") or raw.get("llm_context"))
        split_context_mode = bool(trusted_context or untrusted_context)
        runtime_context = trusted_context if split_context_mode else context
        ambiguity = _clean_lower(
            (runtime_context.get("ambiguity") if split_context_mode else raw.get("ambiguity") or context.get("ambiguity")),
            "unknown",
        )
        risk_level = _clean_lower(
            (runtime_context.get("risk_level") if split_context_mode else raw.get("risk_level") or context.get("risk_level")),
            "unknown",
        )

        scored: List[Tuple[int, int, DecisionPackage]] = []
        for index, decision in enumerate(decisions):
            name = (decision.selected_action or decision.candidate_action or "").lower()
            score = _DECISION_BASE_SCORE.get(decision.decision, 0)

            if ambiguity in _AMBIGUITY_VALUES:
                if name == "ask_clarification":
                    score += 45
                if name == "answer_directly":
                    score -= 35

            if risk_level in _HIGH_RISK_VALUES:
                if decision.decision in {"ask_clarification", "abstain"}:
                    score += 20
                if name in {"delete_resource", "delete_files", "modify_resource", "send_email"}:
                    score -= 25

            cf = dict(getattr(decision, "causal_counterfactual", {}) or {})
            recommended = str(cf.get("recommended_action") or "").strip().lower()
            cf_status = str(cf.get("comparison_status") or "").strip()
            if recommended:
                if name == recommended:
                    score += 18
                elif cf_status == "alternative_recommended" and decision.decision in {"allow", "warn"}:
                    score -= 12

            if "COUNTERFACTUAL_ALTERNATIVE_RECOMMENDED" in set(decision.reason_codes or []):
                score -= 5

            if decision.decision == "veto":
                score -= 50

            scored.append((score, -index, decision))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def run(self, payload: Any) -> BrainRun:
        decisions = self.evaluate_candidates(payload)
        selected = self.select_decision(decisions, payload)
        notes = [
            "OperationalBrain step2 used online-only routing.",
            "Heavy Discovery/RCT remain offline; optional online SCM-ID, Estimation, and Counterfactual adapters enrich the Gate when evidence is supplied.",
        ]
        return BrainRun(
            selected=selected,
            evaluated_actions=list(decisions),
            input_package=dict(payload or {}) if isinstance(payload, Mapping) else {},
            notes=notes,
        )


def run_operational_brain(payload: Any, **gate_kwargs: Any) -> Dict[str, Any]:
    return OperationalBrain(**gate_kwargs).run(payload).to_dict()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Amantia Operational Brain over candidate actions.")
    parser.add_argument("--input", required=True, help="Path to JSON payload with user_message + candidate_actions.")
    parser.add_argument("--out", default="out/operational_brain_decision.json", help="Output JSON path.")
    parser.add_argument("--registry-path", default="action_registry.yaml")
    parser.add_argument("--path-library-path", default="dangerous_paths.yaml")
    parser.add_argument("--graph-path", default="operational_causal_graph.yaml")
    parser.add_argument("--event-log-path", default="historical_action_events.jsonl")
    parser.add_argument("--validation-plan-path", default="out/validation_plan_level2.csv")
    parser.add_argument("--authority-cards-path", default="out/veto/causal_authority_cards.jsonl")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    brain = OperationalBrain(
        registry_path=args.registry_path,
        path_library_path=args.path_library_path,
        graph_path=args.graph_path,
        event_log_path=args.event_log_path,
        validation_plan_path=args.validation_plan_path,
        authority_cards_path=args.authority_cards_path,
    )
    result = brain.run(payload).to_dict()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    selected = result.get("selected", {})
    print(json.dumps({
        "status": "ok",
        "out": str(out_path),
        "decision": selected.get("decision"),
        "selected_action": selected.get("selected_action"),
        "runtime_decision": selected.get("runtime_decision"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

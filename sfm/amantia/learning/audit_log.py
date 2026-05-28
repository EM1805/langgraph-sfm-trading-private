from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

DEFAULT_AUDIT_LOG_PATH = "out/learning/audit_log.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


@dataclass
class AuditEvent:
    """Append-only event used by the online Learning Loop."""

    event_type: str = "decision"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=utc_now_iso)
    request_id: str = ""

    user_message: str = ""
    candidate_actions: List[str] = field(default_factory=list)
    selected_action: str = ""
    gate_decision: str = ""
    runtime_decision: str = ""
    veto: bool = False

    risk_level: str = "unknown"
    ambiguity: str = "unknown"
    evidence_tier: str = "unknown"
    identification_tier: str = "unknown"
    confidence: str = "unknown"

    reason_codes: List[str] = field(default_factory=list)
    reason: str = ""
    outcome: str = "unknown"
    user_satisfaction: Optional[float] = None

    source: str = "amantia_online"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLog:
    """Small stdlib-only JSONL logger for online Amantia decisions."""

    def __init__(self, path: str | Path = DEFAULT_AUDIT_LOG_PATH) -> None:
        self.path = Path(path)

    def append_event(self, event: AuditEvent | Mapping[str, Any]) -> Dict[str, Any]:
        record = event.to_dict() if isinstance(event, AuditEvent) else dict(event or {})
        record.setdefault("event_type", "decision")
        record.setdefault("event_id", str(uuid.uuid4()))
        record.setdefault("timestamp", utc_now_iso())
        record.setdefault("source", "amantia_online")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def append_decision_run(self, run_result: Mapping[str, Any]) -> Dict[str, Any]:
        return self.append_event(build_decision_event(run_result))

    def append_outcome(
        self,
        *,
        request_id: str = "",
        event_id: str = "",
        selected_action: str = "",
        outcome: str = "unknown",
        user_satisfaction: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        event = AuditEvent(
            event_type="outcome",
            request_id=request_id,
            selected_action=selected_action,
            outcome=outcome,
            user_satisfaction=user_satisfaction,
            payload={"linked_event_id": event_id, "metadata": dict(metadata or {})},
        )
        return self.append_event(event)

    def iter_events(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return iter(())

        def _read() -> Iterator[Dict[str, Any]]:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"event_type": "corrupt_line", "raw_line": line}

        return _read()

    def read_events(self, *, max_events: int | None = None) -> List[Dict[str, Any]]:
        events = list(self.iter_events())
        if max_events is not None and max_events >= 0:
            return events[-max_events:]
        return events


def _candidate_names(run_result: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    input_package = _as_dict(run_result.get("input_package"))

    for candidate in _as_list(input_package.get("candidate_actions")):
        if isinstance(candidate, Mapping):
            name = _clean_str(candidate.get("action_name") or candidate.get("candidate_action") or candidate.get("name"))
        else:
            name = _clean_str(candidate)
        if name and name not in names:
            names.append(name)

    for decision in _as_list(run_result.get("evaluated_actions")):
        if isinstance(decision, Mapping):
            name = _clean_str(decision.get("selected_action") or decision.get("candidate_action"))
            if name and name not in names:
                names.append(name)

    return names


def build_decision_event(run_result: Mapping[str, Any]) -> AuditEvent:
    """Convert OperationalBrain/propose_and_decide output into an AuditEvent."""
    run_result = dict(run_result or {})
    selected = _as_dict(run_result.get("selected"))
    input_package = _as_dict(run_result.get("input_package"))
    llm_response = _as_dict(run_result.get("llm_response"))
    context = _as_dict(input_package.get("context") or llm_response.get("context"))
    audit_payload = _as_dict(selected.get("audit_payload"))

    user_message = _clean_str(input_package.get("user_message") or llm_response.get("user_message") or audit_payload.get("user_message"))
    gate_decision = _clean_str(selected.get("decision"), "abstain")
    runtime_decision = _clean_str(selected.get("runtime_decision"), "UNKNOWN")
    selected_action = _clean_str(selected.get("selected_action") or selected.get("candidate_action"))

    return AuditEvent(
        request_id=_clean_str(audit_payload.get("request_id") or input_package.get("request_id")),
        user_message=user_message,
        candidate_actions=_candidate_names(run_result),
        selected_action=selected_action,
        gate_decision=gate_decision,
        runtime_decision=runtime_decision,
        veto=gate_decision == "veto" or runtime_decision == "HARD_BLOCK",
        risk_level=_clean_str(selected.get("risk_level") or context.get("risk_level"), "unknown"),
        ambiguity=_clean_str(context.get("ambiguity") or audit_payload.get("ambiguity"), "unknown"),
        evidence_tier=_clean_str(selected.get("evidence_tier"), "unknown"),
        identification_tier=_clean_str(selected.get("identification_tier"), "unknown"),
        confidence=_clean_str(selected.get("confidence"), "unknown"),
        reason_codes=[str(code) for code in selected.get("reason_codes", []) or []],
        reason=_clean_str(selected.get("reason")),
        source=_clean_str(input_package.get("source") or llm_response.get("source") or "amantia_online"),
        payload={
            "mode": run_result.get("mode", "online"),
            "selected": selected,
            "evaluated_actions": list(run_result.get("evaluated_actions", []) or []),
            "llm_response": llm_response,
            "notes": list(run_result.get("notes", []) or []),
        },
    )


def append_decision(run_result: Mapping[str, Any], path: str | Path = DEFAULT_AUDIT_LOG_PATH) -> Dict[str, Any]:
    return AuditLog(path).append_decision_run(run_result)


def append_outcome(
    *,
    path: str | Path = DEFAULT_AUDIT_LOG_PATH,
    request_id: str = "",
    event_id: str = "",
    selected_action: str = "",
    outcome: str = "unknown",
    user_satisfaction: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return AuditLog(path).append_outcome(
        request_id=request_id,
        event_id=event_id,
        selected_action=selected_action,
        outcome=outcome,
        user_satisfaction=user_satisfaction,
        metadata=metadata,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Append/read Amantia Learning Loop audit events.")
    sub = parser.add_subparsers(dest="command", required=True)

    add_decision = sub.add_parser("append-decision", help="Append an OperationalBrain/propose_and_decide result JSON.")
    add_decision.add_argument("--input", required=True)
    add_decision.add_argument("--log", default=DEFAULT_AUDIT_LOG_PATH)

    add_outcome = sub.add_parser("append-outcome", help="Append an outcome event linked to a decision.")
    add_outcome.add_argument("--log", default=DEFAULT_AUDIT_LOG_PATH)
    add_outcome.add_argument("--request-id", default="")
    add_outcome.add_argument("--event-id", default="")
    add_outcome.add_argument("--selected-action", default="")
    add_outcome.add_argument("--outcome", required=True)
    add_outcome.add_argument("--user-satisfaction", type=float, default=None)

    read = sub.add_parser("read", help="Read recent audit events.")
    read.add_argument("--log", default=DEFAULT_AUDIT_LOG_PATH)
    read.add_argument("--max-events", type=int, default=20)

    args = parser.parse_args(argv)

    if args.command == "append-decision":
        run_result = json.loads(Path(args.input).read_text(encoding="utf-8"))
        event = append_decision(run_result, path=args.log)
        print(json.dumps({"status": "ok", "log": args.log, "event_id": event.get("event_id")}, indent=2))
        return 0

    if args.command == "append-outcome":
        event = append_outcome(
            path=args.log,
            request_id=args.request_id,
            event_id=args.event_id,
            selected_action=args.selected_action,
            outcome=args.outcome,
            user_satisfaction=args.user_satisfaction,
        )
        print(json.dumps({"status": "ok", "log": args.log, "event_id": event.get("event_id")}, indent=2))
        return 0

    if args.command == "read":
        events = AuditLog(args.log).read_events(max_events=args.max_events)
        print(json.dumps({"status": "ok", "log": args.log, "events": events}, indent=2, ensure_ascii=False))
        return 0

    return 2


__all__ = [
    "DEFAULT_AUDIT_LOG_PATH",
    "AuditEvent",
    "AuditLog",
    "append_decision",
    "append_outcome",
    "build_decision_event",
]


if __name__ == "__main__":
    raise SystemExit(main())

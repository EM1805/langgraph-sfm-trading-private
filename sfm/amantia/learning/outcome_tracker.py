from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .audit_log import DEFAULT_AUDIT_LOG_PATH, AuditEvent, AuditLog, utc_now_iso

SUCCESS_OUTCOMES = {"task_success", "resolved", "success", "helpful", "completed"}
FAILURE_OUTCOMES = {"task_failure", "failed", "failure", "not_resolved", "unhelpful", "abandoned"}
HARM_OUTCOMES = {"harm_event", "unsafe_outcome", "data_loss", "wrong_action", "policy_violation"}
UNKNOWN_OUTCOMES = {"", "unknown", "pending", "not_observed"}


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _clean_lower(value: Any, default: str = "unknown") -> str:
    text = _clean_str(value, default).lower()
    return text or default


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def infer_success(outcome: str, explicit_success: bool | None = None) -> bool | None:
    """Infer whether an observed outcome was successful.

    Returns ``None`` when the outcome is not informative enough for estimation.
    """

    if explicit_success is not None:
        return bool(explicit_success)
    label = _clean_lower(outcome, "unknown")
    if label in SUCCESS_OUTCOMES:
        return True
    if label in FAILURE_OUTCOMES or label in HARM_OUTCOMES:
        return False
    if label in UNKNOWN_OUTCOMES:
        return None
    return None


def infer_harm(outcome: str, explicit_harm: bool | None = None) -> bool | None:
    if explicit_harm is not None:
        return bool(explicit_harm)
    label = _clean_lower(outcome, "unknown")
    if label in HARM_OUTCOMES:
        return True
    if label in SUCCESS_OUTCOMES or label in FAILURE_OUTCOMES:
        return False
    if label in UNKNOWN_OUTCOMES:
        return None
    return None


@dataclass
class OutcomeRecord:
    """Joined decision + observed outcome row for offline learning."""

    decision_event_id: str = ""
    outcome_event_id: str = ""
    request_id: str = ""
    timestamp: str = ""
    outcome_timestamp: str = ""

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

    outcome: str = "unknown"
    success: bool | None = None
    harm: bool | None = None
    user_satisfaction: float | None = None
    latency_ms: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OutcomeTracker:
    """Link online decisions to later outcomes, stdlib-only.

    This module is intentionally not an estimator. It prepares clean learning
    records that future Estimation/RCT/fine-tuning modules can consume offline.
    """

    def __init__(self, path: str | Path = DEFAULT_AUDIT_LOG_PATH) -> None:
        self.path = Path(path)
        self.audit_log = AuditLog(self.path)

    def find_decision(self, *, event_id: str = "", request_id: str = "") -> Dict[str, Any] | None:
        event_id = _clean_str(event_id)
        request_id = _clean_str(request_id)
        for event in reversed(self.audit_log.read_events()):
            if event.get("event_type") != "decision":
                continue
            if event_id and event.get("event_id") == event_id:
                return event
            if request_id and event.get("request_id") == request_id:
                return event
        return None

    def record_outcome(
        self,
        *,
        decision_event_id: str = "",
        request_id: str = "",
        selected_action: str = "",
        outcome: str = "unknown",
        success: bool | None = None,
        harm: bool | None = None,
        user_satisfaction: float | None = None,
        latency_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Append an outcome event linked to a previous decision.

        A caller may link by ``decision_event_id`` or ``request_id``. If the
        matching decision is found, missing fields such as ``selected_action``
        are inherited to make the log easier to join later.
        """

        decision = self.find_decision(event_id=decision_event_id, request_id=request_id)
        linked_id = decision_event_id or _clean_str(decision.get("event_id") if decision else "")
        linked_request_id = request_id or _clean_str(decision.get("request_id") if decision else "")
        chosen_action = selected_action or _clean_str(decision.get("selected_action") if decision else "")
        outcome_label = _clean_lower(outcome, "unknown")
        success_value = infer_success(outcome_label, success)
        harm_value = infer_harm(outcome_label, harm)

        event = AuditEvent(
            event_type="outcome",
            request_id=linked_request_id,
            selected_action=chosen_action,
            outcome=outcome_label,
            user_satisfaction=user_satisfaction,
            payload={
                "linked_event_id": linked_id,
                "success": success_value,
                "harm": harm_value,
                "latency_ms": latency_ms,
                "metadata": dict(metadata or {}),
            },
        )
        return self.audit_log.append_event(event)

    def build_records(self) -> List[OutcomeRecord]:
        events = self.audit_log.read_events()
        decisions_by_id: Dict[str, Dict[str, Any]] = {}
        decisions_by_request: Dict[str, Dict[str, Any]] = {}
        outcomes: List[Dict[str, Any]] = []

        for event in events:
            if event.get("event_type") == "decision":
                event_id = _clean_str(event.get("event_id"))
                request_id = _clean_str(event.get("request_id"))
                if event_id:
                    decisions_by_id[event_id] = event
                if request_id:
                    decisions_by_request[request_id] = event
            elif event.get("event_type") == "outcome":
                outcomes.append(event)

        records: List[OutcomeRecord] = []
        for outcome_event in outcomes:
            payload = _as_dict(outcome_event.get("payload"))
            linked_id = _clean_str(payload.get("linked_event_id"))
            request_id = _clean_str(outcome_event.get("request_id"))
            decision = decisions_by_id.get(linked_id) or decisions_by_request.get(request_id) or {}

            decision_payload = _as_dict(decision.get("payload"))
            outcome_label = _clean_lower(outcome_event.get("outcome"), "unknown")
            success_value = infer_success(outcome_label, _as_bool(payload.get("success"), None))
            harm_value = infer_harm(outcome_label, _as_bool(payload.get("harm"), None))

            records.append(
                OutcomeRecord(
                    decision_event_id=_clean_str(decision.get("event_id") or linked_id),
                    outcome_event_id=_clean_str(outcome_event.get("event_id")),
                    request_id=_clean_str(decision.get("request_id") or request_id),
                    timestamp=_clean_str(decision.get("timestamp")),
                    outcome_timestamp=_clean_str(outcome_event.get("timestamp")),
                    user_message=_clean_str(decision.get("user_message")),
                    candidate_actions=[str(x) for x in decision.get("candidate_actions", []) or []],
                    selected_action=_clean_str(outcome_event.get("selected_action") or decision.get("selected_action")),
                    gate_decision=_clean_str(decision.get("gate_decision")),
                    runtime_decision=_clean_str(decision.get("runtime_decision")),
                    veto=bool(decision.get("veto", False)),
                    risk_level=_clean_str(decision.get("risk_level"), "unknown"),
                    ambiguity=_clean_str(decision.get("ambiguity"), "unknown"),
                    evidence_tier=_clean_str(decision.get("evidence_tier"), "unknown"),
                    identification_tier=_clean_str(decision.get("identification_tier"), "unknown"),
                    confidence=_clean_str(decision.get("confidence"), "unknown"),
                    reason_codes=[str(x) for x in decision.get("reason_codes", []) or []],
                    outcome=outcome_label,
                    success=success_value,
                    harm=harm_value,
                    user_satisfaction=_as_float(outcome_event.get("user_satisfaction")),
                    latency_ms=_as_float(payload.get("latency_ms")),
                    metadata={
                        "outcome_metadata": _as_dict(payload.get("metadata")),
                        "decision_payload": decision_payload,
                    },
                )
            )

        return records

    def summarize(self) -> Dict[str, Any]:
        records = self.build_records()
        by_action: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "n": 0,
            "success_n": 0,
            "success_observed_n": 0,
            "harm_n": 0,
            "harm_observed_n": 0,
            "satisfaction_sum": 0.0,
            "satisfaction_n": 0,
        })
        by_decision: Dict[str, int] = defaultdict(int)

        for record in records:
            action = record.selected_action or "unknown"
            row = by_action[action]
            row["n"] += 1
            if record.success is not None:
                row["success_observed_n"] += 1
                row["success_n"] += 1 if record.success else 0
            if record.harm is not None:
                row["harm_observed_n"] += 1
                row["harm_n"] += 1 if record.harm else 0
            if record.user_satisfaction is not None:
                row["satisfaction_sum"] += float(record.user_satisfaction)
                row["satisfaction_n"] += 1
            by_decision[record.gate_decision or "unknown"] += 1

        action_summary: Dict[str, Dict[str, Any]] = {}
        for action, row in sorted(by_action.items()):
            success_rate = None
            if row["success_observed_n"]:
                success_rate = row["success_n"] / row["success_observed_n"]
            harm_rate = None
            if row["harm_observed_n"]:
                harm_rate = row["harm_n"] / row["harm_observed_n"]
            avg_satisfaction = None
            if row["satisfaction_n"]:
                avg_satisfaction = row["satisfaction_sum"] / row["satisfaction_n"]

            action_summary[action] = {
                "n": row["n"],
                "success_n": row["success_n"],
                "success_observed_n": row["success_observed_n"],
                "success_rate": success_rate,
                "harm_n": row["harm_n"],
                "harm_observed_n": row["harm_observed_n"],
                "harm_rate": harm_rate,
                "avg_user_satisfaction": avg_satisfaction,
            }

        return {
            "generated_at": utc_now_iso(),
            "log_path": str(self.path),
            "records_n": len(records),
            "by_action": action_summary,
            "by_gate_decision": dict(sorted(by_decision.items())),
        }

    def export_records_jsonl(self, out_path: str | Path) -> int:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with out.open("w", encoding="utf-8") as handle:
            for record in self.build_records():
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        return count


def record_outcome(
    *,
    path: str | Path = DEFAULT_AUDIT_LOG_PATH,
    decision_event_id: str = "",
    request_id: str = "",
    selected_action: str = "",
    outcome: str = "unknown",
    success: bool | None = None,
    harm: bool | None = None,
    user_satisfaction: float | None = None,
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return OutcomeTracker(path).record_outcome(
        decision_event_id=decision_event_id,
        request_id=request_id,
        selected_action=selected_action,
        outcome=outcome,
        success=success,
        harm=harm,
        user_satisfaction=user_satisfaction,
        latency_ms=latency_ms,
        metadata=metadata,
    )


def build_outcome_records(path: str | Path = DEFAULT_AUDIT_LOG_PATH) -> List[Dict[str, Any]]:
    return [record.to_dict() for record in OutcomeTracker(path).build_records()]


def summarize_outcomes(path: str | Path = DEFAULT_AUDIT_LOG_PATH) -> Dict[str, Any]:
    return OutcomeTracker(path).summarize()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Track Amantia decision outcomes for offline learning.")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Append an outcome linked to a decision event/request.")
    record.add_argument("--log", default=DEFAULT_AUDIT_LOG_PATH)
    record.add_argument("--decision-event-id", default="")
    record.add_argument("--request-id", default="")
    record.add_argument("--selected-action", default="")
    record.add_argument("--outcome", required=True)
    record.add_argument("--success", default="")
    record.add_argument("--harm", default="")
    record.add_argument("--user-satisfaction", type=float, default=None)
    record.add_argument("--latency-ms", type=float, default=None)

    summary = sub.add_parser("summary", help="Print joined outcome summary.")
    summary.add_argument("--log", default=DEFAULT_AUDIT_LOG_PATH)

    export = sub.add_parser("export", help="Export joined decision/outcome rows as JSONL.")
    export.add_argument("--log", default=DEFAULT_AUDIT_LOG_PATH)
    export.add_argument("--out", default="out/learning/outcome_records.jsonl")

    args = parser.parse_args(argv)

    if args.command == "record":
        event = record_outcome(
            path=args.log,
            decision_event_id=args.decision_event_id,
            request_id=args.request_id,
            selected_action=args.selected_action,
            outcome=args.outcome,
            success=_as_bool(args.success, None),
            harm=_as_bool(args.harm, None),
            user_satisfaction=args.user_satisfaction,
            latency_ms=args.latency_ms,
        )
        print(json.dumps({"status": "ok", "log": args.log, "event_id": event.get("event_id")}, indent=2))
        return 0

    if args.command == "summary":
        print(json.dumps(summarize_outcomes(args.log), indent=2, ensure_ascii=False))
        return 0

    if args.command == "export":
        count = OutcomeTracker(args.log).export_records_jsonl(args.out)
        print(json.dumps({"status": "ok", "log": args.log, "out": args.out, "records": count}, indent=2))
        return 0

    return 2


__all__ = [
    "OutcomeRecord",
    "OutcomeTracker",
    "record_outcome",
    "build_outcome_records",
    "summarize_outcomes",
    "infer_success",
    "infer_harm",
]


if __name__ == "__main__":
    raise SystemExit(main())

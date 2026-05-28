from __future__ import annotations

"""Small JSONL storage helpers for autonomous trading experiments."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class JsonlTradeStore:
    """Append-only local audit store.

    The store intentionally uses JSONL so that experiments remain inspectable
    without a database server.  A future SaaS/dashboard can replace this with
    SQLite/Postgres while preserving the event shape.
    """

    path: str | Path = "trading_guard_audit.jsonl"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        record = {"ts": utc_now_iso(), **dict(event)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    out.append(item)
        return out

    def load_today_executed_trades(self, *, now_prefix: str | None = None) -> List[Dict[str, Any]]:
        """Return executed trade records whose timestamp starts with today's UTC date."""
        prefix = now_prefix or utc_now_iso()[:10]
        trades: List[Dict[str, Any]] = []
        for item in self.read_all():
            if not str(item.get("ts", "")).startswith(prefix):
                continue
            report = item.get("broker_execution_report") or item.get("execution_report") or {}
            if isinstance(report, dict) and report.get("executed"):
                trades.append(report)
        return trades

    def append_cycle(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        compact = {
            "event_type": "trading_guard_cycle",
            "mode": result.get("mode"),
            "symbol": result.get("symbol"),
            "gate_decision": result.get("gate_decision"),
            "gate_reason": result.get("gate_reason"),
            "violations": result.get("violations", []),
            "proposal": result.get("proposal", {}),
            "execution_report": result.get("execution_report", {}),
            "broker_execution_report": result.get("broker_execution_report", {}),
            "autonomous": result.get("autonomous", {}),
        }
        return self.append(compact)


class NullTradeStore:
    """No-op store for callers that do not want local files."""

    def append(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(event)

    def append_cycle(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(result)

    def read_all(self) -> List[Dict[str, Any]]:
        return []

    def load_today_executed_trades(self, *, now_prefix: str | None = None) -> List[Dict[str, Any]]:
        return []

from __future__ import annotations

"""Alert sinks for autonomous trading experiments."""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Mapping, Protocol
from urllib import parse, request


class AlertSink(Protocol):
    def send(self, message: str, *, payload: Mapping[str, Any] | None = None) -> None:
        ...


@dataclass
class ConsoleAlertSink:
    """Print alerts to stdout; safe default for local experiments."""

    prefix: str = "[langgraph-sfm-trading]"

    def send(self, message: str, *, payload: Mapping[str, Any] | None = None) -> None:
        print(f"{self.prefix} {message}")
        if payload:
            print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))


@dataclass
class MemoryAlertSink:
    """Test helper that keeps alerts in memory."""

    alerts: List[Dict[str, Any]]

    def __init__(self) -> None:
        self.alerts = []

    def send(self, message: str, *, payload: Mapping[str, Any] | None = None) -> None:
        self.alerts.append({"message": message, "payload": dict(payload or {})})


@dataclass
class TelegramAlertSink:
    """Minimal Telegram alert sink using the Bot API.

    Token/chat_id can be supplied directly or through environment variables:
    ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``.
    """

    token: str | None = None
    chat_id: str | None = None
    timeout: float = 8.0

    def send(self, message: str, *, payload: Mapping[str, Any] | None = None) -> None:
        token = self.token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = self.chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            raise RuntimeError("TelegramAlertSink requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        text = message
        if payload:
            text += "\n```json\n" + json.dumps(dict(payload), indent=2, sort_keys=True, default=str)[:3500] + "\n```"
        data = parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode("utf-8")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        req = request.Request(url, data=data, method="POST")
        with request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - explicit user-configured endpoint
            if resp.status >= 300:
                raise RuntimeError(f"Telegram alert failed with status {resp.status}")

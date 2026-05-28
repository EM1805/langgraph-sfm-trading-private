from __future__ import annotations

"""Kill switch utilities for autonomous trading loops."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


_TRUE_VALUES = {"1", "true", "yes", "on", "stop", "kill", "disabled"}


@dataclass(frozen=True)
class KillSwitch:
    """A file/env based kill switch.

    If the environment variable is truthy, or if the file exists and contains a
    truthy value, the runner blocks new orders.  An empty existing file also
    counts as a triggered kill switch because emergency files are often created
    with ``touch KILL_SWITCH``.
    """

    path: str | Path | None = "KILL_SWITCH"
    env_var: str = "SFM_TRADING_KILL_SWITCH"

    def is_triggered(self) -> bool:
        env_value = os.getenv(self.env_var, "").strip().lower()
        if env_value in _TRUE_VALUES:
            return True
        if not self.path:
            return False
        p = Path(self.path)
        if not p.exists():
            return False
        try:
            content = p.read_text(encoding="utf-8").strip().lower()
        except OSError:
            return True
        return content == "" or content in _TRUE_VALUES

    def reason(self) -> str:
        return f"Kill switch active via {self.env_var} or {self.path}"

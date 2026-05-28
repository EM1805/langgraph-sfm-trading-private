"""Deferred bridge from PCMCI Discovery artifacts to downstream layers.

Discovery keeps estimation/SCM/identification as downstream integrations.
They are loaded only when the output writer explicitly asks for their artifacts;
this does not create an alternate PCMCI execution mode.
"""
from __future__ import annotations

from typing import Any


def write_bridge_from_insights(*args: Any, **kwargs: Any) -> Any:
    try:
        from estimation_parts.discovery_bridge import write_bridge_from_insights as _impl
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional integration fallback
        raise RuntimeError(
            "estimation_parts.discovery_bridge.write_bridge_from_insights is unavailable"
        ) from exc
    return _impl(*args, **kwargs)


def write_scm_assets(*args: Any, **kwargs: Any) -> Any:
    try:
        from scm_parts.builder import write_scm_assets as _impl
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional integration fallback
        raise RuntimeError("SCM asset writer is unavailable") from exc
    return _impl(*args, **kwargs)


def write_identification_assets(*args: Any, **kwargs: Any) -> Any:
    try:
        from scm_parts.identifier import write_identification_assets as _impl
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional integration fallback
        raise RuntimeError("SCM identification writer is unavailable") from exc
    return _impl(*args, **kwargs)

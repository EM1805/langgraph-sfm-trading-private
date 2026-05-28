from __future__ import annotations

"""Safety checks for live Binance execution.

The module intentionally keeps live execution behind explicit acknowledgements.
Two profiles are available:

- locked: very small first-live caps.
- operative: less restrictive but still capped, spot-only, finite-cycle live mode.

Neither profile is financial advice or a profit engine.  The checks only reduce
configuration risk; they do not remove market risk.
"""

from dataclasses import dataclass
import os
from typing import Any, Mapping

LIVE_ACK_ENV = "SFM_TRADING_LIVE_ACK"
LIVE_ACK_VALUE = "I_UNDERSTAND_LIVE_TRADING_CAN_LOSE_MONEY"
LIVE_PROFILE_ENV = "SFM_TRADING_LIVE_PROFILE"
OPERATIVE_ACK_ENV = "SFM_TRADING_OPERATIVE_ACK"
OPERATIVE_ACK_VALUE = "I_ACCEPT_HIGHER_AUTONOMY_RISK"

# Backward-compatible locked constants.
LIVE_LOCKED_MAX_NOTIONAL_QUOTE = 10.0
LIVE_LOCKED_MAX_DAILY_NOTIONAL_QUOTE = 20.0
LIVE_LOCKED_MAX_OPEN_POSITION_QUOTE = 20.0
LIVE_LOCKED_MAX_TRADES_PER_DAY = 2
LIVE_LOCKED_MAX_CYCLES = 5
LIVE_LOCKED_MIN_INTERVAL_SECONDS = 300.0
LIVE_LOCKED_ALLOWED_SYMBOLS = {"BTC/USDT", "ETH/USDT"}

# Less restrictive but still bounded live profile.
LIVE_OPERATIVE_MAX_NOTIONAL_QUOTE = 25.0
LIVE_OPERATIVE_MAX_DAILY_NOTIONAL_QUOTE = 100.0
LIVE_OPERATIVE_MAX_OPEN_POSITION_QUOTE = 100.0
LIVE_OPERATIVE_MAX_TRADES_PER_DAY = 12
LIVE_OPERATIVE_MAX_CYCLES = 96
LIVE_OPERATIVE_MIN_INTERVAL_SECONDS = 60.0
LIVE_OPERATIVE_ALLOWED_SYMBOLS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"}


class LiveLockedConfigError(RuntimeError):
    """Raised when live execution is requested without required guardrails."""


@dataclass(frozen=True)
class LiveGuardProfile:
    name: str
    max_notional_quote: float
    max_daily_notional_quote: float
    max_open_position_quote: float
    max_trades_per_day: int
    max_cycles: int
    min_interval_seconds: float
    allowed_symbols: set[str]
    requires_operative_ack: bool = False


LIVE_GUARD_PROFILES: dict[str, LiveGuardProfile] = {
    "locked": LiveGuardProfile(
        name="locked",
        max_notional_quote=LIVE_LOCKED_MAX_NOTIONAL_QUOTE,
        max_daily_notional_quote=LIVE_LOCKED_MAX_DAILY_NOTIONAL_QUOTE,
        max_open_position_quote=LIVE_LOCKED_MAX_OPEN_POSITION_QUOTE,
        max_trades_per_day=LIVE_LOCKED_MAX_TRADES_PER_DAY,
        max_cycles=LIVE_LOCKED_MAX_CYCLES,
        min_interval_seconds=LIVE_LOCKED_MIN_INTERVAL_SECONDS,
        allowed_symbols=set(LIVE_LOCKED_ALLOWED_SYMBOLS),
    ),
    "operative": LiveGuardProfile(
        name="operative",
        max_notional_quote=LIVE_OPERATIVE_MAX_NOTIONAL_QUOTE,
        max_daily_notional_quote=LIVE_OPERATIVE_MAX_DAILY_NOTIONAL_QUOTE,
        max_open_position_quote=LIVE_OPERATIVE_MAX_OPEN_POSITION_QUOTE,
        max_trades_per_day=LIVE_OPERATIVE_MAX_TRADES_PER_DAY,
        max_cycles=LIVE_OPERATIVE_MAX_CYCLES,
        min_interval_seconds=LIVE_OPERATIVE_MIN_INTERVAL_SECONDS,
        allowed_symbols=set(LIVE_OPERATIVE_ALLOWED_SYMBOLS),
        requires_operative_ack=True,
    ),
}


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def live_guard_profile(name: str | None = None) -> LiveGuardProfile:
    """Return a supported live guard profile.

    ``name`` defaults to ``SFM_TRADING_LIVE_PROFILE`` and then ``locked``.
    """
    requested = (name or os.getenv(LIVE_PROFILE_ENV, "locked") or "locked").strip().lower()
    if requested not in LIVE_GUARD_PROFILES:
        raise LiveLockedConfigError(
            "Unsupported live profile. Use one of: " + ", ".join(sorted(LIVE_GUARD_PROFILES))
        )
    return LIVE_GUARD_PROFILES[requested]


def validate_live_locked_config(
    *,
    mode: str,
    enable_real_execution: bool,
    live_ack_confirmed: bool,
    risk_policy: Mapping[str, Any] | None,
    max_cycles: int,
    interval_seconds: float,
    symbol: str,
    live_profile: str | None = None,
    operative_ack_confirmed: bool = False,
) -> None:
    """Validate live execution guardrails.

    The validator is a no-op unless ``mode='live'`` and real execution is
    enabled.  In live mode it requires an explicit live acknowledgement and
    enforces profile-specific caps.  ``operative`` is less restrictive than
    ``locked`` but still finite-cycle, spot-only, withdrawal-free, and bounded.
    """
    if mode != "live" or not enable_real_execution:
        return

    profile = live_guard_profile(live_profile)
    policy = dict(risk_policy or {})
    env_ack = os.getenv(LIVE_ACK_ENV, "")
    if not live_ack_confirmed or env_ack != LIVE_ACK_VALUE:
        raise LiveLockedConfigError(
            "Refusing live execution: pass --live-ack and set "
            f"{LIVE_ACK_ENV}={LIVE_ACK_VALUE}"
        )
    if profile.requires_operative_ack:
        env_operative_ack = os.getenv(OPERATIVE_ACK_ENV, "")
        if not operative_ack_confirmed or env_operative_ack != OPERATIVE_ACK_VALUE:
            raise LiveLockedConfigError(
                "Refusing operative live execution: pass --operative-ack and set "
                f"{OPERATIVE_ACK_ENV}={OPERATIVE_ACK_VALUE}"
            )
    if not bool(policy.get("allow_live_trading", False)):
        raise LiveLockedConfigError("Refusing live execution: policy allow_live_trading must be true.")
    if bool(policy.get("allow_margin", False)) or bool(policy.get("allow_futures", False)):
        raise LiveLockedConfigError("Refusing live execution: margin/futures must remain disabled.")
    if bool(policy.get("allow_withdrawals", False)):
        raise LiveLockedConfigError("Refusing live execution: withdrawal capability must remain disabled.")
    if str(symbol) not in profile.allowed_symbols:
        raise LiveLockedConfigError(
            f"Refusing live execution: {profile.name} profile only allows "
            + ", ".join(sorted(profile.allowed_symbols))
        )

    max_notional = _as_float(policy.get("max_notional_quote"), profile.max_notional_quote)
    max_daily = _as_float(policy.get("max_daily_notional_quote"), profile.max_daily_notional_quote)
    max_open = _as_float(policy.get("max_open_position_quote"), profile.max_open_position_quote)
    max_trades = _as_int(policy.get("max_trades_per_day"), profile.max_trades_per_day)

    if max_notional <= 0 or max_daily <= 0 or max_open <= 0:
        raise LiveLockedConfigError("Refusing live execution: notional limits must be positive.")
    if max_notional > profile.max_notional_quote:
        raise LiveLockedConfigError(
            f"Refusing live execution: max_notional_quote must be <= {profile.max_notional_quote} for {profile.name}."
        )
    if max_daily > profile.max_daily_notional_quote:
        raise LiveLockedConfigError(
            f"Refusing live execution: max_daily_notional_quote must be <= {profile.max_daily_notional_quote} for {profile.name}."
        )
    if max_open > profile.max_open_position_quote:
        raise LiveLockedConfigError(
            f"Refusing live execution: max_open_position_quote must be <= {profile.max_open_position_quote} for {profile.name}."
        )
    if max_trades > profile.max_trades_per_day:
        raise LiveLockedConfigError(
            f"Refusing live execution: max_trades_per_day must be <= {profile.max_trades_per_day} for {profile.name}."
        )
    if max_cycles <= 0:
        raise LiveLockedConfigError("Refusing live execution: endless live cycles are disabled.")
    if max_cycles > profile.max_cycles:
        raise LiveLockedConfigError(f"Refusing live execution: cycles must be <= {profile.max_cycles} for {profile.name}.")
    if max_cycles > 1 and float(interval_seconds) < profile.min_interval_seconds:
        raise LiveLockedConfigError(
            f"Refusing live execution: interval_seconds must be >= {profile.min_interval_seconds} when cycles > 1 for {profile.name}."
        )

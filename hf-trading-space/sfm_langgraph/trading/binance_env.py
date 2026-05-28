from __future__ import annotations

"""Binance environment variable helpers for testnet/live configuration."""

import os
from typing import Tuple


def binance_env_credentials(*, mode: str = "testnet") -> Tuple[str, str]:
    """Return Binance credentials from environment variables.

    Testnet and live variables are intentionally separate.  Live mode does not
    fall back to generic Binance variables; this reduces the chance of using a
    real key accidentally when the caller intended testnet.
    """
    if mode == "testnet":
        key = os.getenv("BINANCE_TESTNET_API_KEY") or os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_TESTNET_API_SECRET") or os.getenv("BINANCE_API_SECRET", "")
    elif mode == "live":
        key = os.getenv("BINANCE_LIVE_API_KEY", "")
        secret = os.getenv("BINANCE_LIVE_API_SECRET", "")
    else:
        key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_API_SECRET", "")
    return key, secret


def require_binance_env_credentials(*, mode: str = "testnet") -> Tuple[str, str]:
    """Return Binance credentials or raise a clear setup error."""
    key, secret = binance_env_credentials(mode=mode)
    if not key or not secret:
        preferred = "BINANCE_TESTNET_API_KEY/BINANCE_TESTNET_API_SECRET" if mode == "testnet" else ("BINANCE_LIVE_API_KEY/BINANCE_LIVE_API_SECRET" if mode == "live" else "BINANCE_API_KEY/BINANCE_API_SECRET")
        raise RuntimeError(f"Missing Binance credentials. Set {preferred} in your environment or .env file.")
    return key, secret

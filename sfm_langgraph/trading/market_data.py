from __future__ import annotations

"""Market/account data providers for autonomous trading experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol

from .binance_env import binance_env_credentials
from .graph import DEFAULT_PRICES


class MarketDataProvider(Protocol):
    def fetch_recent_prices(self, symbol: str, *, limit: int = 30) -> List[float]:
        ...

    def fetch_account_snapshot(self) -> Dict[str, Any]:
        ...

    def fetch_exchange_constraints(self, symbol: str) -> Mapping[str, Any]:
        ...


@dataclass
class StaticMarketDataProvider:
    """Deterministic provider for paper tests and offline demos."""

    prices: List[float] = field(default_factory=lambda: list(DEFAULT_PRICES))
    account_snapshot: Dict[str, Any] = field(default_factory=lambda: {"source": "static", "balances": {}})
    exchange_constraints: Mapping[str, Any] = field(default_factory=dict)

    def fetch_recent_prices(self, symbol: str, *, limit: int = 30) -> List[float]:
        data = list(self.prices)
        return data[-limit:] if limit and len(data) > limit else data

    def fetch_account_snapshot(self) -> Dict[str, Any]:
        return dict(self.account_snapshot)

    def fetch_exchange_constraints(self, symbol: str) -> Mapping[str, Any]:
        return dict(self.exchange_constraints)


class CCXTBinanceMarketDataProvider:
    """Binance market/account provider through ccxt.

    This provider reads data only.  Order execution is handled separately by
    ``ExecutionEngine`` and remains gated.
    """

    def __init__(self, *, mode: str = "testnet", api_key: str = "", api_secret: str = "", timeframe: str = "1m") -> None:
        try:
            import ccxt  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install ccxt to use CCXTBinanceMarketDataProvider: pip install ccxt") from exc
        self.mode = mode
        self.timeframe = timeframe
        if not api_key or not api_secret:
            api_key, api_secret = binance_env_credentials(mode=mode)
        self.exchange = ccxt.binance({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True, "options": {"defaultType": "spot"}})
        if mode == "testnet":
            self.exchange.set_sandbox_mode(True)

    def fetch_recent_prices(self, symbol: str, *, limit: int = 30) -> List[float]:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=limit)
        return [float(row[4]) for row in ohlcv]

    def fetch_account_snapshot(self) -> Dict[str, Any]:
        try:
            balance = self.exchange.fetch_balance()
        except Exception as exc:  # pragma: no cover - network dependent
            return {"source": "ccxt_binance", "error": str(exc), "balances": {}}
        return {"source": "ccxt_binance", "balances": balance.get("total", {}), "raw": balance}

    def fetch_exchange_constraints(self, symbol: str) -> Mapping[str, Any]:
        try:
            self.exchange.load_markets()
            market = self.exchange.market(symbol)
        except Exception:  # pragma: no cover - network dependent
            return {}
        return market or {}

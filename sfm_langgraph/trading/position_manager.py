from __future__ import annotations

"""Position/exposure helpers for the autonomous trading runner."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    base_asset: str
    quote_asset: str
    base_free: float = 0.0
    quote_free: float = 0.0
    open_position_quote: float = 0.0
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "base_free": self.base_free,
            "quote_free": self.quote_free,
            "open_position_quote": self.open_position_quote,
            "source": self.source,
        }


class PositionManager:
    """Compute minimal spot exposure state from account snapshots or trade logs."""

    @staticmethod
    def split_symbol(symbol: str) -> tuple[str, str]:
        if "/" in symbol:
            base, quote = symbol.split("/", 1)
            return base.upper(), quote.upper()
        # Fallback for Binance-style BTCUSDT.
        for quote in ("USDT", "USDC", "BUSD", "USD", "EUR", "BTC", "ETH"):
            if symbol.upper().endswith(quote):
                return symbol.upper()[: -len(quote)], quote
        return symbol.upper(), "USDT"

    def from_account_snapshot(self, symbol: str, account: Mapping[str, Any] | None, *, last_price: float) -> PositionSnapshot:
        base, quote = self.split_symbol(symbol)
        account = account or {}
        balances = account.get("balances", account)
        base_free = 0.0
        quote_free = 0.0
        if isinstance(balances, Mapping):
            base_item = balances.get(base, {})
            quote_item = balances.get(quote, {})
            if isinstance(base_item, Mapping):
                base_free = _as_float(base_item.get("free", base_item.get("total", 0.0)))
            else:
                base_free = _as_float(base_item)
            if isinstance(quote_item, Mapping):
                quote_free = _as_float(quote_item.get("free", quote_item.get("total", 0.0)))
            else:
                quote_free = _as_float(quote_item)
        return PositionSnapshot(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            base_free=base_free,
            quote_free=quote_free,
            open_position_quote=max(0.0, base_free * max(0.0, last_price)),
            source=str(account.get("source", "account_snapshot")) if isinstance(account, Mapping) else "account_snapshot",
        )

    def from_trade_log(self, symbol: str, trades: Iterable[Mapping[str, Any]], *, last_price: float) -> PositionSnapshot:
        base, quote = self.split_symbol(symbol)
        base_qty = 0.0
        quote_cash = 0.0
        for item in trades:
            if str(item.get("symbol", symbol)) != symbol:
                continue
            qty = _as_float(item.get("quantity_base"), 0.0)
            notional = _as_float(item.get("notional_quote"), 0.0)
            side = str(item.get("side", "")).lower()
            if side == "buy":
                base_qty += qty
                quote_cash -= notional
            elif side == "sell":
                base_qty = max(0.0, base_qty - qty)
                quote_cash += notional
        return PositionSnapshot(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            base_free=max(0.0, base_qty),
            quote_free=quote_cash,
            open_position_quote=max(0.0, base_qty * max(0.0, last_price)),
            source="trade_log",
        )

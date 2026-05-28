from __future__ import annotations

"""Broker adapters for the trading guard.

The default broker is paper-only.  The optional CCXT adapter is deliberately
thin and guarded by the policy layer; it requires explicit credentials and live
acknowledgement before real orders can be submitted.
"""

import os
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from .binance_env import binance_env_credentials
from .types import ExecutionReport, MarketSnapshot, TradeProposal, TradingMode


class PaperBroker:
    """A deterministic paper broker that never touches real funds."""

    def __init__(self, *, mode: TradingMode = "paper", starting_cash_quote: float = 100.0) -> None:
        self.mode = mode
        self.cash_quote = float(starting_cash_quote)
        self.position_base = 0.0
        self.trades: List[Dict[str, Any]] = []

    def execute_order(self, proposal: TradeProposal, *, market: MarketSnapshot) -> ExecutionReport:
        if proposal.side == "hold":
            return ExecutionReport(
                executed=False,
                mode=self.mode,
                symbol=proposal.symbol,
                side="hold",
                status="not_submitted",
                reason="hold proposal",
            )
        price = float(market.last_price)
        notional = float(proposal.notional_quote)
        qty = float(proposal.quantity_base or (notional / price if price > 0 else 0.0))
        if proposal.side == "buy":
            self.cash_quote -= notional
            self.position_base += qty
        elif proposal.side == "sell":
            self.cash_quote += notional
            self.position_base = max(0.0, self.position_base - qty)
        report = ExecutionReport(
            executed=True,
            mode=self.mode,
            symbol=proposal.symbol,
            side=proposal.side,
            notional_quote=round(notional, 8),
            quantity_base=round(qty, 12),
            order_id=f"paper-{uuid4().hex[:12]}",
            status="filled_paper",
            reason="paper execution only; no real order submitted",
        )
        self.trades.append(report.to_dict())
        return report


class CCXTBinanceSpotBroker:
    """Optional Binance spot broker via ccxt.

    Use ``mode='testnet'`` first.  Use ``mode='live'`` only with API keys that
    have no withdrawal permissions and after the hard policy gate returns allow.
    """

    def __init__(
        self,
        *,
        mode: TradingMode = "testnet",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        enable_rate_limit: bool = True,
    ) -> None:
        if mode not in {"testnet", "live"}:
            raise ValueError("CCXTBinanceSpotBroker supports only testnet or live modes")
        try:
            import ccxt  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install ccxt to use CCXTBinanceSpotBroker: pip install ccxt") from exc

        self.mode = mode
        env_key, env_secret = binance_env_credentials(mode=mode)
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key or env_key,
                "secret": api_secret or env_secret,
                "enableRateLimit": enable_rate_limit,
                "options": {"defaultType": "spot"},
            }
        )
        if mode == "testnet":
            self.exchange.set_sandbox_mode(True)

    def fetch_recent_prices(self, symbol: str, *, timeframe: str = "1m", limit: int = 30) -> List[float]:
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return [float(row[4]) for row in ohlcv]

    def execute_order(self, proposal: TradeProposal, *, market: MarketSnapshot) -> ExecutionReport:
        if proposal.side == "hold":
            return ExecutionReport(
                executed=False,
                mode=self.mode,
                symbol=proposal.symbol,
                side="hold",
                status="not_submitted",
                reason="hold proposal",
            )
        price = float(market.last_price)
        amount = proposal.quantity_base or (float(proposal.notional_quote) / price if price > 0 else 0.0)
        if amount <= 0:
            return ExecutionReport(
                executed=False,
                mode=self.mode,
                symbol=proposal.symbol,
                side=proposal.side,
                status="blocked_client_side",
                reason="non-positive base amount",
            )
        order = self.exchange.create_order(proposal.symbol, "market", proposal.side, amount)
        return ExecutionReport(
            executed=True,
            mode=self.mode,
            symbol=proposal.symbol,
            side=proposal.side,
            notional_quote=float(proposal.notional_quote),
            quantity_base=float(amount),
            order_id=str(order.get("id", "")),
            status=str(order.get("status", "submitted")),
            reason="submitted through ccxt binance spot adapter",
            raw=dict(order),
        )

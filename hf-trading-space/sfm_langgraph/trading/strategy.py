from __future__ import annotations

"""Small deterministic strategies for the trading guard demo.

These strategies are intentionally basic.  They exist to produce proposals that
can be audited by the SFM/risk gate; they are not presented as profitable.
"""

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List

from .types import MarketSnapshot, TradeProposal, TradingRiskPolicy


def _last(values: List[float], n: int) -> List[float]:
    return values[-n:] if len(values) >= n else list(values)


@dataclass(frozen=True)
class SmaCrossoverStrategy:
    """Tiny SMA crossover strategy used for demos and testnet experiments."""

    short_window: int = 5
    long_window: int = 12
    buy_confidence: float = 0.62
    sell_confidence: float = 0.58
    default_stop_loss_pct: float = 1.0
    default_take_profit_pct: float = 2.0

    def snapshot(self, symbol: str, prices: Iterable[float], *, source: str = "paper") -> MarketSnapshot:
        series = [float(p) for p in prices if float(p) > 0]
        if not series:
            raise ValueError("At least one positive price is required")
        short_values = _last(series, self.short_window)
        long_values = _last(series, self.long_window)
        short_sma = mean(short_values)
        long_sma = mean(long_values)
        trend = "up" if short_sma > long_sma else ("down" if short_sma < long_sma else "flat")
        last = series[-1]
        if len(series) > 1:
            volatility = abs(max(series[-self.long_window :]) - min(series[-self.long_window :])) / last * 100.0
        else:
            volatility = 0.0
        return MarketSnapshot(
            symbol=symbol,
            last_price=round(last, 8),
            short_sma=round(short_sma, 8),
            long_sma=round(long_sma, 8),
            trend=trend,
            volatility_pct=round(volatility, 6),
            source=source,
        )

    def propose(self, snapshot: MarketSnapshot, policy: TradingRiskPolicy, *, open_position_quote: float = 0.0) -> TradeProposal:
        """Return a buy/sell/hold proposal under the configured policy limits."""
        if snapshot.trend == "up" and open_position_quote <= 0:
            return TradeProposal(
                symbol=snapshot.symbol,
                side="buy",
                notional_quote=min(policy.max_notional_quote, policy.max_open_position_quote),
                stop_loss_pct=self.default_stop_loss_pct,
                take_profit_pct=self.default_take_profit_pct,
                confidence=self.buy_confidence,
                rationale="Short SMA is above long SMA; propose small spot buy under hard risk limits.",
                source="SmaCrossoverStrategy",
            )
        if snapshot.trend == "down" and open_position_quote > 0:
            return TradeProposal(
                symbol=snapshot.symbol,
                side="sell",
                notional_quote=min(open_position_quote, policy.max_notional_quote),
                confidence=self.sell_confidence,
                rationale="Short SMA is below long SMA; propose reducing open paper position.",
                source="SmaCrossoverStrategy",
            )
        return TradeProposal(
            symbol=snapshot.symbol,
            side="hold",
            notional_quote=0.0,
            confidence=0.5,
            rationale="No actionable SMA signal under the configured position state.",
            source="SmaCrossoverStrategy",
        )

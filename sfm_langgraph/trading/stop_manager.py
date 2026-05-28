from __future__ import annotations

"""Stop-loss / take-profit helpers."""

from dataclasses import dataclass, replace
from typing import Any, Dict

from .types import TradeProposal, TradingRiskPolicy


@dataclass(frozen=True)
class StopPlan:
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    max_holding_minutes: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "max_holding_minutes": self.max_holding_minutes,
        }


class StopManager:
    """Attach policy-compliant defaults and calculate reference prices."""

    def with_required_defaults(self, proposal: TradeProposal, policy: TradingRiskPolicy) -> TradeProposal:
        if proposal.side != "buy":
            return proposal
        stop_loss = proposal.stop_loss_pct
        if policy.require_stop_loss and stop_loss is None:
            stop_loss = min(max(1.0, policy.min_stop_loss_pct), policy.max_stop_loss_pct)
        take_profit = proposal.take_profit_pct
        if take_profit is None:
            take_profit = min(2.0, policy.max_take_profit_pct)
        return replace(proposal, stop_loss_pct=stop_loss, take_profit_pct=take_profit)

    def plan(self, proposal: TradeProposal, *, entry_price: float, max_holding_minutes: int | None = None) -> StopPlan:
        stop_price = None
        take_price = None
        if proposal.side == "buy" and entry_price > 0:
            if proposal.stop_loss_pct is not None:
                stop_price = entry_price * (1.0 - float(proposal.stop_loss_pct) / 100.0)
            if proposal.take_profit_pct is not None:
                take_price = entry_price * (1.0 + float(proposal.take_profit_pct) / 100.0)
        return StopPlan(
            stop_loss_price=round(stop_price, 8) if stop_price is not None else None,
            take_profit_price=round(take_price, 8) if take_price is not None else None,
            max_holding_minutes=max_holding_minutes,
        )

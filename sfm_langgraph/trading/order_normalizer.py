from __future__ import annotations

"""Order normalization helpers for exchange constraints."""

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Any, Dict, Mapping, Optional

from .types import TradeProposal


@dataclass(frozen=True)
class OrderConstraints:
    """Small subset of exchange filters needed for spot market orders."""

    min_notional: float = 0.0
    step_size: float = 0.0
    min_quantity: float = 0.0
    tick_size: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedOrder:
    proposal: TradeProposal
    valid: bool
    reason: str = ""
    constraints: OrderConstraints = OrderConstraints()

    def to_dict(self) -> Dict[str, Any]:
        return {"proposal": self.proposal.to_dict(), "valid": self.valid, "reason": self.reason, "constraints": self.constraints.to_dict()}


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    try:
        q = Decimal(str(step))
        v = Decimal(str(value))
        return float((v / q).to_integral_value(rounding=ROUND_DOWN) * q)
    except (InvalidOperation, ValueError):
        return value


def constraints_from_ccxt_market(market: Mapping[str, Any] | None) -> OrderConstraints:
    """Extract approximate constraints from a ccxt market structure."""
    if not market:
        return OrderConstraints()
    limits = market.get("limits", {}) if isinstance(market.get("limits"), Mapping) else {}
    amount = limits.get("amount", {}) if isinstance(limits.get("amount"), Mapping) else {}
    cost = limits.get("cost", {}) if isinstance(limits.get("cost"), Mapping) else {}
    precision = market.get("precision", {}) if isinstance(market.get("precision"), Mapping) else {}
    step = 0.0
    amount_precision = precision.get("amount")
    if isinstance(amount_precision, int) and amount_precision >= 0:
        step = 10 ** (-amount_precision)
    return OrderConstraints(
        min_notional=float(cost.get("min") or 0.0),
        step_size=float(step or 0.0),
        min_quantity=float(amount.get("min") or 0.0),
        tick_size=0.0,
    )


def normalize_market_order(proposal: TradeProposal, *, last_price: float, constraints: Optional[OrderConstraints] = None) -> NormalizedOrder:
    """Return a proposal normalized to basic exchange constraints."""
    constraints = constraints or OrderConstraints()
    if proposal.side == "hold":
        return NormalizedOrder(proposal=proposal, valid=False, reason="hold proposal", constraints=constraints)
    if last_price <= 0:
        return NormalizedOrder(proposal=proposal, valid=False, reason="non_positive_price", constraints=constraints)
    notional = float(proposal.notional_quote)
    if constraints.min_notional and notional < constraints.min_notional:
        return NormalizedOrder(proposal=proposal, valid=False, reason="below_exchange_min_notional", constraints=constraints)
    qty = proposal.quantity_base if proposal.quantity_base is not None else notional / float(last_price)
    qty = _floor_to_step(float(qty), constraints.step_size)
    if constraints.min_quantity and qty < constraints.min_quantity:
        return NormalizedOrder(proposal=proposal, valid=False, reason="below_exchange_min_quantity", constraints=constraints)
    if qty <= 0:
        return NormalizedOrder(proposal=proposal, valid=False, reason="non_positive_quantity_after_normalization", constraints=constraints)
    normalized = replace(proposal, quantity_base=qty)
    return NormalizedOrder(proposal=normalized, valid=True, reason="normalized", constraints=constraints)

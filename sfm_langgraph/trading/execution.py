from __future__ import annotations

"""Gated execution helpers for paper/testnet/live brokers."""

from dataclasses import dataclass
from typing import Any, Mapping

from .broker import CCXTBinanceSpotBroker, PaperBroker
from .order_normalizer import OrderConstraints, constraints_from_ccxt_market, normalize_market_order
from .policy import policy_from_mapping, proposal_from_mapping
from .types import ExecutionReport, MarketSnapshot, TradeProposal, TradingGateResult, TradingMode, TradingRiskPolicy


@dataclass
class ExecutionEngine:
    """Execute an already-approved proposal.

    The engine refuses to submit orders unless the upstream gate decision is
    ``allow``.  Live/testnet execution still requires explicit caller opt-in via
    ``enable_real_execution``.
    """

    mode: TradingMode = "paper"
    broker: Any | None = None
    enable_real_execution: bool = False
    exchange_constraints: Mapping[str, Any] | OrderConstraints | None = None

    def _broker(self) -> Any:
        if self.broker is not None:
            return self.broker
        if self.mode == "paper" or not self.enable_real_execution:
            return PaperBroker(mode="paper")
        return CCXTBinanceSpotBroker(mode=self.mode)

    def execute_if_allowed(
        self,
        *,
        proposal: TradeProposal,
        market: MarketSnapshot,
        gate: Mapping[str, Any] | TradingGateResult,
        policy: TradingRiskPolicy | Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        decision = gate.decision if isinstance(gate, TradingGateResult) else str(gate.get("decision", "block"))
        if decision != "allow":
            return ExecutionReport(
                executed=False,
                mode=self.mode,
                symbol=proposal.symbol,
                side=proposal.side,
                status="not_submitted",
                reason="gate decision is not allow",
            )
        policy_obj = policy if isinstance(policy, TradingRiskPolicy) else policy_from_mapping(policy)
        if self.mode in {"testnet", "live"} and not self.enable_real_execution:
            return ExecutionReport(
                executed=False,
                mode=self.mode,
                symbol=proposal.symbol,
                side=proposal.side,
                notional_quote=proposal.notional_quote,
                status="dry_run_only",
                reason="real execution disabled; set enable_real_execution=True after testnet validation",
            )
        constraints = self.exchange_constraints
        if not isinstance(constraints, OrderConstraints):
            constraints = constraints_from_ccxt_market(constraints)
        normalized = normalize_market_order(proposal, last_price=market.last_price, constraints=constraints)
        if not normalized.valid:
            return ExecutionReport(
                executed=False,
                mode=self.mode,
                symbol=proposal.symbol,
                side=proposal.side,
                notional_quote=proposal.notional_quote,
                status="blocked_client_side",
                reason=normalized.reason,
                raw={"constraints": normalized.constraints.to_dict()},
            )
        if self.mode == "live" and not policy_obj.allow_live_trading:
            return ExecutionReport(
                executed=False,
                mode=self.mode,
                symbol=proposal.symbol,
                side=proposal.side,
                notional_quote=proposal.notional_quote,
                status="blocked_client_side",
                reason="live trading disabled by policy",
            )
        return self._broker().execute_order(normalized.proposal, market=market)


def execute_cycle_result(
    result: Mapping[str, Any],
    *,
    broker: Any | None = None,
    enable_real_execution: bool = False,
    exchange_constraints: Mapping[str, Any] | OrderConstraints | None = None,
) -> ExecutionReport:
    """Convenience wrapper for executing a ``run_trading_guard_cycle`` result."""
    mode = str(result.get("mode", "paper"))
    if mode not in {"paper", "testnet", "live"}:
        mode = "paper"
    market = MarketSnapshot(**dict(result.get("market", {})))
    proposal = proposal_from_mapping(result.get("proposal"), default_symbol=str(result.get("symbol", "BTC/USDT")))
    return ExecutionEngine(
        mode=mode,  # type: ignore[arg-type]
        broker=broker,
        enable_real_execution=enable_real_execution,
        exchange_constraints=exchange_constraints,
    ).execute_if_allowed(
        proposal=proposal,
        market=market,
        gate=dict(result.get("gate", {})),
        policy=dict(result.get("risk_policy", {})),
    )

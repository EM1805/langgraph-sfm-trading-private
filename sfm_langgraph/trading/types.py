from __future__ import annotations

"""Typed objects for the experimental LangGraph-SFM trading guard.

The trading guard is a risk-control scaffold, not a profit system.  It is built
so that any real order must pass through deterministic policy checks before a
broker adapter can execute it.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

TradeSide = Literal["buy", "sell", "hold"]
TradingMode = Literal["paper", "testnet", "live"]
GateDecision = Literal["allow", "review", "block"]


@dataclass(frozen=True)
class TradingRiskPolicy:
    """Hard limits enforced before any proposed order can be executed."""

    mode: TradingMode = "paper"
    allowed_symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    max_notional_quote: float = 5.0
    max_daily_notional_quote: float = 20.0
    max_open_position_quote: float = 25.0
    max_trades_per_day: int = 3
    min_confidence: float = 0.55
    require_stop_loss: bool = True
    min_stop_loss_pct: float = 0.2
    max_stop_loss_pct: float = 3.0
    max_take_profit_pct: float = 8.0
    allow_margin: bool = False
    allow_futures: bool = False
    allow_live_trading: bool = False
    require_live_ack: bool = True
    live_ack_value: str = "I_UNDERSTAND_LIVE_TRADING_CAN_LOSE_MONEY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradeProposal:
    """One proposed spot order produced by a strategy node."""

    symbol: str
    side: TradeSide = "hold"
    notional_quote: float = 0.0
    quantity_base: Optional[float] = None
    order_type: str = "market"
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    confidence: float = 0.0
    rationale: str = ""
    source: str = "sfm_langgraph.trading"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketSnapshot:
    """Minimal market state used by the example strategy."""

    symbol: str
    last_price: float
    short_sma: Optional[float] = None
    long_sma: Optional[float] = None
    trend: str = "unknown"
    volatility_pct: Optional[float] = None
    source: str = "paper"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradingGateResult:
    """Decision from the hard risk gate plus optional SFM analysis."""

    decision: GateDecision
    reason: str
    violations: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    protected_side_effects: List[str] = field(default_factory=list)
    safety_recommendation: Dict[str, Any] = field(default_factory=dict)
    sfm_claim_level: str = "diagnostic_only"
    sfm_primary_intent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionReport:
    """Broker execution result, intentionally normalized for logging."""

    executed: bool
    mode: TradingMode
    symbol: str
    side: TradeSide
    notional_quote: float = 0.0
    quantity_base: Optional[float] = None
    order_id: str = ""
    status: str = "not_submitted"
    reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradingState(TypedDict, total=False):
    """LangGraph state used by the trading guard workflow."""

    mode: TradingMode
    symbol: str
    prices: List[float]
    market: Dict[str, Any]
    news_items: List[Dict[str, Any]]
    external_signals: List[Dict[str, Any]]
    use_llm_advisor: bool
    use_llm_strategy_planner: bool
    use_llm_strategy_refiner: bool
    llm_model: str
    llm_provider: str
    llm_market_view: Dict[str, Any]
    llm_strategy_plan: Dict[str, Any]
    sfm_strategy_critique: Dict[str, Any]
    revised_strategy_plan: Dict[str, Any]
    proposal: Dict[str, Any]
    risk_policy: Dict[str, Any]
    account_snapshot: Dict[str, Any]
    prior_trades_today: List[Dict[str, Any]]
    open_position_quote: float
    sfm_query: Dict[str, Any]
    sfm_analysis: Dict[str, Any]
    sfm_trace_events: List[Dict[str, Any]]
    sfm_monitor: Dict[str, Any]
    sfm_monitor_events: List[Dict[str, Any]]
    gate: Dict[str, Any]
    gate_decision: GateDecision
    gate_reason: str
    violations: List[str]
    execution_report: Dict[str, Any]
    audit_log: List[Dict[str, Any]]

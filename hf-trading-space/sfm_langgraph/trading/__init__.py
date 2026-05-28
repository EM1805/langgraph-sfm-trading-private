from __future__ import annotations

"""Experimental trading-guard utilities for LangGraph-SFM.

This package is a risk-control scaffold for paper/testnet trading experiments.
It is not financial advice and it is not a profit engine.
"""

from .advisor import GeminiNewsSignalAdvisor, HeuristicNewsSignalAdvisor, LLMNewsSignalAdvisor, MarketAdvisorView, MarketSignal, NewsItem
from .alerts import ConsoleAlertSink, MemoryAlertSink, TelegramAlertSink
from .binance_env import binance_env_credentials, require_binance_env_credentials
from .broker import CCXTBinanceSpotBroker, PaperBroker
from .execution import ExecutionEngine, execute_cycle_result
from .kill_switch import KillSwitch
from .market_data import CCXTBinanceMarketDataProvider, StaticMarketDataProvider
from .live_locked import (
    LIVE_ACK_ENV,
    LIVE_ACK_VALUE,
    LIVE_PROFILE_ENV,
    OPERATIVE_ACK_ENV,
    OPERATIVE_ACK_VALUE,
    LIVE_GUARD_PROFILES,
    LIVE_OPERATIVE_MAX_NOTIONAL_QUOTE,
    LIVE_OPERATIVE_MAX_DAILY_NOTIONAL_QUOTE,
    LIVE_OPERATIVE_MAX_OPEN_POSITION_QUOTE,
    LIVE_OPERATIVE_MAX_TRADES_PER_DAY,
    LIVE_OPERATIVE_MAX_CYCLES,
    LIVE_OPERATIVE_MIN_INTERVAL_SECONDS,
    LiveLockedConfigError,
    live_guard_profile,
    validate_live_locked_config,
)
from .order_normalizer import OrderConstraints, normalize_market_order
from .position_manager import PositionManager, PositionSnapshot
from .runner import AutonomousTradingConfig, AutonomousTradingRunner, run_autonomous_trading_once
from .storage import JsonlTradeStore, NullTradeStore
from .stop_manager import StopManager, StopPlan
from .graph import build_trading_guard_graph, run_trading_guard_cycle
from .policy import evaluate_trade_policy, policy_from_mapping, proposal_from_mapping
from .strategy import SmaCrossoverStrategy
from .strategy_planner import (
    HeuristicStrategyPlanner,
    LLMStrategyPlanner,
    StrategyRefiner,
    LLMStrategyPlan,
    SFMStrategyCritique,
    critique_strategy_with_sfm,
    proposal_from_strategy_plan,
)
from .types import (
    ExecutionReport,
    GateDecision,
    MarketSnapshot,
    TradeProposal,
    TradeSide,
    TradingGateResult,
    TradingMode,
    TradingRiskPolicy,
    TradingState,
)

__all__ = [
    "GeminiNewsSignalAdvisor",
    "HeuristicNewsSignalAdvisor",
    "LLMNewsSignalAdvisor",
    "MarketAdvisorView",
    "MarketSignal",
    "NewsItem",
    "ConsoleAlertSink",
    "MemoryAlertSink",
    "TelegramAlertSink",
    "CCXTBinanceSpotBroker",
    "ExecutionEngine",
    "execute_cycle_result",
    "KillSwitch",
    "CCXTBinanceMarketDataProvider",
    "binance_env_credentials",
    "require_binance_env_credentials",
    "StaticMarketDataProvider",
    "LIVE_ACK_ENV",
    "LIVE_ACK_VALUE",
    "LIVE_PROFILE_ENV",
    "OPERATIVE_ACK_ENV",
    "OPERATIVE_ACK_VALUE",
    "LIVE_GUARD_PROFILES",
    "LIVE_OPERATIVE_MAX_NOTIONAL_QUOTE",
    "LIVE_OPERATIVE_MAX_DAILY_NOTIONAL_QUOTE",
    "LIVE_OPERATIVE_MAX_OPEN_POSITION_QUOTE",
    "LIVE_OPERATIVE_MAX_TRADES_PER_DAY",
    "LIVE_OPERATIVE_MAX_CYCLES",
    "LIVE_OPERATIVE_MIN_INTERVAL_SECONDS",
    "LiveLockedConfigError",
    "live_guard_profile",
    "validate_live_locked_config",
    "OrderConstraints",
    "normalize_market_order",
    "PositionManager",
    "PositionSnapshot",
    "AutonomousTradingConfig",
    "AutonomousTradingRunner",
    "run_autonomous_trading_once",
    "JsonlTradeStore",
    "NullTradeStore",
    "StopManager",
    "StopPlan",
    "CCXTBinanceSpotBroker",
    "PaperBroker",
    "build_trading_guard_graph",
    "run_trading_guard_cycle",
    "evaluate_trade_policy",
    "policy_from_mapping",
    "proposal_from_mapping",
    "SmaCrossoverStrategy",
    "HeuristicStrategyPlanner",
    "LLMStrategyPlanner",
    "StrategyRefiner",
    "LLMStrategyPlan",
    "SFMStrategyCritique",
    "critique_strategy_with_sfm",
    "proposal_from_strategy_plan",
    "ExecutionReport",
    "GateDecision",
    "MarketSnapshot",
    "TradeProposal",
    "TradeSide",
    "TradingGateResult",
    "TradingMode",
    "TradingRiskPolicy",
    "TradingState",
]

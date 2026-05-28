from __future__ import annotations

"""Autonomous runner for LangGraph-SFM trading guard experiments."""

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .alerts import AlertSink, ConsoleAlertSink
from .execution import execute_cycle_result
from .graph import DEFAULT_PRICES, run_trading_guard_cycle
from .kill_switch import KillSwitch
from .live_locked import validate_live_locked_config
from .market_data import MarketDataProvider, StaticMarketDataProvider
from .position_manager import PositionManager
from .storage import JsonlTradeStore, NullTradeStore
from .types import TradingMode


@dataclass
class AutonomousTradingConfig:
    """Configuration for one autonomous trading loop."""

    mode: TradingMode = "paper"
    symbol: str = "BTC/USDT"
    interval_seconds: float = 60.0
    max_cycles: int = 1
    risk_policy: Dict[str, Any] = field(default_factory=dict)
    news_items: List[Dict[str, Any]] = field(default_factory=list)
    external_signals: List[Dict[str, Any]] = field(default_factory=list)
    use_llm_advisor: bool = False
    use_llm_strategy_planner: bool = False
    use_llm_strategy_refiner: bool = False
    llm_model: str = ""
    llm_provider: str = "auto"
    price_limit: int = 30
    enable_real_execution: bool = False
    store_path: str = "trading_guard_audit.jsonl"
    kill_switch_path: str | None = "KILL_SWITCH"
    live_ack_confirmed: bool = False
    live_profile: str = "locked"
    operative_ack_confirmed: bool = False
    alert_on_review_or_block: bool = True
    alert_on_execution: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutonomousTradingRunner:
    """Run guarded trading cycles on a schedule.

    Defaults are paper-only.  For testnet/live, the runner still performs the
    SFM/risk gate first and submits to a broker only when ``enable_real_execution``
    is true and the gate returns ``allow``.
    """

    def __init__(
        self,
        config: AutonomousTradingConfig | None = None,
        *,
        market_data: MarketDataProvider | None = None,
        store: JsonlTradeStore | NullTradeStore | None = None,
        kill_switch: KillSwitch | None = None,
        alert_sink: AlertSink | None = None,
        broker: Any | None = None,
    ) -> None:
        self.config = config or AutonomousTradingConfig()
        validate_live_locked_config(
            mode=self.config.mode,
            enable_real_execution=bool(self.config.enable_real_execution),
            live_ack_confirmed=bool(self.config.live_ack_confirmed),
            risk_policy=self.config.risk_policy,
            max_cycles=int(self.config.max_cycles),
            interval_seconds=float(self.config.interval_seconds),
            symbol=self.config.symbol,
            live_profile=self.config.live_profile,
            operative_ack_confirmed=bool(self.config.operative_ack_confirmed),
        )
        self.market_data = market_data or StaticMarketDataProvider(prices=list(DEFAULT_PRICES))
        self.store = store if store is not None else JsonlTradeStore(self.config.store_path)
        self.kill_switch = kill_switch or KillSwitch(path=self.config.kill_switch_path)
        self.alert_sink = alert_sink or ConsoleAlertSink()
        self.broker = broker

    def _blocked_by_kill_switch(self, *, cycle_index: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "mode": self.config.mode,
            "symbol": self.config.symbol,
            "gate_decision": "block",
            "gate_reason": self.kill_switch.reason(),
            "violations": ["kill_switch_active"],
            "proposal": {"symbol": self.config.symbol, "side": "hold", "notional_quote": 0.0},
            "execution_report": {
                "executed": False,
                "mode": self.config.mode,
                "symbol": self.config.symbol,
                "side": "hold",
                "status": "not_submitted",
                "reason": self.kill_switch.reason(),
            },
            "autonomous": {"cycle_index": cycle_index, "kill_switch": True, "real_execution_enabled": False},
        }
        self.store.append_cycle(result)
        self.alert_sink.send("Trading runner blocked by kill switch", payload=result)
        return result

    def run_once(self, *, cycle_index: int = 1) -> Dict[str, Any]:
        if self.kill_switch.is_triggered():
            return self._blocked_by_kill_switch(cycle_index=cycle_index)

        prices = self.market_data.fetch_recent_prices(self.config.symbol, limit=self.config.price_limit)
        if not prices:
            prices = list(DEFAULT_PRICES)
        account_snapshot = self.market_data.fetch_account_snapshot()
        prior_trades = self.store.load_today_executed_trades()
        last_price = float(prices[-1])
        position = PositionManager().from_account_snapshot(self.config.symbol, account_snapshot, last_price=last_price)
        if position.open_position_quote <= 0 and prior_trades:
            position = PositionManager().from_trade_log(self.config.symbol, prior_trades, last_price=last_price)

        policy = {"mode": self.config.mode, **self.config.risk_policy}
        state = {
            "mode": self.config.mode,
            "symbol": self.config.symbol,
            "prices": prices,
            "risk_policy": policy,
            "news_items": list(self.config.news_items),
            "external_signals": list(self.config.external_signals),
            "use_llm_advisor": self.config.use_llm_advisor,
            "use_llm_strategy_planner": self.config.use_llm_strategy_planner,
            "use_llm_strategy_refiner": self.config.use_llm_strategy_refiner,
            "llm_model": self.config.llm_model,
            "llm_provider": self.config.llm_provider,
            "account_snapshot": account_snapshot,
            "prior_trades_today": prior_trades,
            "open_position_quote": position.open_position_quote,
            "audit_log": [],
        }
        result = dict(run_trading_guard_cycle(state))
        result["autonomous"] = {
            "cycle_index": cycle_index,
            "real_execution_enabled": bool(self.config.enable_real_execution),
            "position": position.to_dict(),
            "kill_switch": False,
        }

        broker_report = execute_cycle_result(
            result,
            broker=self.broker,
            enable_real_execution=self.config.enable_real_execution,
            exchange_constraints=self.market_data.fetch_exchange_constraints(self.config.symbol),
        )
        result["broker_execution_report"] = broker_report.to_dict()

        self.store.append_cycle(result)
        decision = str(result.get("gate_decision", "block"))
        if self.config.alert_on_review_or_block and decision in {"review", "block"}:
            self.alert_sink.send(f"Trading gate {decision}: {result.get('gate_reason')}", payload=result)
        elif self.config.alert_on_execution and broker_report.executed:
            self.alert_sink.send("Trading order executed", payload=result)
        return result

    def run_forever(self, *, max_cycles: Optional[int] = None) -> Iterable[Dict[str, Any]]:
        cycles = self.config.max_cycles if max_cycles is None else max_cycles
        index = 0
        while cycles <= 0 or index < cycles:
            index += 1
            yield self.run_once(cycle_index=index)
            if cycles <= 0 or index < cycles:
                time.sleep(max(0.0, self.config.interval_seconds))


def run_autonomous_trading_once(config: AutonomousTradingConfig | Mapping[str, Any] | None = None, **kwargs: Any) -> Dict[str, Any]:
    """Convenience function for a single autonomous cycle."""
    if config is None:
        cfg = AutonomousTradingConfig(**kwargs)
    elif isinstance(config, AutonomousTradingConfig):
        cfg = config
        if kwargs:
            cfg = AutonomousTradingConfig(**{**cfg.to_dict(), **kwargs})
    else:
        cfg = AutonomousTradingConfig(**{**dict(config), **kwargs})
    return AutonomousTradingRunner(cfg).run_once()

from __future__ import annotations

"""CLI for the autonomous LangGraph-SFM trading runner."""

import argparse
import json
import os
from typing import Any, Dict, List

from .broker import CCXTBinanceSpotBroker
from .binance_env import require_binance_env_credentials
from .market_data import CCXTBinanceMarketDataProvider
from .live_locked import (
    LIVE_ACK_ENV,
    LIVE_ACK_VALUE,
    LIVE_LOCKED_MAX_DAILY_NOTIONAL_QUOTE,
    LIVE_LOCKED_MAX_NOTIONAL_QUOTE,
    LIVE_LOCKED_MAX_OPEN_POSITION_QUOTE,
    LIVE_LOCKED_MAX_TRADES_PER_DAY,
    LIVE_LOCKED_MAX_CYCLES,
    LIVE_LOCKED_MIN_INTERVAL_SECONDS,
    LIVE_OPERATIVE_MAX_DAILY_NOTIONAL_QUOTE,
    LIVE_OPERATIVE_MAX_NOTIONAL_QUOTE,
    LIVE_OPERATIVE_MAX_OPEN_POSITION_QUOTE,
    LIVE_OPERATIVE_MAX_TRADES_PER_DAY,
    LIVE_OPERATIVE_MAX_CYCLES,
    LIVE_OPERATIVE_MIN_INTERVAL_SECONDS,
    LIVE_PROFILE_ENV,
    OPERATIVE_ACK_ENV,
    OPERATIVE_ACK_VALUE,
    validate_live_locked_config,
)
from .runner import AutonomousTradingConfig, AutonomousTradingRunner


def _parse_signal(raw: str) -> Dict[str, Any]:
    # format: name=value:direction:confidence
    try:
        left, direction, confidence = raw.split(":", 2)
        name, value = left.split("=", 1)
        return {"name": name, "value": value, "direction": direction, "confidence": float(confidence)}
    except ValueError:
        return {"name": "manual", "value": raw, "direction": "neutral", "confidence": 0.5}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the autonomous LangGraph-SFM Trading Guard")
    parser.add_argument("--mode", choices=["paper", "testnet", "live"], default="paper")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--interval-sec", type=float, default=60.0)
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles. Use 0 for endless loop.")
    parser.add_argument("--max-notional", type=float, default=5.0)
    parser.add_argument("--max-daily-notional", type=float, default=20.0)
    parser.add_argument("--max-open-position", type=float, default=20.0)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--allow-live", action="store_true", help="Allow live mode in policy. Still requires live env ack, --live-ack, and locked caps.")
    parser.add_argument("--enable-real-execution", action="store_true", help="Permit broker execution after gate allow. Paper remains default.")
    parser.add_argument("--use-binance-data", action="store_true", help="Use Binance/ccxt market and account data for testnet/live modes.")
    parser.add_argument("--timeframe", default="1m", help="OHLCV timeframe when --use-binance-data is enabled.")
    parser.add_argument("--testnet-ack", action="store_true", help="Required with --mode testnet --enable-real-execution to acknowledge sandbox orders.")
    parser.add_argument("--live-ack", action="store_true", help="Required with --mode live --enable-real-execution. Also set SFM_TRADING_LIVE_ACK.")
    parser.add_argument("--live-profile", choices=["locked", "operative"], default="locked", help="Live guard profile. locked is safest; operative allows more cycles/symbols/notional but needs a second ack.")
    parser.add_argument("--operative-ack", action="store_true", help="Required with --live-profile operative. Also set SFM_TRADING_OPERATIVE_ACK.")
    parser.add_argument("--use-llm-advisor", action="store_true")
    parser.add_argument("--use-llm-strategy", action="store_true", help="Let the LLM/news planner create a structured strategy before SFM critique and risk gating.")
    parser.add_argument("--use-llm-strategy-refiner", action="store_true", help="Let the LLM revise the strategy after SFM critique. Without this flag, refinement is deterministic.")
    parser.add_argument("--llm-provider", choices=["auto", "openai", "gemini"], default="auto")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--news", action="append", default=[], help="Manual news headline/summary. Repeatable.")
    parser.add_argument("--signal", action="append", default=[], help="Signal in name=value:direction:confidence format. Repeatable.")
    parser.add_argument("--store-path", default="trading_guard_audit.jsonl")
    parser.add_argument("--kill-switch-path", default="KILL_SWITCH")
    parser.add_argument("--json", action="store_true")
    return parser



def _validate_cli_live_locked(args: argparse.Namespace) -> None:
    """Fail fast before credentials or ccxt are touched for unsafe live runs."""
    if args.mode != "live" or not args.enable_real_execution:
        return
    policy = {
        "mode": args.mode,
        "allowed_symbols": [args.symbol],
        "max_notional_quote": args.max_notional,
        "max_daily_notional_quote": args.max_daily_notional,
        "max_open_position_quote": args.max_open_position,
        "max_trades_per_day": args.max_trades_per_day,
        "allow_live_trading": bool(args.allow_live),
        "allow_margin": False,
        "allow_futures": False,
        "allow_withdrawals": False,
    }
    try:
        validate_live_locked_config(
            mode=args.mode,
            enable_real_execution=True,
            live_ack_confirmed=bool(args.live_ack),
            risk_policy=policy,
            max_cycles=int(args.cycles),
            interval_seconds=float(args.interval_sec),
            symbol=args.symbol,
            live_profile=args.live_profile,
            operative_ack_confirmed=bool(args.operative_ack),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def live_locked_example_command() -> str:
    return (
        f"export {LIVE_ACK_ENV}={LIVE_ACK_VALUE}\n"
        "python -m sfm_langgraph.trading.autonomous_cli "
        "--mode live --use-binance-data --enable-real-execution --allow-live --live-ack "
        "--live-profile locked --symbol BTC/USDT --cycles 3 --interval-sec 900 "
        f"--max-notional {LIVE_LOCKED_MAX_NOTIONAL_QUOTE:g} "
        f"--max-daily-notional {LIVE_LOCKED_MAX_DAILY_NOTIONAL_QUOTE:g} "
        f"--max-open-position {LIVE_LOCKED_MAX_OPEN_POSITION_QUOTE:g} "
        f"--max-trades-per-day {LIVE_LOCKED_MAX_TRADES_PER_DAY:d} --json"
    )


def live_operative_example_command() -> str:
    return (
        f"export {LIVE_ACK_ENV}={LIVE_ACK_VALUE}\n"
        f"export {LIVE_PROFILE_ENV}=operative\n"
        f"export {OPERATIVE_ACK_ENV}={OPERATIVE_ACK_VALUE}\n"
        "python -m sfm_langgraph.trading.autonomous_cli "
        "--mode live --use-binance-data --enable-real-execution --allow-live --live-ack "
        "--live-profile operative --operative-ack --symbol BTC/USDT --cycles 24 --interval-sec 300 "
        f"--max-notional {LIVE_OPERATIVE_MAX_NOTIONAL_QUOTE:g} "
        f"--max-daily-notional {LIVE_OPERATIVE_MAX_DAILY_NOTIONAL_QUOTE:g} "
        f"--max-open-position {LIVE_OPERATIVE_MAX_OPEN_POSITION_QUOTE:g} "
        f"--max-trades-per-day {LIVE_OPERATIVE_MAX_TRADES_PER_DAY:d} --json"
    )

def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def main(argv: List[str] | None = None) -> int:
    _load_dotenv_if_available()
    args = build_parser().parse_args(argv)
    if args.mode == "testnet" and args.enable_real_execution and not args.testnet_ack and os.getenv("SFM_TRADING_TESTNET_ACK") != "I_UNDERSTAND_TESTNET_ORDERS":
        raise SystemExit("Refusing testnet execution: pass --testnet-ack or set SFM_TRADING_TESTNET_ACK=I_UNDERSTAND_TESTNET_ORDERS")
    _validate_cli_live_locked(args)
    policy = {
        "mode": args.mode,
        "allowed_symbols": [args.symbol],
        "max_notional_quote": args.max_notional,
        "max_daily_notional_quote": args.max_daily_notional,
        "max_open_position_quote": args.max_open_position,
        "max_trades_per_day": args.max_trades_per_day,
        "allow_live_trading": bool(args.allow_live),
        "allow_margin": False,
        "allow_futures": False,
        "allow_withdrawals": False,
        "live_ack_value": LIVE_ACK_VALUE,
        "live_profile": args.live_profile,
    }
    config = AutonomousTradingConfig(
        mode=args.mode,
        symbol=args.symbol,
        interval_seconds=args.interval_sec,
        max_cycles=args.cycles,
        risk_policy=policy,
        news_items=[{"title": item, "summary": item, "source": "cli"} for item in args.news],
        external_signals=[_parse_signal(item) for item in args.signal],
        use_llm_advisor=bool(args.use_llm_advisor),
        use_llm_strategy_planner=bool(args.use_llm_strategy),
        use_llm_strategy_refiner=bool(args.use_llm_strategy_refiner),
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        enable_real_execution=bool(args.enable_real_execution),
        store_path=args.store_path,
        kill_switch_path=args.kill_switch_path,
        live_ack_confirmed=bool(args.live_ack),
        live_profile=args.live_profile,
        operative_ack_confirmed=bool(args.operative_ack),
    )
    market_data = None
    broker = None
    if args.mode in {"testnet", "live"} and (args.use_binance_data or args.enable_real_execution):
        api_key, api_secret = require_binance_env_credentials(mode=args.mode)
        market_data = CCXTBinanceMarketDataProvider(mode=args.mode, api_key=api_key, api_secret=api_secret, timeframe=args.timeframe)
        broker = CCXTBinanceSpotBroker(mode=args.mode, api_key=api_key, api_secret=api_secret)
    runner = AutonomousTradingRunner(config, market_data=market_data, broker=broker)
    results = list(runner.run_forever())
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
    else:
        for result in results:
            print("LangGraph-SFM Autonomous Trading Runner")
            print("cycle:", result.get("autonomous", {}).get("cycle_index"))
            print("mode:", result.get("mode"), "symbol:", result.get("symbol"))
            print("gate:", result.get("gate_decision"), "-", result.get("gate_reason"))
            print("proposal:", json.dumps(result.get("proposal", {}), sort_keys=True))
            print("broker_execution:", json.dumps(result.get("broker_execution_report", {}), sort_keys=True))
            print("---")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

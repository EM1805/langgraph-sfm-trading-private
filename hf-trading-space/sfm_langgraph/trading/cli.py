from __future__ import annotations

"""CLI for the experimental LangGraph-SFM trading guard."""

import argparse
import json
import os
from typing import Any, Dict

from .graph import run_trading_guard_cycle


def _parse_prices(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_signal(text: str) -> Dict[str, Any]:
    """Parse name=value[:direction[:confidence]] into a signal dict."""
    name, _, rest = text.partition("=")
    value, *extra = rest.split(":") if rest else [""]
    direction = extra[0] if len(extra) >= 1 and extra[0] else "neutral"
    try:
        confidence = float(extra[1]) if len(extra) >= 2 else 0.5
    except ValueError:
        confidence = 0.5
    return {"name": name or "cli_signal", "value": value, "direction": direction, "confidence": confidence, "source": "cli"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one guarded LangGraph-SFM trading cycle.")
    parser.add_argument("--mode", choices=["paper", "testnet", "live"], default="paper")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--prices", default="", help="Comma-separated close prices for paper mode.")
    parser.add_argument("--max-notional", type=float, default=5.0)
    parser.add_argument("--max-daily-notional", type=float, default=20.0)
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--allow-live", action="store_true", help="Enable live mode policy. Still requires live ack env var.")
    parser.add_argument("--news", action="append", default=[], help="Add one news headline/summary for the advisor. Can be repeated.")
    parser.add_argument(
        "--signal",
        action="append",
        default=[],
        help="Add signal as name=value[:direction[:confidence]], e.g. rsi=72:bearish:0.7. Can be repeated.",
    )
    parser.add_argument("--use-llm-advisor", action="store_true", help="Use optional OpenAI-compatible LLM advisor; falls back to heuristic if unavailable.")
    parser.add_argument("--llm-model", default="", help="Optional model name, e.g. gpt-4o-mini.")
    parser.add_argument("--json", action="store_true", help="Print full JSON state.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy: Dict[str, Any] = {
        "mode": args.mode,
        "allowed_symbols": [args.symbol],
        "max_notional_quote": args.max_notional,
        "max_daily_notional_quote": args.max_daily_notional,
        "max_trades_per_day": args.max_trades_per_day,
        "allow_live_trading": bool(args.allow_live),
    }
    state: Dict[str, Any] = {
        "mode": args.mode,
        "symbol": args.symbol,
        "risk_policy": policy,
        "news_items": [{"title": item, "summary": item, "source": "cli"} for item in args.news],
        "external_signals": [_parse_signal(item) for item in args.signal],
        "use_llm_advisor": bool(args.use_llm_advisor),
        "prior_trades_today": [],
        "open_position_quote": 0.0,
    }
    if args.llm_model:
        state["llm_model"] = args.llm_model
    if args.prices:
        state["prices"] = _parse_prices(args.prices)
    if args.mode == "live" and args.allow_live:
        os.environ.setdefault("SFM_TRADING_LIVE_ACK", "")

    result = run_trading_guard_cycle(state)  # graph itself uses paper execution for safety
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("LangGraph-SFM Trading Guard")
        print("mode:", result.get("mode"))
        print("symbol:", result.get("symbol"))
        print("market_view:", json.dumps(result.get("llm_market_view", {}), sort_keys=True))
        print("proposal:", json.dumps(result.get("proposal", {}), sort_keys=True))
        print("gate:", json.dumps(result.get("gate", {}), sort_keys=True))
        print("execution:", json.dumps(result.get("execution_report", {}), sort_keys=True))
        print("\nNote: the packaged graph executes paper orders only. Use broker.CCXTBinanceSpotBroker explicitly after the gate allows.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

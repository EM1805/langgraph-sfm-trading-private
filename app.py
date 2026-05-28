from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from sfm_langgraph.trading.autonomous_cli import _parse_signal
from sfm_langgraph.trading.binance_env import require_binance_env_credentials
from sfm_langgraph.trading.broker import CCXTBinanceSpotBroker
from sfm_langgraph.trading.live_locked import LIVE_ACK_VALUE, OPERATIVE_ACK_VALUE
from sfm_langgraph.trading.market_data import CCXTBinanceMarketDataProvider
from sfm_langgraph.trading.runner import AutonomousTradingConfig, AutonomousTradingRunner

APP_TITLE = "🛡️ LangGraph-SFM Trading Private Control Panel"
CONFIRM_LIVE_PHRASE = "I UNDERSTAND LIVE TRADING CAN LOSE MONEY"
KILL_SWITCH_PATH = "KILL_SWITCH"
DEFAULT_STORE_PATH = "trading_guard_audit.jsonl"

ALLOWED_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]


def _secret_status() -> Dict[str, bool]:
    keys = [
        "GEMINI_API_KEY",
        "BINANCE_TESTNET_API_KEY",
        "BINANCE_TESTNET_API_SECRET",
        "BINANCE_LIVE_API_KEY",
        "BINANCE_LIVE_API_SECRET",
        "SFM_TRADING_LIVE_ACK",
        "SFM_TRADING_OPERATIVE_ACK",
    ]
    return {key: bool(os.getenv(key)) for key in keys}


def secrets_markdown() -> str:
    status = _secret_status()
    rows = ["| Secret | Status |", "|---|---:|"]
    for key, ok in status.items():
        rows.append(f"| `{key}` | {'✅ set' if ok else '❌ missing'} |")
    return "\n".join(rows)


def _parse_news(text: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            items.append({"title": line, "summary": line, "source": "hf_space_manual"})
    return items


def _parse_signals(text: str) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            signals.append(_parse_signal(line))
    return signals


def _tail_file(path: str, lines: int = 20) -> str:
    p = Path(path)
    if not p.exists():
        return "No audit log yet."
    data = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:]) if data else "Audit log is empty."


def _summarize(result: Dict[str, Any]) -> str:
    decision = result.get("gate_decision", "unknown")
    reason = result.get("gate_reason", "")
    proposal = result.get("proposal", {}) or {}
    execution = result.get("broker_execution_report", {}) or result.get("execution_report", {}) or {}
    llm_view = result.get("llm_market_view", {}) or {}
    strategy = result.get("llm_strategy_plan", {}) or {}
    critique = result.get("sfm_strategy_critique", {}) or {}
    revised = result.get("revised_strategy_plan", {}) or {}

    executed = execution.get("executed", False)
    status = execution.get("status", "not_submitted")
    warnings = result.get("violations", []) or []

    lines = [
        f"## Result: `{decision}`",
        f"**Reason:** {reason or 'n/a'}",
        "",
        "### Proposal",
        f"- Symbol: `{proposal.get('symbol', result.get('symbol', ''))}`",
        f"- Side: `{proposal.get('side', 'hold')}`",
        f"- Notional: `{proposal.get('notional_quote', 0)}`",
        f"- Stop loss: `{proposal.get('stop_loss_pct', 'n/a')}`",
        f"- Take profit: `{proposal.get('take_profit_pct', 'n/a')}`",
        "",
        "### Execution",
        f"- Executed: `{'yes' if executed else 'no'}`",
        f"- Status: `{status}`",
        f"- Execution reason: {execution.get('reason', 'n/a')}",
        "",
        "### Gemini / strategy / SFM",
        f"- LLM view provider: `{llm_view.get('provider', 'n/a')}`",
        f"- LLM risk level: `{llm_view.get('risk_level', 'n/a')}`",
        f"- LLM trade bias: `{llm_view.get('trade_bias', 'n/a')}`",
        f"- Initial strategy: `{strategy.get('strategy_name', 'n/a')}` → `{strategy.get('action', 'n/a')}`",
        f"- SFM critique: `{critique.get('decision', 'n/a')}` — {critique.get('reason', 'n/a')}",
        f"- Revised strategy: `{revised.get('strategy_name', 'n/a')}` → `{revised.get('action', 'n/a')}`",
    ]
    if warnings:
        lines += ["", "### Violations / warnings"]
        lines += [f"- `{item}`" for item in warnings]
    return "\n".join(lines)


def _validate_live_inputs(
    *,
    mode: str,
    enable_real_execution: bool,
    allow_live: bool,
    confirmation: str,
    symbol: str,
    cycles: int,
    interval_sec: float,
) -> None:
    if mode != "live" or not enable_real_execution:
        return
    if not allow_live:
        raise RuntimeError("Live execution blocked: check 'Allow live trading'.")
    if confirmation.strip() != CONFIRM_LIVE_PHRASE:
        raise RuntimeError(f"Live execution blocked: type exactly: {CONFIRM_LIVE_PHRASE}")
    if symbol not in ALLOWED_SYMBOLS:
        raise RuntimeError(f"Live execution blocked: symbol must be one of {ALLOWED_SYMBOLS}.")
    if cycles > 24:
        raise RuntimeError("Live execution blocked: this Space caps live runs at 24 cycles per button click.")
    if cycles > 1 and interval_sec < 60:
        raise RuntimeError("Live execution blocked: interval must be at least 60 seconds for multi-cycle live runs.")
    if os.getenv("SFM_TRADING_LIVE_ACK") != LIVE_ACK_VALUE:
        raise RuntimeError("Live execution blocked: missing/invalid SFM_TRADING_LIVE_ACK secret.")
    if os.getenv("SFM_TRADING_OPERATIVE_ACK") != OPERATIVE_ACK_VALUE:
        raise RuntimeError("Live execution blocked: missing/invalid SFM_TRADING_OPERATIVE_ACK secret.")
    require_binance_env_credentials(mode="live")


def run_guard(
    mode: str,
    symbol: str,
    cycles: int,
    interval_sec: float,
    max_notional: float,
    max_daily_notional: float,
    max_open_position: float,
    max_trades_per_day: int,
    use_binance_data: bool,
    enable_real_execution: bool,
    allow_live: bool,
    live_profile: str,
    use_gemini_advisor: bool,
    use_llm_strategy: bool,
    use_llm_strategy_refiner: bool,
    gemini_model: str,
    news_text: str,
    signal_text: str,
    confirmation: str,
) -> Tuple[str, str, str]:
    try:
        cycles = int(cycles)
        interval_sec = float(interval_sec)
        _validate_live_inputs(
            mode=mode,
            enable_real_execution=enable_real_execution,
            allow_live=allow_live,
            confirmation=confirmation,
            symbol=symbol,
            cycles=cycles,
            interval_sec=interval_sec,
        )

        risk_policy = {
            "mode": mode,
            "allowed_symbols": [symbol],
            "max_notional_quote": float(max_notional),
            "max_daily_notional_quote": float(max_daily_notional),
            "max_open_position_quote": float(max_open_position),
            "max_trades_per_day": int(max_trades_per_day),
            "allow_live_trading": bool(allow_live),
            "allow_margin": False,
            "allow_futures": False,
            "allow_withdrawals": False,
            "live_profile": live_profile,
        }

        cfg = AutonomousTradingConfig(
            mode=mode,  # type: ignore[arg-type]
            symbol=symbol,
            interval_seconds=interval_sec,
            max_cycles=cycles,
            risk_policy=risk_policy,
            news_items=_parse_news(news_text),
            external_signals=_parse_signals(signal_text),
            use_llm_advisor=bool(use_gemini_advisor),
            use_llm_strategy_planner=bool(use_llm_strategy),
            use_llm_strategy_refiner=bool(use_llm_strategy_refiner),
            llm_model=gemini_model.strip() or "gemini-2.5-flash",
            llm_provider="gemini" if use_gemini_advisor or use_llm_strategy else "auto",
            enable_real_execution=bool(enable_real_execution),
            store_path=DEFAULT_STORE_PATH,
            kill_switch_path=KILL_SWITCH_PATH,
            live_ack_confirmed=bool(mode == "live" and enable_real_execution),
            live_profile=live_profile,
            operative_ack_confirmed=bool(live_profile == "operative" and mode == "live" and enable_real_execution),
        )

        market_data = None
        broker = None
        if mode in {"testnet", "live"} and (use_binance_data or enable_real_execution):
            api_key, api_secret = require_binance_env_credentials(mode=mode)
            market_data = CCXTBinanceMarketDataProvider(mode=mode, api_key=api_key, api_secret=api_secret)
            broker = CCXTBinanceSpotBroker(mode=mode, api_key=api_key, api_secret=api_secret)

        runner = AutonomousTradingRunner(cfg, market_data=market_data, broker=broker)
        results = list(runner.run_forever())
        last = results[-1] if results else {}
        summary = _summarize(last)
        pretty = json.dumps(results, indent=2, sort_keys=True, default=str)
        tail = _tail_file(DEFAULT_STORE_PATH, lines=10)
        return summary, pretty, tail
    except Exception as exc:
        err = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return f"## Error\n`{err['error']}`", json.dumps(err, indent=2), _tail_file(DEFAULT_STORE_PATH, lines=10)


def activate_kill_switch() -> Tuple[str, str]:
    Path(KILL_SWITCH_PATH).write_text("active\n", encoding="utf-8")
    return "Kill switch activated. New cycles will be blocked.", secrets_markdown()


def clear_kill_switch() -> Tuple[str, str]:
    p = Path(KILL_SWITCH_PATH)
    if p.exists():
        p.unlink()
    return "Kill switch cleared.", secrets_markdown()


def refresh_status() -> Tuple[str, str]:
    ks = "ACTIVE" if Path(KILL_SWITCH_PATH).exists() else "inactive"
    status = f"**Kill switch:** `{ks}`\n\n" + secrets_markdown()
    return status, _tail_file(DEFAULT_STORE_PATH, lines=10)


DESCRIPTION = f"""
# {APP_TITLE}

Private control panel for a guarded trading agent:

`Gemini news/signals → LLM strategy → SFM critique/refinement → SFM/risk gate → Binance Spot executor`

Live trading is disabled unless you explicitly enable real execution, allow live trading, set the required Hugging Face Secrets, and type the confirmation phrase.

**Live confirmation phrase:** `{CONFIRM_LIVE_PHRASE}`
"""

with gr.Blocks(title="LangGraph-SFM Trading Private") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        status_md = gr.Markdown(value=secrets_markdown(), label="Secret status")
        audit_tail = gr.Code(label="Audit log tail", language="json", value=_tail_file(DEFAULT_STORE_PATH, lines=10))

    with gr.Row():
        refresh = gr.Button("Refresh status")
        kill = gr.Button("Activate kill switch", variant="stop")
        clear_kill = gr.Button("Clear kill switch")

    gr.Markdown("## Run configuration")
    with gr.Row():
        mode = gr.Dropdown(label="Mode", choices=["paper", "testnet", "live"], value="paper")
        symbol = gr.Dropdown(label="Symbol", choices=ALLOWED_SYMBOLS, value="BTC/USDT")
        live_profile = gr.Dropdown(label="Live profile", choices=["locked", "operative"], value="operative")

    with gr.Row():
        cycles = gr.Slider(label="Cycles", minimum=1, maximum=24, step=1, value=1)
        interval_sec = gr.Slider(label="Interval seconds", minimum=0, maximum=1800, step=60, value=0)

    with gr.Row():
        max_notional = gr.Number(label="Max notional per trade", value=10.0)
        max_daily_notional = gr.Number(label="Max daily notional", value=20.0)
        max_open_position = gr.Number(label="Max open position", value=20.0)
        max_trades_per_day = gr.Number(label="Max trades per day", value=2, precision=0)

    with gr.Row():
        use_binance_data = gr.Checkbox(label="Use Binance market/account data", value=False)
        enable_real_execution = gr.Checkbox(label="Enable real execution", value=False)
        allow_live = gr.Checkbox(label="Allow live trading", value=False)

    gr.Markdown("## Gemini / strategy")
    with gr.Row():
        use_gemini_advisor = gr.Checkbox(label="Use Gemini news advisor", value=True)
        use_llm_strategy = gr.Checkbox(label="Use Gemini strategy planner", value=True)
        use_llm_strategy_refiner = gr.Checkbox(label="Use Gemini strategy refiner after SFM critique", value=False)
        gemini_model = gr.Textbox(label="Gemini model", value=os.getenv("SFM_TRADING_GEMINI_MODEL", "gemini-2.5-flash"))

    news_text = gr.Textbox(
        label="News items, one per line",
        lines=4,
        value="BTC momentum positive, ETF inflows rising, no major negative regulatory news reported",
    )
    signal_text = gr.Textbox(
        label="Signals, one per line. Format: name=value:direction:confidence",
        lines=3,
        value="trend=sma_up:bullish:0.90",
    )
    confirmation = gr.Textbox(label="Live confirmation phrase", placeholder=CONFIRM_LIVE_PHRASE)

    run_btn = gr.Button("Run controlled cycle(s)", variant="primary")
    summary = gr.Markdown(label="Summary")
    result_json = gr.Code(label="Full JSON result", language="json")

    inputs = [
        mode,
        symbol,
        cycles,
        interval_sec,
        max_notional,
        max_daily_notional,
        max_open_position,
        max_trades_per_day,
        use_binance_data,
        enable_real_execution,
        allow_live,
        live_profile,
        use_gemini_advisor,
        use_llm_strategy,
        use_llm_strategy_refiner,
        gemini_model,
        news_text,
        signal_text,
        confirmation,
    ]
    run_btn.click(run_guard, inputs=inputs, outputs=[summary, result_json, audit_tail])
    refresh.click(refresh_status, outputs=[status_md, audit_tail])
    kill.click(activate_kill_switch, outputs=[summary, status_md])
    clear_kill.click(clear_kill_switch, outputs=[summary, status_md])

if __name__ == "__main__":
    demo.launch()

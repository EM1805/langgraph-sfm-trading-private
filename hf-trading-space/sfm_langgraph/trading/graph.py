from __future__ import annotations

"""LangGraph workflow for a guarded trading-agent cycle."""

from typing import Any, Dict, List

from sfm_langgraph.monitor import SFMAgentMonitor
from sfm_langgraph.node import SFMIntentAnalyzerNode

from .advisor import HeuristicNewsSignalAdvisor, LLMNewsSignalAdvisor
from .broker import PaperBroker
from .policy import evaluate_trade_policy, policy_from_mapping, proposal_from_mapping
from .strategy import SmaCrossoverStrategy
from .strategy_planner import (
    HeuristicStrategyPlanner,
    LLMStrategyPlanner,
    StrategyRefiner,
    critique_strategy_with_sfm,
    proposal_from_strategy_plan,
)
from .types import MarketSnapshot, TradingState

DEFAULT_PRICES = [
    101.0,
    100.7,
    100.9,
    101.3,
    101.8,
    102.1,
    102.4,
    102.9,
    103.3,
    103.6,
    104.2,
    104.7,
]


def _append_audit(state: TradingState, node: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    audit = list(state.get("audit_log", []))
    audit.append({"node": node, **payload})
    return audit


def market_snapshot_node(state: TradingState) -> TradingState:
    symbol = state.get("symbol", "BTC/USDT")
    prices = state.get("prices") or DEFAULT_PRICES
    strategy = SmaCrossoverStrategy()
    snapshot = strategy.snapshot(symbol, prices, source=str(state.get("mode", "paper")))
    return {
        "market": snapshot.to_dict(),
        "audit_log": _append_audit(state, "market_snapshot", snapshot.to_dict()),
    }


def news_signal_advisor_node(state: TradingState) -> TradingState:
    """Read news/signals and produce an advisory market view.

    If ``use_llm_advisor`` is false, this uses a deterministic heuristic.  If it
    is true, it attempts an OpenAI-compatible LLM call and falls back safely when
    credentials/dependencies are missing.
    """
    market = MarketSnapshot(**state["market"])
    if state.get("use_llm_advisor", False):
        advisor = LLMNewsSignalAdvisor(model=state.get("llm_model") or None, provider=str(state.get("llm_provider", "auto")))
    else:
        advisor = HeuristicNewsSignalAdvisor()
    view = advisor.analyze(
        market=market,
        news_items=state.get("news_items", []),
        signals=state.get("external_signals", []),
    )
    return {
        "llm_market_view": view.to_dict(),
        "audit_log": _append_audit(state, "news_signal_advisor", view.to_dict()),
    }


def llm_strategy_planner_node(state: TradingState) -> TradingState:
    """Create an optional LLM/news-driven strategy plan.

    Disabled by default. When enabled, the LLM proposes a structured strategy
    plan, but it cannot execute. The plan is passed to the SFM strategy critic
    and then the hard risk gate.
    """
    if not state.get("use_llm_strategy_planner", False):
        return {
            "audit_log": _append_audit(state, "llm_strategy_planner", {"enabled": False}),
        }
    market = MarketSnapshot(**state["market"])
    policy = policy_from_mapping(state.get("risk_policy"))
    market_view = state.get("llm_market_view", {}) or {}
    if state.get("use_llm_advisor", False):
        planner = LLMStrategyPlanner(model=state.get("llm_model") or None, provider=str(state.get("llm_provider", "auto")))
    else:
        planner = HeuristicStrategyPlanner()
    plan = planner.plan(
        market=market,
        market_view=market_view,
        risk_policy=policy,
        news_items=state.get("news_items", []),
        signals=state.get("external_signals", []),
        open_position_quote=float(state.get("open_position_quote", 0.0)),
    )
    return {
        "llm_strategy_plan": plan.to_dict(),
        "audit_log": _append_audit(state, "llm_strategy_planner", plan.to_dict()),
    }


def sfm_strategy_critic_node(state: TradingState) -> TradingState:
    """Ask SFM to critique the LLM strategy plan before it becomes a proposal."""
    if not state.get("use_llm_strategy_planner", False) or not state.get("llm_strategy_plan"):
        return {
            "audit_log": _append_audit(state, "sfm_strategy_critic", {"enabled": False}),
        }
    market = MarketSnapshot(**state["market"])
    policy = policy_from_mapping(state.get("risk_policy"))
    critique = critique_strategy_with_sfm(
        plan=state.get("llm_strategy_plan", {}),
        market=market,
        market_view=state.get("llm_market_view", {}) or {},
        risk_policy=policy,
    )
    return {
        "sfm_strategy_critique": critique.to_dict(),
        "audit_log": _append_audit(state, "sfm_strategy_critic", critique.to_dict()),
    }


def strategy_refiner_node(state: TradingState) -> TradingState:
    """Refine the LLM plan using SFM critique before hard policy gating."""
    if not state.get("use_llm_strategy_planner", False) or not state.get("llm_strategy_plan"):
        return {
            "audit_log": _append_audit(state, "strategy_refiner", {"enabled": False}),
        }
    market = MarketSnapshot(**state["market"])
    policy = policy_from_mapping(state.get("risk_policy"))
    refiner = StrategyRefiner(model=state.get("llm_model") or None, provider=str(state.get("llm_provider", "auto")))
    refined = refiner.refine(
        plan=state.get("llm_strategy_plan", {}),
        critique=state.get("sfm_strategy_critique", {}),
        market=market,
        market_view=state.get("llm_market_view", {}) or {},
        risk_policy=policy,
        news_items=state.get("news_items", []),
        signals=state.get("external_signals", []),
        use_llm=bool(state.get("use_llm_strategy_refiner", False) and state.get("use_llm_advisor", False)),
    )
    return {
        "revised_strategy_plan": refined.to_dict(),
        "audit_log": _append_audit(state, "strategy_refiner", refined.to_dict()),
    }


def strategy_node(state: TradingState) -> TradingState:
    policy = policy_from_mapping(state.get("risk_policy"))
    market = MarketSnapshot(**state["market"])
    if state.get("use_llm_strategy_planner", False) and state.get("revised_strategy_plan"):
        proposal = proposal_from_strategy_plan(state.get("revised_strategy_plan", {}), policy=policy)
    elif state.get("use_llm_strategy_planner", False) and state.get("llm_strategy_plan"):
        proposal = proposal_from_strategy_plan(state.get("llm_strategy_plan", {}), policy=policy)
    else:
        proposal = SmaCrossoverStrategy().propose(
            market,
            policy,
            open_position_quote=float(state.get("open_position_quote", 0.0)),
        )
    view = state.get("llm_market_view", {}) or {}
    view_rationale = str(view.get("rationale", "")).strip()
    if view_rationale:
        proposal = proposal.__class__(
            **{
                **proposal.to_dict(),
                "rationale": proposal.rationale + " News/signal advisor context: " + view_rationale,
                "source": proposal.source + "+NewsSignalAdvisor",
            }
        )
    return {
        "proposal": proposal.to_dict(),
        "audit_log": _append_audit(state, "strategy", proposal.to_dict()),
    }


def build_sfm_query_node(state: TradingState) -> TradingState:
    proposal = proposal_from_mapping(state.get("proposal"), default_symbol=state.get("symbol", "BTC/USDT"))
    policy = policy_from_mapping(state.get("risk_policy"))
    market_view = state.get("llm_market_view", {}) or {}
    action = f"{proposal.side}_{proposal.symbol}_{proposal.notional_quote}_spot_order"
    candidate_goals = [
        {"goal_variable": "execute_small_spot_trade"},
        {"goal_variable": "increase_portfolio_value"},
        {"goal_variable": "capital_loss", "protected_outcomes": ["capital_loss"]},
        {"goal_variable": "overtrading", "protected_outcomes": ["overtrading"]},
        {"goal_variable": "unauthorized_asset", "protected_outcomes": ["unauthorized_asset"]},
        {"goal_variable": "oversized_trade", "protected_outcomes": ["oversized_trade"]},
        {"goal_variable": "news_driven_overreaction", "protected_outcomes": ["news_driven_overreaction"]},
    ]
    query = {
        "observed_action": action,
        "candidate_goals": candidate_goals,
        "protected_outcome": "capital_loss",
        "stated_goal": "execute a small governed spot trade only if risk policy allows it",
        "market_advisor_context": market_view,
        "graph": {
            "nodes": [
                "agent_action",
                "execute_small_spot_trade",
                "increase_portfolio_value",
                "capital_loss",
                "overtrading",
                "unauthorized_asset",
                "oversized_trade",
                "news_driven_overreaction",
            ],
            "edges": [
                ["agent_action", "execute_small_spot_trade"],
                ["agent_action", "increase_portfolio_value"],
                ["agent_action", "capital_loss"],
                ["agent_action", "overtrading"],
                ["agent_action", "unauthorized_asset"],
                ["agent_action", "oversized_trade"],
                ["agent_action", "news_driven_overreaction"],
            ],
        },
        "normative_policy": {
            "deny_goals": [
                "capital_loss",
                "overtrading",
                "unauthorized_asset",
                "oversized_trade",
                "live_trading_without_ack",
                "margin_or_futures_exposure",
                "news_driven_overreaction",
            ],
            "allowed_symbols": policy.allowed_symbols,
            "max_notional_quote": policy.max_notional_quote,
            "llm_market_view": market_view,
        },
    }
    return {
        "sfm_query": query,
        "observed_action": action,
        "candidate_goals": candidate_goals,
        "protected_outcome": "capital_loss",
        "declared_goal": "execute a small governed spot trade only if risk policy allows it",
        "graph": query["graph"],
        "normative_policy": query["normative_policy"],
        "audit_log": _append_audit(state, "sfm_query", {"observed_action": action}),
    }


def trading_gate_node(state: TradingState) -> TradingState:
    policy = policy_from_mapping(state.get("risk_policy"))
    proposal = proposal_from_mapping(state.get("proposal"), default_symbol=state.get("symbol", "BTC/USDT"))
    result = evaluate_trade_policy(
        proposal,
        policy,
        prior_trades_today=state.get("prior_trades_today", []),
        open_position_quote=float(state.get("open_position_quote", 0.0)),
        sfm_analysis=state.get("sfm_analysis", {}),
        market_view=state.get("llm_market_view", {}),
    )
    return {
        "gate": result.to_dict(),
        "gate_decision": result.decision,
        "gate_reason": result.reason,
        "violations": result.violations,
        "audit_log": _append_audit(state, "trading_gate", result.to_dict()),
    }


def route_after_gate(state: TradingState) -> str:
    return "execute" if state.get("gate_decision") == "allow" else "skip"


def execute_order_node(state: TradingState) -> TradingState:
    # The reusable graph uses paper execution only.  For ccxt/binance execution,
    # call evaluate_trade_policy first, then pass the proposal to the broker in
    # your own service layer.
    broker = PaperBroker(mode="paper")
    market = MarketSnapshot(**state["market"])
    proposal = proposal_from_mapping(state.get("proposal"), default_symbol=state.get("symbol", "BTC/USDT"))
    report = broker.execute_order(proposal, market=market)
    return {
        "execution_report": report.to_dict(),
        "audit_log": _append_audit(state, "execute_order", report.to_dict()),
    }


def skip_order_node(state: TradingState) -> TradingState:
    report = {
        "executed": False,
        "mode": state.get("mode", "paper"),
        "symbol": state.get("symbol", "BTC/USDT"),
        "side": (state.get("proposal") or {}).get("side", "hold"),
        "status": "not_submitted",
        "reason": state.get("gate_reason", "blocked or review required"),
    }
    return {
        "execution_report": report,
        "audit_log": _append_audit(state, "skip_order", report),
    }


def build_trading_guard_graph():
    """Build the LangGraph workflow.

    Importing this function does not require LangGraph.  Calling it does.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install langgraph to build the trading guard graph: pip install langgraph") from exc

    builder = StateGraph(TradingState)
    builder.add_node("market_snapshot", market_snapshot_node)
    builder.add_node("news_signal_advisor", news_signal_advisor_node)
    builder.add_node("llm_strategy_planner", llm_strategy_planner_node)
    builder.add_node("sfm_strategy_critic", sfm_strategy_critic_node)
    builder.add_node("strategy_refiner", strategy_refiner_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("build_sfm_query", build_sfm_query_node)
    builder.add_node("sfm_intent_analyzer", SFMIntentAnalyzerNode())
    builder.add_node("trading_gate", trading_gate_node)
    builder.add_node("execute_order", execute_order_node)
    builder.add_node("skip_order", skip_order_node)
    builder.add_node("sfm_monitor", SFMAgentMonitor())

    builder.add_edge(START, "market_snapshot")
    builder.add_edge("market_snapshot", "news_signal_advisor")
    builder.add_edge("news_signal_advisor", "llm_strategy_planner")
    builder.add_edge("llm_strategy_planner", "sfm_strategy_critic")
    builder.add_edge("sfm_strategy_critic", "strategy_refiner")
    builder.add_edge("strategy_refiner", "strategy")
    builder.add_edge("strategy", "build_sfm_query")
    builder.add_edge("build_sfm_query", "sfm_intent_analyzer")
    builder.add_edge("sfm_intent_analyzer", "trading_gate")
    builder.add_conditional_edges("trading_gate", route_after_gate, {"execute": "execute_order", "skip": "skip_order"})
    builder.add_edge("execute_order", "sfm_monitor")
    builder.add_edge("skip_order", "sfm_monitor")
    builder.add_edge("sfm_monitor", END)
    return builder.compile()


def _run_trading_guard_sequential(initial: TradingState) -> TradingState:
    """Fallback runner used when LangGraph is not installed.

    The node order is the same as the LangGraph workflow, but execution is plain
    Python so policy tests and paper demos still work with the base package.
    """
    current: TradingState = dict(initial)  # type: ignore[assignment]
    for node in (market_snapshot_node, news_signal_advisor_node, llm_strategy_planner_node, sfm_strategy_critic_node, strategy_refiner_node, strategy_node, build_sfm_query_node, SFMIntentAnalyzerNode(), trading_gate_node):
        current.update(node(current))  # type: ignore[arg-type]
    if route_after_gate(current) == "execute":
        current.update(execute_order_node(current))
    else:
        current.update(skip_order_node(current))
    current.update(SFMAgentMonitor()(current))
    return current


def run_trading_guard_cycle(state: TradingState | None = None) -> TradingState:
    initial: TradingState = {
        "mode": "paper",
        "symbol": "BTC/USDT",
        "prices": DEFAULT_PRICES,
        "risk_policy": {},
        "news_items": [],
        "external_signals": [],
        "use_llm_advisor": False,
        "use_llm_strategy_planner": False,
        "use_llm_strategy_refiner": False,
        "prior_trades_today": [],
        "open_position_quote": 0.0,
        "audit_log": [],
    }
    if state:
        initial.update(state)
    try:
        graph = build_trading_guard_graph()
    except RuntimeError:
        return _run_trading_guard_sequential(initial)
    return graph.invoke(initial, config={"recursion_limit": 50})

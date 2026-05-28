from __future__ import annotations

"""LLM strategy planning with SFM-style critique and safe refinement.

The planner is deliberately advisory.  It may transform news, signals, and
market context into a structured strategy plan, but the plan must still pass the
SFM critic, SFM final gate, hard risk policy, and execution gate before any
broker receives an order.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol

from sfm_langgraph.node import SFMIntentAnalyzerNode

from .advisor import HeuristicNewsSignalAdvisor, LLMNewsSignalAdvisor, MarketAdvisorView
from .policy import policy_from_mapping, proposal_from_mapping
from .types import MarketSnapshot, TradeProposal, TradingRiskPolicy


@dataclass(frozen=True)
class LLMStrategyPlan:
    """Structured strategy produced by an LLM or deterministic fallback."""

    symbol: str
    action: str = "hold"  # buy | sell | hold
    strategy_name: str = "guarded_hold"
    market_thesis: str = "No actionable strategy."
    max_notional_quote: float = 0.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_holding_minutes: int = 0
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provider: str = "heuristic_strategy_planner"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SFMStrategyCritique:
    """SFM-style critique of a proposed strategy before final execution gate."""

    decision: str = "review"  # allow | review | block
    problems: List[str] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)
    recommended_revision: Dict[str, Any] = field(default_factory=dict)
    reason: str = "Strategy requires conservative review."
    sfm_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyPlanner(Protocol):
    def plan(
        self,
        *,
        market: MarketSnapshot,
        market_view: Mapping[str, Any],
        risk_policy: TradingRiskPolicy,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
        open_position_quote: float = 0.0,
    ) -> LLMStrategyPlan:
        ...


def _coerce_action(action: Any) -> str:
    lowered = str(action or "hold").lower().strip()
    if lowered in {"buy", "long"}:
        return "buy"
    if lowered in {"sell", "exit"}:
        return "sell"
    return "hold"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list_str(value: Any, *, limit: int = 10) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value][:limit]
    try:
        return [str(item) for item in list(value)[:limit]]
    except Exception:
        return [str(value)][:limit]


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "{}").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned or "{}")


def _plan_from_mapping(data: Mapping[str, Any], *, symbol: str, provider: str) -> LLMStrategyPlan:
    action = _coerce_action(data.get("action", data.get("side", "hold")))
    confidence = max(0.0, min(1.0, _as_float(data.get("confidence"), 0.0)))
    return LLMStrategyPlan(
        symbol=str(data.get("symbol", symbol)),
        action=action,
        strategy_name=str(data.get("strategy_name", "llm_guarded_strategy")),
        market_thesis=str(data.get("market_thesis", data.get("rationale", ""))),
        max_notional_quote=max(0.0, _as_float(data.get("max_notional_quote", data.get("notional_quote")), 0.0)),
        stop_loss_pct=None if data.get("stop_loss_pct") in {None, ""} else _as_float(data.get("stop_loss_pct")),
        take_profit_pct=None if data.get("take_profit_pct") in {None, ""} else _as_float(data.get("take_profit_pct")),
        max_holding_minutes=max(0, _as_int(data.get("max_holding_minutes"), 0)),
        confidence=confidence,
        evidence=_list_str(data.get("evidence"), limit=8),
        risks=_list_str(data.get("risks"), limit=8),
        warnings=_list_str(data.get("warnings"), limit=8),
        provider=provider,
        raw=dict(data),
    )


def _strategy_payload(
    *,
    market: MarketSnapshot,
    market_view: Mapping[str, Any],
    risk_policy: TradingRiskPolicy,
    news_items: Iterable[Mapping[str, Any]] | None,
    signals: Iterable[Mapping[str, Any]] | None,
    open_position_quote: float,
    critique: Mapping[str, Any] | None = None,
    previous_plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "market": market.to_dict(),
        "market_view": dict(market_view or {}),
        "risk_policy": risk_policy.to_dict(),
        "news_items": list(news_items or []),
        "signals": list(signals or []),
        "open_position_quote": open_position_quote,
        "previous_plan": dict(previous_plan or {}),
        "sfm_strategy_critique": dict(critique or {}),
        "instruction": (
            "Create one conservative spot-trading strategy plan for a guarded LangGraph-SFM trading agent. "
            "This is not financial advice and must be suitable for a safety gate. Return only JSON with: "
            "symbol, action buy|sell|hold, strategy_name, market_thesis, max_notional_quote, "
            "stop_loss_pct, take_profit_pct, max_holding_minutes, confidence 0..1, evidence[], risks[], warnings[]. "
            "Prefer hold if evidence is weak, risk is medium/high, news is unverified, or the critique says block/no_trade. "
            "If action is buy, include a stop_loss_pct and take_profit_pct. Never request margin, futures, leverage, or withdrawals."
        ),
    }


class HeuristicStrategyPlanner:
    """Offline strategy planner derived from the advisor view and risk policy."""

    def plan(
        self,
        *,
        market: MarketSnapshot,
        market_view: Mapping[str, Any],
        risk_policy: TradingRiskPolicy,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
        open_position_quote: float = 0.0,
    ) -> LLMStrategyPlan:
        risk_level = str(market_view.get("risk_level", "unknown")).lower()
        bias = str(market_view.get("trade_bias", "hold_bias")).lower()
        confidence = max(0.0, min(1.0, _as_float(market_view.get("confidence"), 0.0)))
        trend = (market.trend or "unknown").lower()
        max_notional = min(float(risk_policy.max_notional_quote), 10.0)
        evidence = _list_str(market_view.get("key_events"), limit=6)
        risks = _list_str(market_view.get("warnings"), limit=6)

        if risk_level == "high" or bias == "no_trade":
            return LLMStrategyPlan(
                symbol=market.symbol,
                action="hold",
                strategy_name="sfm_guarded_no_trade",
                market_thesis="High-risk or no-trade advisory context; safest strategy is to wait.",
                confidence=max(confidence, 0.55),
                evidence=evidence,
                risks=risks + ["risk_context_requires_no_trade"],
                provider="heuristic_strategy_planner",
            )
        if bias == "buy_bias" and confidence >= 0.58 and trend in {"up", "unknown"}:
            return LLMStrategyPlan(
                symbol=market.symbol,
                action="buy",
                strategy_name="guarded_news_momentum_buy",
                market_thesis="Bullish news/signal context supports a small guarded spot entry under strict stop-loss controls.",
                max_notional_quote=max_notional,
                stop_loss_pct=max(risk_policy.min_stop_loss_pct, min(1.5, risk_policy.max_stop_loss_pct)),
                take_profit_pct=min(3.0, risk_policy.max_take_profit_pct),
                max_holding_minutes=180,
                confidence=max(confidence, 0.62),
                evidence=evidence or ["bullish_advisory_bias"],
                risks=risks + ["news_may_be_priced_in", "short_term_volatility"],
                provider="heuristic_strategy_planner",
            )
        if bias == "sell_bias" and confidence >= 0.58 and open_position_quote > 0:
            return LLMStrategyPlan(
                symbol=market.symbol,
                action="sell",
                strategy_name="guarded_risk_reduction_sell",
                market_thesis="Bearish news/signal context supports reducing an existing spot exposure.",
                max_notional_quote=min(max_notional, open_position_quote),
                stop_loss_pct=None,
                take_profit_pct=None,
                max_holding_minutes=0,
                confidence=max(confidence, 0.62),
                evidence=evidence or ["bearish_advisory_bias"],
                risks=risks,
                provider="heuristic_strategy_planner",
            )
        return LLMStrategyPlan(
            symbol=market.symbol,
            action="hold",
            strategy_name="guarded_hold_weak_signal",
            market_thesis="Signals are not strong enough for an autonomous trade.",
            confidence=confidence,
            evidence=evidence,
            risks=risks + ["weak_or_conflicting_evidence"],
            provider="heuristic_strategy_planner",
        )


class LLMStrategyPlanner:
    """LLM strategy planner supporting Gemini/OpenAI via existing advisor plumbing."""

    def __init__(self, *, model: Optional[str] = None, provider: str = "auto", temperature: float = 0.0) -> None:
        self.model = model or os.getenv("SFM_TRADING_LLM_MODEL", "")
        self.provider = (provider or os.getenv("SFM_TRADING_LLM_PROVIDER", "auto")).lower()
        self.temperature = float(temperature)
        self.fallback = HeuristicStrategyPlanner()

    def _resolved_provider(self) -> str:
        if self.provider in {"gemini", "google"}:
            return "gemini"
        if self.provider in {"openai", "chatgpt"}:
            return "openai"
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "heuristic"

    def plan(
        self,
        *,
        market: MarketSnapshot,
        market_view: Mapping[str, Any],
        risk_policy: TradingRiskPolicy,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
        open_position_quote: float = 0.0,
    ) -> LLMStrategyPlan:
        resolved = self._resolved_provider()
        if resolved == "heuristic":
            plan = self.fallback.plan(
                market=market,
                market_view=market_view,
                risk_policy=risk_policy,
                news_items=news_items,
                signals=signals,
                open_position_quote=open_position_quote,
            )
            return LLMStrategyPlan(**{**plan.to_dict(), "provider": "heuristic_strategy_fallback_no_llm_key"})
        payload = _strategy_payload(
            market=market,
            market_view=market_view,
            risk_policy=risk_policy,
            news_items=news_items,
            signals=signals,
            open_position_quote=open_position_quote,
        )
        prompt = "Return only valid JSON, no markdown.\n\n" + json.dumps(payload, sort_keys=True)
        if resolved == "gemini":
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore

                model = self.model or os.getenv("SFM_TRADING_GEMINI_MODEL", "gemini-2.5-flash")
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
                try:
                    config = types.GenerateContentConfig(temperature=self.temperature, response_mime_type="application/json")
                except Exception:
                    config = None
                kwargs: Dict[str, Any] = {"model": model, "contents": prompt}
                if config is not None:
                    kwargs["config"] = config
                response = client.models.generate_content(**kwargs)
                data = _parse_json_object(getattr(response, "text", "") or "{}")
                return _plan_from_mapping(data, symbol=market.symbol, provider=f"gemini_strategy:{model}")
            except Exception as exc:
                plan = self.fallback.plan(
                    market=market,
                    market_view=market_view,
                    risk_policy=risk_policy,
                    news_items=news_items,
                    signals=signals,
                    open_position_quote=open_position_quote,
                )
                return LLMStrategyPlan(**{**plan.to_dict(), "provider": "heuristic_strategy_fallback_gemini_error", "raw": {"llm_error": f"{type(exc).__name__}: {exc}"}})
        try:
            from openai import OpenAI  # type: ignore

            model = self.model or os.getenv("SFM_TRADING_OPENAI_MODEL", "gpt-4o-mini")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model=model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You create conservative strategy plans for a safety-gated spot trading agent."},
                    {"role": "user", "content": json.dumps(payload, sort_keys=True)},
                ],
            )
            data = _parse_json_object(response.choices[0].message.content or "{}")
            return _plan_from_mapping(data, symbol=market.symbol, provider=f"openai_strategy:{model}")
        except Exception as exc:
            plan = self.fallback.plan(
                market=market,
                market_view=market_view,
                risk_policy=risk_policy,
                news_items=news_items,
                signals=signals,
                open_position_quote=open_position_quote,
            )
            return LLMStrategyPlan(**{**plan.to_dict(), "provider": "heuristic_strategy_fallback_openai_error", "raw": {"llm_error": f"{type(exc).__name__}: {exc}"}})


def proposal_from_strategy_plan(plan: Mapping[str, Any] | LLMStrategyPlan, *, policy: TradingRiskPolicy) -> TradeProposal:
    data = plan.to_dict() if isinstance(plan, LLMStrategyPlan) else dict(plan or {})
    action = _coerce_action(data.get("action"))
    if action == "hold":
        return TradeProposal(
            symbol=str(data.get("symbol", policy.allowed_symbols[0] if policy.allowed_symbols else "BTC/USDT")),
            side="hold",
            notional_quote=0.0,
            confidence=_as_float(data.get("confidence"), 0.0),
            rationale=str(data.get("market_thesis", "LLM/SFM strategy plan chose hold.")),
            source=str(data.get("provider", "llm_strategy_plan")),
        )
    notional = min(max(0.0, _as_float(data.get("max_notional_quote"), 0.0)), float(policy.max_notional_quote))
    return TradeProposal(
        symbol=str(data.get("symbol", policy.allowed_symbols[0] if policy.allowed_symbols else "BTC/USDT")),
        side=action,  # type: ignore[arg-type]
        notional_quote=notional,
        order_type="market",
        stop_loss_pct=None if data.get("stop_loss_pct") in {None, ""} else _as_float(data.get("stop_loss_pct")),
        take_profit_pct=None if data.get("take_profit_pct") in {None, ""} else _as_float(data.get("take_profit_pct")),
        confidence=_as_float(data.get("confidence"), 0.0),
        rationale=str(data.get("market_thesis", "LLM generated guarded strategy plan.")),
        source=str(data.get("provider", "llm_strategy_plan")),
    )


def critique_strategy_with_sfm(
    *,
    plan: Mapping[str, Any],
    market: MarketSnapshot,
    market_view: Mapping[str, Any],
    risk_policy: TradingRiskPolicy,
) -> SFMStrategyCritique:
    action = _coerce_action(plan.get("action"))
    notional = _as_float(plan.get("max_notional_quote"), 0.0)
    confidence = _as_float(plan.get("confidence"), 0.0)
    problems: List[str] = []
    required_changes: List[str] = []
    recommended: Dict[str, Any] = {}

    if action == "hold":
        return SFMStrategyCritique(
            decision="block",
            problems=["strategy_recommends_hold"],
            required_changes=["wait_for_actionable_signal"],
            recommended_revision={"action": "hold", "max_notional_quote": 0.0},
            reason="LLM strategy planner selected hold/no-trade; no order should be submitted.",
            sfm_analysis={},
        )

    if str(market_view.get("risk_level", "unknown")).lower() == "high":
        problems.append("high_news_risk")
        required_changes.append("switch_to_no_trade_or_human_review")
        recommended.update({"action": "hold", "max_notional_quote": 0.0})
    if str(market_view.get("trade_bias", "hold_bias")).lower() == "no_trade":
        problems.append("advisor_recommends_no_trade")
        required_changes.append("wait_for_news_risk_to_clear")
        recommended.update({"action": "hold", "max_notional_quote": 0.0})
    if confidence < risk_policy.min_confidence:
        problems.append("strategy_confidence_below_policy")
        required_changes.append("raise_confidence_or_reduce_to_hold")
    if notional <= 0:
        problems.append("non_positive_strategy_notional")
        required_changes.append("set_positive_notional_within_policy_or_hold")
    if notional > risk_policy.max_notional_quote:
        problems.append("strategy_notional_exceeds_policy")
        required_changes.append("reduce_notional_to_policy_limit")
        recommended["max_notional_quote"] = min(notional, risk_policy.max_notional_quote)
    if action == "buy" and risk_policy.require_stop_loss and plan.get("stop_loss_pct") in {None, ""}:
        problems.append("missing_stop_loss")
        required_changes.append("add_policy_compliant_stop_loss")
        recommended["stop_loss_pct"] = min(max(1.5, risk_policy.min_stop_loss_pct), risk_policy.max_stop_loss_pct)
    if action == "buy" and plan.get("take_profit_pct") in {None, ""}:
        required_changes.append("add_take_profit_or_exit_plan")
        recommended["take_profit_pct"] = min(3.0, risk_policy.max_take_profit_pct)
    if str(plan.get("symbol", market.symbol)) not in risk_policy.allowed_symbols:
        problems.append("unauthorized_strategy_symbol")
        required_changes.append("use_allowed_symbol_only")
        recommended["symbol"] = risk_policy.allowed_symbols[0] if risk_policy.allowed_symbols else market.symbol

    temp_state: Dict[str, Any] = {
        "observed_action": f"strategy_{action}_{market.symbol}_{notional}_spot_trade",
        "declared_goal": "produce a small governed spot-trading strategy only if safety controls are present",
        "candidate_goals": [
            {"goal_variable": "execute_small_spot_trade"},
            {"goal_variable": "increase_portfolio_value"},
            {"goal_variable": "capital_loss", "protected_outcomes": ["capital_loss"]},
            {"goal_variable": "overtrading", "protected_outcomes": ["overtrading"]},
            {"goal_variable": "news_driven_overreaction", "protected_outcomes": ["news_driven_overreaction"]},
            {"goal_variable": "oversized_trade", "protected_outcomes": ["oversized_trade"]},
        ],
        "protected_outcome": "capital_loss",
        "graph": {
            "nodes": ["agent_strategy", "execute_small_spot_trade", "increase_portfolio_value", "capital_loss", "overtrading", "news_driven_overreaction", "oversized_trade"],
            "edges": [
                ["agent_strategy", "execute_small_spot_trade"],
                ["agent_strategy", "increase_portfolio_value"],
                ["agent_strategy", "capital_loss"],
                ["agent_strategy", "overtrading"],
                ["agent_strategy", "news_driven_overreaction"],
                ["agent_strategy", "oversized_trade"],
            ],
        },
        "normative_policy": {
            "deny_goals": ["capital_loss", "overtrading", "news_driven_overreaction", "oversized_trade", "margin_or_futures_exposure"],
            "strategy_plan": dict(plan),
            "market_view": dict(market_view),
            "max_notional_quote": risk_policy.max_notional_quote,
        },
    }
    try:
        temp_state.update(SFMIntentAnalyzerNode()(temp_state))
        sfm_analysis = dict(temp_state.get("sfm_analysis", {}))
    except Exception as exc:
        sfm_analysis = {"sfm_strategy_critic_error": f"{type(exc).__name__}: {exc}", "gate_status": "review"}

    sfm_gate = str(sfm_analysis.get("gate_status", "review")).lower()
    if sfm_gate == "block":
        problems.append("sfm_strategy_gate_block")
    elif sfm_gate == "review":
        required_changes.append("preserve_human_review_or_small_size")

    if not recommended:
        recommended = {
            "action": action,
            "symbol": plan.get("symbol", market.symbol),
            "max_notional_quote": min(notional, risk_policy.max_notional_quote),
        }
    recommended.setdefault("symbol", plan.get("symbol", market.symbol))
    recommended.setdefault("action", action)
    recommended.setdefault("stop_loss_pct", plan.get("stop_loss_pct"))
    recommended.setdefault("take_profit_pct", plan.get("take_profit_pct"))
    recommended.setdefault("max_holding_minutes", plan.get("max_holding_minutes", 180))

    if any(item in problems for item in ["high_news_risk", "advisor_recommends_no_trade", "sfm_strategy_gate_block"]):
        decision = "block"
    elif problems or sfm_gate == "review":
        decision = "review"
    else:
        decision = "allow"
    reason = "SFM strategy critic found: " + ", ".join(sorted(set(problems or required_changes or ["no_hard_problems"])))
    return SFMStrategyCritique(
        decision=decision,
        problems=sorted(set(problems)),
        required_changes=sorted(set(required_changes)),
        recommended_revision=recommended,
        reason=reason,
        sfm_analysis=sfm_analysis,
    )


class StrategyRefiner:
    """Refine a plan using SFM critique; uses safe deterministic fallback."""

    def __init__(self, *, model: Optional[str] = None, provider: str = "auto", temperature: float = 0.0) -> None:
        self.model = model or ""
        self.provider = provider
        self.temperature = float(temperature)

    def refine(
        self,
        *,
        plan: Mapping[str, Any],
        critique: Mapping[str, Any],
        market: MarketSnapshot,
        market_view: Mapping[str, Any],
        risk_policy: TradingRiskPolicy,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
        use_llm: bool = False,
    ) -> LLMStrategyPlan:
        if str(critique.get("decision", "review")).lower() == "block":
            data = {**dict(plan), **dict(critique.get("recommended_revision") or {}), "action": "hold", "max_notional_quote": 0.0}
            return _plan_from_mapping(data, symbol=market.symbol, provider="sfm_deterministic_refiner_block_to_hold")
        if not use_llm:
            data = {**dict(plan), **dict(critique.get("recommended_revision") or {})}
            if data.get("action") == "buy":
                data.setdefault("stop_loss_pct", min(max(1.5, risk_policy.min_stop_loss_pct), risk_policy.max_stop_loss_pct))
                data.setdefault("take_profit_pct", min(3.0, risk_policy.max_take_profit_pct))
            data["max_notional_quote"] = min(_as_float(data.get("max_notional_quote"), 0.0), risk_policy.max_notional_quote)
            return _plan_from_mapping(data, symbol=market.symbol, provider="sfm_deterministic_strategy_refiner")

        payload = _strategy_payload(
            market=market,
            market_view=market_view,
            risk_policy=risk_policy,
            news_items=news_items,
            signals=signals,
            open_position_quote=0.0,
            critique=critique,
            previous_plan=plan,
        )
        planner = LLMStrategyPlanner(model=self.model, provider=self.provider, temperature=self.temperature)
        # The planner prompt receives critique/previous_plan via the payload below.
        # Reuse provider plumbing manually by temporarily asking through a synthetic market view.
        # If provider fails, deterministic refinement below is still safe.
        try:
            # Since LLMStrategyPlanner.plan does not accept critique directly, call its provider-specific
            # internals by creating a conservative market view augmented with critique text. This keeps
            # the public API simple while still giving the LLM the SFM feedback.
            augmented_view = {
                **dict(market_view),
                "sfm_strategy_critique": critique,
                "previous_strategy_plan": plan,
                "rationale": json.dumps(payload, sort_keys=True)[:6000],
            }
            refined = planner.plan(
                market=market,
                market_view=augmented_view,
                risk_policy=risk_policy,
                news_items=news_items,
                signals=signals,
                open_position_quote=0.0,
            )
            return LLMStrategyPlan(**{**refined.to_dict(), "provider": refined.provider + "+sfm_refined"})
        except Exception:
            data = {**dict(plan), **dict(critique.get("recommended_revision") or {})}
            return _plan_from_mapping(data, symbol=market.symbol, provider="sfm_deterministic_strategy_refiner_after_llm_error")

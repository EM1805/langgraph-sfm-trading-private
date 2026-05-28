from __future__ import annotations

"""Optional news/signal advisor for the trading guard.

The advisor is intentionally advisory only.  It can read news and external
signals, but it never executes orders.  The hard risk policy and SFM gate remain
responsible for allow/review/block decisions.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol

from .types import MarketSnapshot


@dataclass(frozen=True)
class NewsItem:
    """Small normalized news item supplied by a caller or data fetcher."""

    title: str
    summary: str = ""
    source: str = "manual"
    url: str = ""
    published_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketSignal:
    """Small normalized market signal supplied by a caller or strategy."""

    name: str
    value: str | float | int | bool
    direction: str = "neutral"
    confidence: float = 0.5
    source: str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketAdvisorView:
    """Structured advisory view produced from news/signals.

    ``trade_bias`` is not an order.  The trading gate may use it as risk context,
    but the policy layer still makes the final decision.
    """

    symbol: str
    risk_level: str = "unknown"  # low | medium | high | unknown
    news_sentiment: str = "neutral"  # bullish | bearish | mixed | neutral | unknown
    trade_bias: str = "hold_bias"  # buy_bias | sell_bias | hold_bias | no_trade
    confidence: float = 0.0
    rationale: str = ""
    key_events: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provider: str = "heuristic"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsSignalAdvisor(Protocol):
    """Protocol implemented by heuristic and LLM advisors."""

    def analyze(
        self,
        *,
        market: MarketSnapshot,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
    ) -> MarketAdvisorView:
        ...


_HIGH_RISK_TERMS = {
    "hack",
    "exploit",
    "lawsuit",
    "sec",
    "ban",
    "halt",
    "delist",
    "delisting",
    "insolvent",
    "bankruptcy",
    "fraud",
    "investigation",
    "outage",
    "breach",
    "regulatory",
    "sanction",
}

_BULLISH_TERMS = {
    "approval",
    "approved",
    "inflow",
    "partnership",
    "adoption",
    "upgrade",
    "bullish",
    "breakout",
    "record high",
    "accumulation",
}

_BEARISH_TERMS = {
    "selloff",
    "outflow",
    "bearish",
    "liquidation",
    "crash",
    "downgrade",
    "rejection",
    "resistance",
    "loss",
}


def _normalize_news(items: Iterable[Mapping[str, Any]] | None) -> List[NewsItem]:
    normalized: List[NewsItem] = []
    for item in items or []:
        normalized.append(
            NewsItem(
                title=str(item.get("title", item.get("headline", ""))),
                summary=str(item.get("summary", item.get("description", ""))),
                source=str(item.get("source", "manual")),
                url=str(item.get("url", "")),
                published_at=str(item.get("published_at", item.get("date", ""))),
            )
        )
    return normalized


def _normalize_signals(items: Iterable[Mapping[str, Any]] | None) -> List[MarketSignal]:
    normalized: List[MarketSignal] = []
    for item in items or []:
        normalized.append(
            MarketSignal(
                name=str(item.get("name", "signal")),
                value=item.get("value", ""),
                direction=str(item.get("direction", "neutral")).lower(),
                confidence=float(item.get("confidence", 0.5) or 0.0),
                source=str(item.get("source", "manual")),
            )
        )
    return normalized


def _contains_any(text: str, terms: set[str]) -> List[str]:
    lowered = text.lower()
    return sorted(term for term in terms if term in lowered)


class HeuristicNewsSignalAdvisor:
    """No-network fallback advisor used for paper tests and offline demos."""

    def analyze(
        self,
        *,
        market: MarketSnapshot,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
    ) -> MarketAdvisorView:
        news = _normalize_news(news_items)
        normalized_signals = _normalize_signals(signals)
        text = "\n".join(f"{item.title}. {item.summary}" for item in news)

        high_terms = _contains_any(text, _HIGH_RISK_TERMS)
        bullish_terms = _contains_any(text, _BULLISH_TERMS)
        bearish_terms = _contains_any(text, _BEARISH_TERMS)
        positive_signals = [s for s in normalized_signals if s.direction in {"up", "bullish", "positive"} and s.confidence >= 0.55]
        negative_signals = [s for s in normalized_signals if s.direction in {"down", "bearish", "negative"} and s.confidence >= 0.55]

        warnings: List[str] = []
        key_events: List[str] = []
        if high_terms:
            warnings.append("high_risk_news_terms:" + ",".join(high_terms))
            key_events.extend([item.title for item in news[:3] if item.title])
            return MarketAdvisorView(
                symbol=market.symbol,
                risk_level="high",
                news_sentiment="bearish" if bearish_terms else "mixed",
                trade_bias="no_trade",
                confidence=0.72,
                rationale="High-risk news terms were detected; safest next step is no trade or human review.",
                key_events=key_events,
                warnings=warnings,
                provider="heuristic",
            )

        bullish_score = len(bullish_terms) + len(positive_signals)
        bearish_score = len(bearish_terms) + len(negative_signals)
        if bullish_score > bearish_score:
            sentiment = "bullish"
            bias = "buy_bias"
            confidence = min(0.75, 0.5 + 0.08 * bullish_score)
        elif bearish_score > bullish_score:
            sentiment = "bearish"
            bias = "sell_bias"
            confidence = min(0.75, 0.5 + 0.08 * bearish_score)
        elif bullish_score or bearish_score:
            sentiment = "mixed"
            bias = "hold_bias"
            confidence = 0.52
        else:
            sentiment = "neutral"
            bias = "hold_bias"
            confidence = 0.45

        if positive_signals or negative_signals:
            key_events.extend([f"{s.name}={s.value} ({s.direction})" for s in [*positive_signals, *negative_signals][:4]])
        key_events.extend([item.title for item in news[:3] if item.title])
        risk_level = "medium" if sentiment == "mixed" else "low"
        return MarketAdvisorView(
            symbol=market.symbol,
            risk_level=risk_level,
            news_sentiment=sentiment,
            trade_bias=bias,
            confidence=round(confidence, 3),
            rationale="Heuristic news/signal classifier produced an advisory market bias. This is not an order.",
            key_events=key_events,
            warnings=warnings,
            provider="heuristic",
        )


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


def _advisor_payload(market: MarketSnapshot, news: List[Dict[str, Any]], signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "market": market.to_dict(),
        "news_items": news,
        "signals": signals,
        "instruction": (
            "Classify market/news risk for a guarded paper/testnet/live spot trading agent. "
            "Do not give financial advice. Do not output an order. Return only JSON with: "
            "risk_level low|medium|high|unknown, news_sentiment bullish|bearish|mixed|neutral|unknown, "
            "trade_bias buy_bias|sell_bias|hold_bias|no_trade, confidence 0..1, rationale, key_events, warnings. "
            "Use no_trade for hacks, regulatory shocks, insolvency, exchange outages, unverified high-risk news, or weak evidence."
        ),
    }


def _view_from_llm_data(*, market: MarketSnapshot, data: Mapping[str, Any], provider: str) -> MarketAdvisorView:
    return MarketAdvisorView(
        symbol=market.symbol,
        risk_level=str(data.get("risk_level", "unknown")).lower(),
        news_sentiment=str(data.get("news_sentiment", "unknown")).lower(),
        trade_bias=str(data.get("trade_bias", "hold_bias")).lower(),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rationale=str(data.get("rationale", "")),
        key_events=[str(x) for x in data.get("key_events", [])][:8],
        warnings=[str(x) for x in data.get("warnings", [])][:8],
        provider=provider,
        raw=dict(data),
    )


class GeminiNewsSignalAdvisor:
    """Optional Gemini advisor using the official Google GenAI SDK.

    Set ``GEMINI_API_KEY`` and install ``google-genai``. The advisor is still
    advisory only: it classifies news/signals and never executes orders.
    """

    def __init__(self, *, model: Optional[str] = None, temperature: float = 0.0) -> None:
        self.model = model or os.getenv("SFM_TRADING_GEMINI_MODEL", os.getenv("SFM_TRADING_LLM_MODEL", "gemini-3.5-flash"))
        self.temperature = float(temperature)
        self.fallback = HeuristicNewsSignalAdvisor()

    def analyze(
        self,
        *,
        market: MarketSnapshot,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
    ) -> MarketAdvisorView:
        news = [item.to_dict() for item in _normalize_news(news_items)]
        normalized_signals = [item.to_dict() for item in _normalize_signals(signals)]
        api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not api_key:
            view = self.fallback.analyze(market=market, news_items=news, signals=normalized_signals)
            return MarketAdvisorView(**{**view.to_dict(), "provider": "heuristic_fallback_no_gemini_key"})
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception:
            view = self.fallback.analyze(market=market, news_items=news, signals=normalized_signals)
            return MarketAdvisorView(**{**view.to_dict(), "provider": "heuristic_fallback_google_genai_not_installed"})

        payload = _advisor_payload(market, news, normalized_signals)
        prompt = (
            "You are a conservative market-risk classifier for a trading safety gate. "
            "Return only a valid JSON object, no markdown.\n\n"
            + json.dumps(payload, sort_keys=True)
        )
        try:
            client = genai.Client(api_key=api_key)
            try:
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                )
            except Exception:
                config = None
            kwargs: Dict[str, Any] = {"model": self.model, "contents": prompt}
            if config is not None:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            data = _parse_json_object(getattr(response, "text", "") or "{}")
            return _view_from_llm_data(market=market, data=data, provider=f"gemini:{self.model}")
        except Exception as exc:
            view = self.fallback.analyze(market=market, news_items=news, signals=normalized_signals)
            raw = {"llm_error": f"{type(exc).__name__}: {exc}"}
            return MarketAdvisorView(**{**view.to_dict(), "provider": "heuristic_fallback_gemini_error", "raw": raw})


class LLMNewsSignalAdvisor:
    """Optional LLM advisor supporting OpenAI or Gemini.

    ``provider`` can be ``openai``, ``gemini``, or ``auto``. In ``auto`` mode,
    Gemini is used when ``GEMINI_API_KEY`` is present; otherwise OpenAI is used
    when ``OPENAI_API_KEY`` is present; otherwise the deterministic heuristic
    fallback is used.
    """

    def __init__(self, *, model: Optional[str] = None, temperature: float = 0.0, provider: str = "auto") -> None:
        self.provider = (provider or os.getenv("SFM_TRADING_LLM_PROVIDER", "auto")).lower()
        self.model = model or os.getenv("SFM_TRADING_LLM_MODEL", "")
        self.temperature = float(temperature)
        self.fallback = HeuristicNewsSignalAdvisor()

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

    def analyze(
        self,
        *,
        market: MarketSnapshot,
        news_items: Iterable[Mapping[str, Any]] | None = None,
        signals: Iterable[Mapping[str, Any]] | None = None,
    ) -> MarketAdvisorView:
        resolved = self._resolved_provider()
        if resolved == "gemini":
            model = self.model or os.getenv("SFM_TRADING_GEMINI_MODEL", "gemini-3.5-flash")
            return GeminiNewsSignalAdvisor(model=model, temperature=self.temperature).analyze(
                market=market, news_items=news_items, signals=signals
            )
        if resolved == "heuristic":
            view = self.fallback.analyze(market=market, news_items=news_items, signals=signals)
            return MarketAdvisorView(**{**view.to_dict(), "provider": "heuristic_fallback_no_llm_key"})

        news = [item.to_dict() for item in _normalize_news(news_items)]
        normalized_signals = [item.to_dict() for item in _normalize_signals(signals)]
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            view = self.fallback.analyze(market=market, news_items=news, signals=normalized_signals)
            return MarketAdvisorView(**{**view.to_dict(), "provider": "heuristic_fallback_no_openai_key"})
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            view = self.fallback.analyze(market=market, news_items=news, signals=normalized_signals)
            return MarketAdvisorView(**{**view.to_dict(), "provider": "heuristic_fallback_openai_not_installed"})

        payload = _advisor_payload(market, news, normalized_signals)
        model = self.model or os.getenv("SFM_TRADING_OPENAI_MODEL", "gpt-4o-mini")
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a conservative market-risk classifier for a trading safety gate."},
                    {"role": "user", "content": json.dumps(payload, sort_keys=True)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            return _view_from_llm_data(market=market, data=data, provider=f"openai:{model}")
        except Exception as exc:
            view = self.fallback.analyze(market=market, news_items=news, signals=normalized_signals)
            raw = {"llm_error": f"{type(exc).__name__}: {exc}"}
            return MarketAdvisorView(**{**view.to_dict(), "provider": "heuristic_fallback_llm_error", "raw": raw})

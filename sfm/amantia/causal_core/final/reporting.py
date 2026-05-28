from __future__ import annotations

"""Human-readable reporting for SFM audit outputs.

The SFM engine intentionally returns rich structured JSON for automated gates.
This module adds a thin reporting layer for humans: product teams, safety
reviewers, governance boards, and incident-review workflows.  It does not
create new evidence; it narrates the already-computed diagnostic layers and
preserves the same epistemic limits.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class SFMReportSection:
    """One human-readable report section with machine-readable support."""

    title: str
    status: str = "unassessed"
    bullets: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SFMAuditReport:
    """Rendered SFM audit report plus compact machine summary."""

    generated: bool = False
    format: str = "markdown"
    title: str = "SFM Audit Report"
    executive_summary: str = ""
    verdict: str = "unassessed"
    gate_status: str = "review"
    goal_variable: str = ""
    observed_action: str = ""
    confidence_level: str = "none"
    markdown: str = ""
    sections: List[Dict[str, Any]] = field(default_factory=list)
    machine_summary: Dict[str, Any] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)
    limits: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _unique(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _clean_str(item)
        if text and text not in out:
            out.append(text)
    return out


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes", "y", "pass", "passed", "allow", "allowed"}:
            return True
        if low in {"false", "0", "no", "n", "fail", "failed", "block", "blocked"}:
            return False
    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_score(value: Any) -> str:
    return f"{_safe_float(value):.3f}"


def _line(label: str, value: Any) -> str:
    text = _clean_str(value, "unavailable")
    return f"- **{label}:** {text}"


def _section_to_markdown(section: SFMReportSection) -> str:
    lines = [f"## {section.title}", "", _line("Status", section.status)]
    for bullet in section.bullets:
        if bullet:
            lines.append(f"- {bullet}")
    if section.warnings:
        lines.append("- **Warnings:** " + "; ".join(section.warnings))
    if section.reason_codes:
        lines.append("- **Reason codes:** " + ", ".join(section.reason_codes[:12]))
    return "\n".join(lines).strip()


class SFMAuditReportGenerator:
    """Build a human-readable report from a full or compact SFM result."""

    def generate(self, payload: Any, *, include_raw: bool = False, title: str = "SFM Audit Report") -> SFMAuditReport:
        data = _as_dict(payload)
        summary = _as_dict(data.get("alignment_summary"))
        if not summary:
            summary = _as_dict(data.get("summary"))

        goal = _clean_str(summary.get("goal_variable") or data.get("most_likely_goal") or data.get("goal_variable"), "unknown_goal")
        action = _clean_str(summary.get("observed_action") or data.get("observed_action"), "unknown_action")
        verdict = _clean_str(summary.get("verdict") or data.get("verdict"), "unassessed")
        gate = _clean_str(summary.get("gate_status") or data.get("gate_status"), "review")
        confidence = _clean_str(summary.get("confidence_level") or data.get("confidence_level"), "none")
        intent_score = _safe_float(summary.get("intent_score", data.get("intent_score", 0.0)))
        inferred = _safe_bool(summary.get("intent_supported", data.get("inferred", False)))
        authority = _clean_str(summary.get("authority_status") or data.get("authority_status"), "diagnostic_only")

        warnings = _unique([*_as_list(summary.get("warnings")), *_as_list(data.get("limits"))])
        blocks = _unique(_as_list(summary.get("blocking_reasons")))
        codes = _unique([*_as_list(summary.get("reason_codes")), *_as_list(data.get("reason_codes"))])

        executive_summary = (
            f"Observed action `{action}` was assessed against candidate goal `{goal}`. "
            f"The SFM governance verdict is `{verdict}` with gate status `{gate}` "
            f"and confidence `{confidence}`. Intent support is `{str(inferred).lower()}` "
            f"with score {_fmt_score(intent_score)}. This report is diagnostic and should be "
            "read together with the structured JSON evidence."
        )

        sections = self._build_sections(data, summary, goal=goal, action=action, intent_score=intent_score, inferred=inferred, authority=authority)
        machine_summary = {
            "verdict": verdict,
            "gate_status": gate,
            "allow_execution": _safe_bool(summary.get("allow_execution"), False),
            "goal_variable": goal,
            "observed_action": action,
            "confidence_level": confidence,
            "intent_supported": inferred,
            "intent_score": round(intent_score, 6),
            "authority_status": authority,
            "falsification_passed": _safe_bool(summary.get("falsification_passed"), True),
            "constraints_satisfied": _safe_bool(summary.get("constraints_satisfied"), True),
            "normatively_aligned": _safe_bool(summary.get("normatively_aligned"), False),
            "prohibited": _safe_bool(summary.get("prohibited"), False),
            "requires_escalation": _safe_bool(summary.get("requires_escalation"), False),
            "robustness_status": _clean_str(summary.get("robustness_status"), "unassessed"),
            "recommended_action": _clean_str(summary.get("recommended_action")),
            "blocking_reasons": blocks,
            "warnings": warnings,
        }
        markdown = self._render_markdown(
            title=title,
            executive_summary=executive_summary,
            machine_summary=machine_summary,
            sections=sections,
        )
        return SFMAuditReport(
            generated=True,
            format="markdown",
            title=title,
            executive_summary=executive_summary,
            verdict=verdict,
            gate_status=gate,
            goal_variable=goal,
            observed_action=action,
            confidence_level=confidence,
            markdown=markdown,
            sections=[section.to_dict() for section in sections],
            machine_summary=machine_summary,
            reason_codes=codes,
            warnings=warnings,
            blocking_reasons=blocks,
            limits=_unique(_as_list(data.get("limits"))),
            raw=data if include_raw else {},
        )

    def _build_sections(
        self,
        data: Mapping[str, Any],
        summary: Mapping[str, Any],
        *,
        goal: str,
        action: str,
        intent_score: float,
        inferred: bool,
        authority: str,
    ) -> List[SFMReportSection]:
        causal = _as_dict(data.get("causal_support"))
        twin = _as_dict(data.get("twin_support"))
        belief = _as_dict(data.get("belief_support"))
        falsification = _as_dict(data.get("falsification_support"))
        constraint = _as_dict(data.get("constraint_support"))
        normative = _as_dict(data.get("normative_support"))
        robustness = _as_dict(data.get("robustness_support"))
        recommendation = _as_dict(data.get("action_recommendation_support"))
        temporal = _as_dict(data.get("temporal_goal_drift_support"))
        context = _as_dict(data.get("context_conditioning_support"))
        hierarchy = _as_dict(data.get("hierarchical_goal_support"))

        sections: List[SFMReportSection] = []
        sections.append(
            SFMReportSection(
                title="Governance verdict",
                status=_clean_str(summary.get("gate_status"), "review"),
                bullets=[
                    f"Verdict `{_clean_str(summary.get('verdict'), 'unassessed')}` for action `{action}` and goal `{goal}`.",
                    f"Confidence `{_clean_str(summary.get('confidence_level'), 'none')}` with authority `{authority}`.",
                    f"Execution allowed: `{str(_safe_bool(summary.get('allow_execution'), False)).lower()}`.",
                ],
                evidence={"alignment_summary": dict(summary)},
                warnings=_unique(_as_list(summary.get("warnings"))),
                reason_codes=_unique(_as_list(summary.get("reason_codes"))),
            )
        )
        sections.append(
            SFMReportSection(
                title="Intent evidence",
                status="supported" if inferred else "not_claimable",
                bullets=[
                    f"Intent score: {_fmt_score(intent_score)}.",
                    f"Support level: `{_clean_str(data.get('support_level'), 'none')}`.",
                    f"SCM identification tier: `{_clean_str(causal.get('identification_tier') or causal.get('identification_status'), 'unassessed')}`.",
                    f"Twin-policy action changes when goal removed: `{str(_safe_bool(twin.get('action_changes_when_goal_removed'), False)).lower()}`.",
                    f"Intent under agent beliefs: `{str(_safe_bool(belief.get('intent_under_agent_beliefs'), False)).lower()}`.",
                ],
                evidence={"causal_support": causal, "twin_support": twin, "belief_support": belief},
                reason_codes=_unique([*_as_list(causal.get("reason_codes")), *_as_list(twin.get("reason_codes")), *_as_list(belief.get("reason_codes"))]),
            )
        )
        sections.append(
            SFMReportSection(
                title="Falsification and side-effect checks",
                status="passed" if _safe_bool(data.get("falsification_passed"), True) else "failed",
                bullets=[
                    f"Falsification passed: `{str(_safe_bool(data.get('falsification_passed'), True)).lower()}`.",
                    f"Side effects excluded: `{str(_safe_bool(data.get('side_effects_excluded'), False)).lower()}`.",
                    f"Falsification status: `{_clean_str(falsification.get('falsification_status'), 'unassessed')}`.",
                ],
                evidence={"falsification_support": falsification},
                reason_codes=_unique(_as_list(falsification.get("reason_codes"))),
            )
        )
        sections.append(
            SFMReportSection(
                title="Constraints and normative alignment",
                status=_clean_str(summary.get("verdict"), "unassessed"),
                bullets=[
                    f"Constraints satisfied: `{str(_safe_bool(summary.get('constraints_satisfied'), True)).lower()}`.",
                    f"Normatively aligned: `{str(_safe_bool(summary.get('normatively_aligned'), False)).lower()}`.",
                    f"Prohibited: `{str(_safe_bool(summary.get('prohibited'), False)).lower()}`.",
                    f"Escalation required: `{str(_safe_bool(summary.get('requires_escalation'), False)).lower()}`.",
                ],
                evidence={"constraint_support": constraint, "normative_support": normative},
                reason_codes=_unique([*_as_list(constraint.get("reason_codes")), *_as_list(normative.get("reason_codes"))]),
            )
        )
        sections.append(
            SFMReportSection(
                title="Robustness and uncertainty",
                status=_clean_str(summary.get("robustness_status"), "unassessed"),
                bullets=[
                    f"Robust to uncertainty: `{str(_safe_bool(summary.get('robust_to_uncertainty'), False)).lower()}`.",
                    f"Uncertainty review required: `{str(_safe_bool(summary.get('uncertainty_review_required'), False)).lower()}`.",
                    f"Pessimistic intent score: {_fmt_score(summary.get('pessimistic_intent_score', 0.0))}.",
                ],
                evidence={"robustness_support": robustness},
                warnings=_unique(_as_list(robustness.get("warnings"))),
                reason_codes=_unique(_as_list(robustness.get("reason_codes"))),
            )
        )
        sections.append(
            SFMReportSection(
                title="Recommendation",
                status=_clean_str(summary.get("recommendation_status"), "unassessed"),
                bullets=[
                    f"Recommended action: `{_clean_str(summary.get('recommended_action') or recommendation.get('recommended_action'), 'unavailable')}`.",
                    f"Recommendation matches observed action: `{str(_safe_bool(summary.get('recommendation_matches_observed'), False)).lower()}`.",
                    f"Recommended action allowed: `{str(_safe_bool(summary.get('recommended_action_allowed'), False)).lower()}`.",
                ],
                evidence={"action_recommendation_support": recommendation},
                reason_codes=_unique(_as_list(recommendation.get("reason_codes"))),
            )
        )
        if _safe_bool(temporal.get("assessed"), False) or _safe_bool(context.get("assessed"), False) or _safe_bool(hierarchy.get("assessed"), False):
            sections.append(
                SFMReportSection(
                    title="Temporal, context, and hierarchy diagnostics",
                    status="assessed",
                    bullets=[
                        f"Temporal stability: `{_clean_str(temporal.get('stability_status'), 'unassessed')}`.",
                        f"Context-conditioned policy: `{str(_safe_bool(context.get('context_conditioning_detected'), False)).lower()}`.",
                        f"Selected ultimate goal: `{_clean_str(hierarchy.get('selected_ultimate_goal'), 'unavailable')}`.",
                    ],
                    evidence={
                        "temporal_goal_drift_support": temporal,
                        "context_conditioning_support": context,
                        "hierarchical_goal_support": hierarchy,
                    },
                    reason_codes=_unique([*_as_list(temporal.get("reason_codes")), *_as_list(context.get("reason_codes")), *_as_list(hierarchy.get("reason_codes"))]),
                )
            )
        return sections

    def _render_markdown(
        self,
        *,
        title: str,
        executive_summary: str,
        machine_summary: Mapping[str, Any],
        sections: List[SFMReportSection],
    ) -> str:
        lines = [f"# {title}", "", "## Executive summary", "", executive_summary, "", "## Machine verdict", ""]
        for key in [
            "verdict",
            "gate_status",
            "allow_execution",
            "goal_variable",
            "observed_action",
            "confidence_level",
            "intent_supported",
            "intent_score",
            "authority_status",
            "robustness_status",
            "recommended_action",
        ]:
            lines.append(_line(key.replace("_", " ").title(), machine_summary.get(key, "")))
        if machine_summary.get("blocking_reasons"):
            lines.append(_line("Blocking reasons", "; ".join(machine_summary["blocking_reasons"])))
        if machine_summary.get("warnings"):
            lines.append(_line("Warnings", "; ".join(machine_summary["warnings"])))
        lines.append("")
        lines.extend(_section_to_markdown(section) + "\n" for section in sections)
        lines.append("## Epistemic note\n")
        lines.append(
            "This report is an SFM diagnostic artifact, not a metaphysical proof of final causality. "
            "It summarizes evidence from causal, counterfactual, belief, falsification, normative, "
            "recommendation, and uncertainty layers when those layers are enabled."
        )
        return "\n".join(lines).strip() + "\n"


def render_sfm_audit_report(payload: Mapping[str, Any], *, include_raw: bool = False) -> Dict[str, Any]:
    """Convenience helper for generating a report from an SFM result dict."""

    return SFMAuditReportGenerator().generate(payload, include_raw=include_raw).to_dict()

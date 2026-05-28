from __future__ import annotations

"""Authority classification for SCM identification results.

This module does not perform identification.  It classifies which component is
allowed to be treated as the public authority for a result after the ID facade
has gathered recursive-ID, backdoor/frontdoor, factorization, and hedge
signals.  Diagnostic modules may explain or provide estimator-friendly shortcut
formulas; the final authority status is explicit and machine-readable here.
"""

from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Optional


def _s(value: object) -> str:
    return "" if value is None else str(value).strip()


def _json(payload: object) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps({"serialization_status": "failed", "repr": repr(payload)}, sort_keys=True)


@dataclass(frozen=True)
class IDAuthorityDiagnostic:
    """Public authority classification for one ID result."""

    authority_status: str
    identification_authority: str
    authority_basis: str = ""
    authority_reason_codes: str = ""
    diagnostic_roles_json: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def id_authority_diagnostic(
    *,
    identifiable: bool,
    id_strategy: str,
    id_algorithm_level: str,
    recursive: Optional[object] = None,
    backdoor: Optional[object] = None,
    frontdoor: Optional[object] = None,
    factorization: Optional[object] = None,
    hedge: Optional[object] = None,
    formal_hedge: Optional[object] = None,
    reason_codes: str = "",
    failure_reason: str = "",
) -> IDAuthorityDiagnostic:
    """Classify the final authority behind an ID row.

    Policy:
    - recursive-ID is the preferred positive authority whenever it identified a
      formula;
    - backdoor/frontdoor/factorization remain diagnostics or estimator-friendly
      shortcut routes, and are marked as shortcuts when used;
    - a formal hedge is an impossibility authority only for non-identified rows;
    - weak hedge diagnostics never become formal non-identification authority.
    """

    strategy = _s(id_strategy)
    level = _s(id_algorithm_level)
    recursive_status = _s(getattr(recursive, "recursive_status", "")) if recursive is not None else ""
    recursive_identified = bool(getattr(recursive, "recursive_identified", False)) if recursive is not None else False
    formal_hedge_certified = bool(getattr(formal_hedge, "formal_hedge_certified", False)) if formal_hedge is not None else False

    roles: List[Dict[str, object]] = []
    if recursive is not None:
        roles.append({
            "component": "recursive_id",
            "role": "authority" if recursive_identified else "blocked_or_pending_authority_candidate",
            "status": recursive_status,
            "identified": recursive_identified,
            "blocker_class": _s(getattr(recursive, "recursive_blocker_class", "")),
            "pending_operator": _s(getattr(recursive, "recursive_pending_operator", "")),
        })
    if backdoor is not None:
        roles.append({
            "component": "backdoor_criterion",
            "role": "diagnostic_shortcut",
            "status": _s(getattr(backdoor, "backdoor_status", "")),
            "passed": bool(getattr(backdoor, "backdoor_ok", False)),
        })
    if frontdoor is not None:
        roles.append({
            "component": "frontdoor_criterion",
            "role": "diagnostic_shortcut",
            "status": _s(getattr(frontdoor, "frontdoor_status", "")),
            "passed": bool(getattr(frontdoor, "frontdoor_ok", False)),
        })
    if factorization is not None:
        roles.append({
            "component": "observed_dag_factorization",
            "role": "base_case_diagnostic",
            "status": _s(getattr(factorization, "factorization_status", "")),
            "passed": bool(getattr(factorization, "factorization_ok", False)),
        })
    if hedge is not None:
        roles.append({
            "component": "simple_hedge_diagnostic",
            "role": "warning_only_unless_confirmed_by_formal_fail_branch",
            "status": _s(getattr(hedge, "hedge_status", "")),
            "possible_hedge": bool(getattr(hedge, "possible_hedge", False)),
        })
    if formal_hedge is not None:
        roles.append({
            "component": "formal_hedge",
            "role": "nonidentification_authority" if (formal_hedge_certified and not identifiable) else "diagnostic_not_final_authority",
            "status": _s(getattr(formal_hedge, "formal_hedge_status", "")),
            "certified": bool(formal_hedge_certified and not identifiable),
        })

    diagnostic_roles_json = _json(roles)
    reasons = _s(reason_codes) or _s(failure_reason)

    if identifiable:
        if recursive_identified:
            basis = recursive_status or "recursive_id_identified"
            if strategy in {"backdoor_adjustment", "frontdoor_limited", "observed_dag_truncated_factorization"}:
                basis = f"recursive_id_confirmed_{strategy}:{basis}"
            return IDAuthorityDiagnostic(
                "identified_authoritative",
                "recursive_id",
                basis,
                reasons or "RECURSIVE_ID_AUTHORITY_IDENTIFIED",
                diagnostic_roles_json,
            )
        if strategy == "no_directed_effect":
            return IDAuthorityDiagnostic(
                "identified_authoritative_graphical_zero_effect",
                "graphical_zero_effect",
                "no_directed_path_base_case",
                reasons or "NO_DIRECTED_PATH_ZERO_EFFECT",
                diagnostic_roles_json,
            )
        if strategy in {"backdoor_adjustment", "frontdoor_limited", "observed_dag_truncated_factorization"}:
            return IDAuthorityDiagnostic(
                "identified_limited_shortcut_without_recursive_authority",
                "limited_graphical_shortcut",
                strategy,
                reasons or "LIMITED_SHORTCUT_USED_WITHOUT_RECURSIVE_AUTHORITY",
                diagnostic_roles_json,
            )
        return IDAuthorityDiagnostic(
            "identified_authority_unspecified",
            "unknown_positive_route",
            level or strategy,
            reasons or "IDENTIFIED_AUTHORITY_UNSPECIFIED",
            diagnostic_roles_json,
        )

    if formal_hedge_certified:
        return IDAuthorityDiagnostic(
            "nonidentified_authoritative_formal_hedge",
            "recursive_id_fail_branch",
            "formal_hedge_certificate",
            reasons or _s(getattr(formal_hedge, "hedge_reason_codes", "")) or "FORMAL_HEDGE_CERTIFICATE",
            diagnostic_roles_json,
        )

    if recursive is not None and recursive_status:
        return IDAuthorityDiagnostic(
            "blocked_by_recursive_id_pending_full_id",
            "recursive_id",
            recursive_status,
            reasons or _s(getattr(recursive, "reason_codes", "")) or "RECURSIVE_ID_BLOCKED_OR_PENDING",
            diagnostic_roles_json,
        )

    return IDAuthorityDiagnostic(
        "blocked_conservative_no_authoritative_nonidentification_certificate",
        "conservative_router",
        level or strategy,
        reasons or "BLOCKED_WITHOUT_FORMAL_HEDGE_CERTIFICATE",
        diagnostic_roles_json,
    )


__all__ = ["IDAuthorityDiagnostic", "id_authority_diagnostic"]

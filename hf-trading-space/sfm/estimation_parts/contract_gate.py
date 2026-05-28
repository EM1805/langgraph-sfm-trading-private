"""Contract authority helpers for the estimation layer.

This module is intentionally lightweight.  It centralizes the rules that decide
whether an insight is allowed to produce a Pearl-style causal estimate.  The
rules are derived from the canonical ``out/causal_contract.csv`` handoff.

Important: this module does not estimate effects and does not discover graph
structure.  It only explains authorization status.
"""
from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


from typing import Dict, Iterable, List

import pandas as pd

from . import _utils as U


RAW_AUTHORITY_TOKENS = {"raw_discovery_only", "pcmci_raw_input_only", "discovery_only"}
AUTHORIZED_LEVELS = {"identified_estimable"}
AUTHORIZED_STATUSES = {
    "identified",
    "backdoor_identified",
    "frontdoor_identified",
    "identified_backdoor",
}

REPORT_COLUMNS = [
    "insight_id",
    "contract_required",
    "contract_row_present",
    "identified",
    "identification_status",
    "identification_strategy",
    "authority_level",
    "source_authority",
    "adjustment_set",
    "forbidden_adjustment_set",
    "estimator_selected",
    "pearl_estimation_authorized",
    "causal_claim_status",
    "pearl_authority_reason",
    "reason_codes",
]


as_str = U.as_str
safe_float = U.safe_float
parse_list = U.parse_list

def parse_source_authority(value) -> set[str]:
    return {x.strip().lower() for x in as_str(value).split("|") if x.strip()}


def adjustment_status_is_valid(meta: Dict[str, object]) -> bool:
    status = as_str(meta.get("adjustment_set_status", "")).lower()
    if status in {"valid_empty", "valid_nonempty"}:
        return True
    if status in {"missing", "invalid"}:
        return False
    if parse_list(meta.get("adjustment_set", meta.get("suggested_adjustment_set", ""))):
        return True
    mins = as_str(meta.get("minimal_adjustment_sets", ""))
    return mins.startswith("[[") or mins == "[[]]"


def pearl_contract_authority(insight_meta: Dict[str, object]) -> Dict[str, object]:
    """Return Pearl-estimation authority for an insight metadata row.

    The result is deliberately verbose so downstream outputs can explain why an
    insight was accepted, downgraded to diagnostic-only, or rejected.
    """
    required = int(safe_float(insight_meta.get("__contract_required", 0), 0)) == 1
    present = int(safe_float(insight_meta.get("__contract_row_present", 0), 0)) == 1
    authority_level = as_str(insight_meta.get("authority_level", insight_meta.get("graph_authority_level", ""))).lower()
    status = as_str(insight_meta.get("identification_status", insight_meta.get("graph_identification_status", ""))).lower()
    strategy = as_str(insight_meta.get("identification_strategy", insight_meta.get("graph_identification_strategy", ""))).lower()
    source_authority = as_str(insight_meta.get("source_authority", insight_meta.get("contract_source_authority", ""))).lower()
    identified_flag = int(safe_float(insight_meta.get("identified", insight_meta.get("graph_identified", 0)), 0)) == 1

    source_parts = parse_source_authority(source_authority)
    raw_only = bool(source_parts) and source_parts.issubset(RAW_AUTHORITY_TOKENS)

    authorized = False
    if required and not present:
        reason = "missing_causal_contract_row"
    elif raw_only or authority_level in {"raw_discovery_only", "discovery_only", "weak_or_unaligned"}:
        reason = "raw_discovery_not_pearl_authority"
    else:
        estimator_enabled = as_str(insight_meta.get("estimation_enabled", "")).lower() in {"1", "true", "yes"}
        adj = as_str(insight_meta.get("adjustment_set", insight_meta.get("suggested_adjustment_set", "")))
        adjustment_valid = adjustment_status_is_valid(insight_meta)
        backdoor_like = ("backdoor" in strategy) or adjustment_valid or bool(adj)
        if authority_level in AUTHORIZED_LEVELS and (estimator_enabled or (backdoor_like and adjustment_valid)):
            authorized = True
            reason = "formal_backdoor_identification_authorized"
        elif identified_flag and backdoor_like and adjustment_valid and authority_level not in {"identified_needs_estimation"}:
            authorized = True
            reason = "formal_backdoor_identification_authorized"
        elif identified_flag or authority_level == "identified_needs_estimation" or status in AUTHORIZED_STATUSES:
            reason = "identified_but_no_enabled_estimator"
        elif present:
            reason = "contract_present_but_not_formally_identified"
        else:
            reason = "legacy_no_contract_mode"

    return {
        "contract_required": int(required),
        "contract_row_present": int(present),
        "pearl_estimation_authorized": int(authorized),
        "causal_claim_status": "pearl_authorized_for_estimation" if authorized else "diagnostic_only_not_pearl_authorized",
        "pearl_authority_reason": reason,
        "authority_level": authority_level,
        "identification_status": status,
        "identification_strategy": strategy,
        "source_authority": source_authority,
        "identified": int(identified_flag),
    }


def _iter_contract_meta(contract: pd.DataFrame | None, *, contract_required: bool) -> Iterable[Dict[str, object]]:
    if contract is None or len(contract) == 0:
        if contract_required:
            yield {
                "insight_id": "",
                "__contract_required": 1,
                "__contract_row_present": 0,
            }
        return
    for _, row in contract.iterrows():
        meta = row.to_dict()
        meta["__contract_required"] = 1
        meta["__contract_row_present"] = 1
        yield meta


def build_authority_report(
    insights: pd.DataFrame | None,
    contract: pd.DataFrame | None,
    *,
    contract_required: bool = False,
) -> pd.DataFrame:
    """Build one row per insight/contract row explaining estimation authority."""
    rows: List[Dict[str, object]] = []

    if contract is not None and len(contract) > 0:
        iterable = _iter_contract_meta(contract, contract_required=True)
    elif insights is not None and len(insights) > 0:
        iterable = []
        for _, row in insights.iterrows():
            meta = row.to_dict()
            meta["__contract_required"] = int(contract_required)
            meta["__contract_row_present"] = 0
            iterable.append(meta)
    else:
        iterable = _iter_contract_meta(contract, contract_required=contract_required)

    for meta in iterable:
        auth = pearl_contract_authority(meta)
        estimator = "none"
        reason_codes = as_str(auth.get("pearl_authority_reason", "")).upper()
        if int(auth.get("pearl_estimation_authorized", 0)) == 1:
            strategy = as_str(meta.get("identification_strategy", meta.get("graph_identification_strategy", ""))).lower()
            adj = as_str(meta.get("adjustment_set", meta.get("suggested_adjustment_set", "")))
            if "backdoor" in strategy or "adjustment" in strategy or adj:
                estimator = "backdoor_ridge_adjustment"
                reason_codes = ""
            elif "frontdoor" in strategy:
                estimator = "not_enabled_frontdoor"
                reason_codes = "FRONTDOOR_ESTIMATOR_NOT_ENABLED"
            else:
                estimator = "not_enabled_unknown_estimand"
                reason_codes = "ESTIMATOR_NOT_ENABLED_FOR_STRATEGY"

        rows.append({
            "insight_id": as_str(meta.get("insight_id", "")),
            "contract_required": int(auth.get("contract_required", 0)),
            "contract_row_present": int(auth.get("contract_row_present", 0)),
            "identified": int(auth.get("identified", 0)),
            "identification_status": as_str(auth.get("identification_status", "")),
            "identification_strategy": as_str(auth.get("identification_strategy", "")),
            "authority_level": as_str(auth.get("authority_level", "")),
            "source_authority": as_str(auth.get("source_authority", "")),
            "adjustment_set": as_str(meta.get("adjustment_set", meta.get("suggested_adjustment_set", ""))),
            "forbidden_adjustment_set": as_str(meta.get("forbidden_adjustment_set", meta.get("forbidden_controls", ""))),
            "estimator_selected": estimator,
            "pearl_estimation_authorized": int(auth.get("pearl_estimation_authorized", 0)),
            "causal_claim_status": as_str(auth.get("causal_claim_status", "")),
            "pearl_authority_reason": as_str(auth.get("pearl_authority_reason", "")),
            "reason_codes": reason_codes,
        })

    out = pd.DataFrame(rows)
    for col in REPORT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[REPORT_COLUMNS].copy()

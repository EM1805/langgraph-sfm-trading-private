from __future__ import annotations

from runtime_env import configure_scientific_runtime
configure_scientific_runtime()


"""Contract gate for strong do-operator estimation.

Strong ``do_authorized`` remains strict.  Less-conservative policies only allow
clearly labelled diagnostic estimates from contract-backed rows; they never
upgrade those diagnostics into causal authority.
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

STRICT = "strict"
BALANCED = "balanced"
EXPLORATORY = "exploratory"
VALID_POLICIES = {STRICT, BALANCED, EXPLORATORY}
DEFAULT_DO_POLICY = BALANCED


@dataclass(frozen=True)
class DoPolicy:
    name: str = DEFAULT_DO_POLICY
    allow_diagnostic_estimates: bool = True
    require_contract_row_for_diagnostic: bool = True
    allow_missing_estimation_enabled_for_diagnostic: bool = True
    allow_needs_estimation_authority_for_diagnostic: bool = True
    allow_missing_adjustment_status_if_adjustment_present: bool = True
    allow_frontdoor_chain_lite_diagnostic: bool = True

    @property
    def strong_authority_is_strict(self) -> bool:
        return True


def normalize_policy(value: object = None) -> str:
    raw = str(value or os.environ.get("AMANTIA_DO_POLICY") or DEFAULT_DO_POLICY).strip().lower()
    if raw in {"safe", "safety", "conservative"}:
        return STRICT
    if raw in {"default", "balanced", "review"}:
        return BALANCED
    if raw in {"explore", "exploratory", "diagnostic"}:
        return EXPLORATORY
    return DEFAULT_DO_POLICY


def get_do_policy(value: object = None) -> DoPolicy:
    name = normalize_policy(value)
    if name == STRICT:
        return DoPolicy(
            name=STRICT,
            allow_diagnostic_estimates=False,
            require_contract_row_for_diagnostic=True,
            allow_missing_estimation_enabled_for_diagnostic=False,
            allow_needs_estimation_authority_for_diagnostic=False,
            allow_missing_adjustment_status_if_adjustment_present=True,
            allow_frontdoor_chain_lite_diagnostic=False,
        )
    if name == EXPLORATORY:
        return DoPolicy(
            name=EXPLORATORY,
            allow_diagnostic_estimates=True,
            require_contract_row_for_diagnostic=True,
            allow_missing_estimation_enabled_for_diagnostic=True,
            allow_needs_estimation_authority_for_diagnostic=True,
            allow_missing_adjustment_status_if_adjustment_present=True,
            allow_frontdoor_chain_lite_diagnostic=True,
        )
    return DoPolicy(name=BALANCED)


def join_reason_codes(codes: Iterable[str]) -> str:
    out: List[str] = []
    for code in codes:
        c = str(code or "").strip()
        if c and c not in out:
            out.append(c)
    return "|".join(out)




BACKDOOR_DO_SUPPORTED_STRATEGIES = {"backdoor", "backdoor_adjustment", "adjustment", "backdoor_matching"}
FRONTDOOR_DO_SUPPORTED_STRATEGIES = {"frontdoor", "frontdoor_adjustment", "frontdoor_formula"}
STRONG_DO_SUPPORTED_STRATEGIES = BACKDOOR_DO_SUPPORTED_STRATEGIES | FRONTDOOR_DO_SUPPORTED_STRATEGIES
VALID_ADJUSTMENT_STATUSES = {"valid_empty", "valid_nonempty"}
VALID_FRONTDOOR_STATUSES = {"frontdoor_valid", "valid_frontdoor", "frontdoor_mediators_valid", "valid_limited_frontdoor"}
ID_SYMBOLIC_OK = {"", "identified_symbolic_formula"}
ID_HARD_BLOCK_PREFIXES = ("blocked", "unsupported_requires_full_id")
ID_HARD_BLOCK_FRAGMENTS = ("possible_hedge", "requires_symbolic_c_factor", "directed_cycle", "invalid_backdoor", "invalid_frontdoor")
CANONICAL_ID_AUTHORITY_ARTIFACTS = {"id_algorithm_audit"}
CANONICAL_ID_AUTHORITY_SOURCES = {"scm_id_algorithm"}


@dataclass(frozen=True)
class DoAuthorization:
    do_authorized: bool
    do_mode: str
    treatment: str
    outcome: str
    authority_level: str = ""
    identification_strategy: str = ""
    id_status: str = ""
    symbolic_formula_status: str = ""
    hedge_detected: int = 0
    recursive_id_status: str = ""
    c_factor_status: str = ""
    district_status: str = ""
    adjustment_set: str = ""
    adjustment_set_status: str = ""
    mediators: str = ""
    frontdoor_status: str = ""
    effect_claim_authority: str = ""
    estimation_enabled: int = 0
    contract_row_present: int = 0
    contract_index: int = -1
    canonical_id_authority: int = 0
    id_algorithm_level: str = ""
    source_artifacts: str = ""
    source_authority: str = ""
    analysis_policy: str = "balanced"
    diagnostic_estimation_allowed: int = 0
    diagnostic_authority_level: str = ""
    causal_authority_from_diagnostic: int = 0
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _norm(value: object) -> str:
    s = str(value or "").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _truthy(value: object) -> bool:
    s = _norm(value).lower()
    if s in {"1", "true", "yes", "y", "enabled", "identified_estimable"}:
        return True
    if s in {"0", "false", "no", "n", "disabled", ""}:
        return False
    try:
        return float(s) != 0.0
    except (TypeError, ValueError, OverflowError):
        return False



def _id_contract_block_reasons(row: Dict[str, object]) -> List[str]:
    """Hard ID-audit blockers that do-contract must never override."""
    reasons: List[str] = []
    id_status = _norm(row.get("id_status") or row.get("identification_status")).lower()
    symbolic = _norm(row.get("symbolic_formula_status")).lower()
    hedge = _norm(row.get("hedge_status")).lower()
    recursive = _norm(row.get("recursive_id_status")).lower()
    cfactor = _norm(row.get("c_factor_status")).lower()
    district = _norm(row.get("district_status")).lower()
    if id_status.startswith(ID_HARD_BLOCK_PREFIXES) or any(tok in id_status for tok in ID_HARD_BLOCK_FRAGMENTS):
        reasons.append("ID_STATUS_BLOCKED")
    if symbolic not in ID_SYMBOLIC_OK:
        reasons.append("ID_SYMBOLIC_FORMULA_NOT_IDENTIFIED")
    if _truthy(row.get("hedge_detected")) or "possible_hedge" in hedge:
        reasons.append("ID_HEDGE_DETECTED")
    if recursive.startswith("blocked") or "requires_symbolic_c_factor" in recursive:
        reasons.append("ID_RECURSIVE_BLOCKED")
    if "unresolved" in cfactor or "requires_recursive" in cfactor:
        reasons.append("ID_C_FACTOR_UNRESOLVED")
    if "possible_hedge" in district:
        reasons.append("ID_DISTRICT_POSSIBLE_HEDGE")
    return reasons


def _id_symbolic_formula_ok(row: Dict[str, object]) -> bool:
    return _norm(row.get("symbolic_formula_status")).lower() in ID_SYMBOLIC_OK


def _pipe_tokens(value: object) -> List[str]:
    return [tok.strip().lower() for tok in _norm(value).replace(",", "|").split("|") if tok.strip()]


def has_canonical_id_authority(row: Dict[str, object]) -> bool:
    """Return True only when a row carries canonical SCM-ID authority.

    Legacy identification/reporting rows may still describe a backdoor/frontdoor
    pattern, but strong do-estimation must be gated by the canonical
    id_algorithm audit or by a row that mirrors its canonical metadata.
    """
    artifacts = set(_pipe_tokens(row.get("source_artifacts")))
    sources = set(_pipe_tokens(row.get("source_authority")))
    if artifacts & CANONICAL_ID_AUTHORITY_ARTIFACTS:
        return True
    if sources & CANONICAL_ID_AUTHORITY_SOURCES:
        return True
    if _truthy(row.get("canonical_id_available")):
        return True
    # Step 30 legacy mirrors expose canonical_id_level as id_algorithm_level.
    # Treat that as canonical provenance; old Pearl-lite rows do not carry it.
    if _norm(row.get("id_algorithm_level")):
        return True
    return False


def _join_codes(codes: Iterable[str]) -> str:
    return join_reason_codes(codes)


def load_causal_contract(out_dir: str = "out", contract_path: Optional[str] = None) -> pd.DataFrame:
    path = Path(contract_path or os.path.join(out_dir, "causal_contract.csv"))
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return pd.DataFrame()


def _match_contract_rows(contract_df: pd.DataFrame, treatment: str, outcome: str) -> pd.DataFrame:
    if contract_df is None or contract_df.empty:
        return pd.DataFrame()
    t = _norm(treatment)
    y = _norm(outcome)
    cols_t = [c for c in ["treatment_col", "source", "cause", "from"] if c in contract_df.columns]
    cols_y = [c for c in ["outcome_col", "target", "effect", "to"] if c in contract_df.columns]
    if not cols_t or not cols_y:
        return pd.DataFrame()
    mask = pd.Series(False, index=contract_df.index)
    for tc in cols_t:
        for yc in cols_y:
            mask = mask | ((contract_df[tc].astype(str).str.strip() == t) & (contract_df[yc].astype(str).str.strip() == y))
    return contract_df.loc[mask].copy()


def _select_best_contract_row(rows: pd.DataFrame) -> tuple[int, Dict[str, object]]:
    if rows is None or rows.empty:
        return -1, {}

    def score(row: pd.Series) -> tuple:
        estimation_enabled = 1 if _truthy(row.get("estimation_enabled")) else 0
        authority = _norm(row.get("authority_level")).lower()
        authority_score = 3 if authority == "identified_estimable" else 2 if "identified" in authority else 0
        strategy = _norm(row.get("identification_strategy")).lower()
        strategy_score = 2 if strategy in STRONG_DO_SUPPORTED_STRATEGIES or "backdoor" in strategy or "frontdoor" in strategy else 0
        adj_status = _norm(row.get("adjustment_set_status")).lower()
        fd_status = _norm(row.get("frontdoor_status") or row.get("frontdoor_verification_level")).lower()
        med = _norm(row.get("mediators"))
        adj_score = 1 if (adj_status in VALID_ADJUSTMENT_STATUSES or fd_status in VALID_FRONTDOOR_STATUSES or med) else 0
        id_blocked = 1 if _id_contract_block_reasons(row.to_dict()) else 0
        id_symbolic = 1 if _id_symbolic_formula_ok(row.to_dict()) else 0
        return (-id_blocked, estimation_enabled, authority_score, strategy_score, adj_score, id_symbolic)

    best_idx = max(rows.index, key=lambda idx: score(rows.loc[idx]))
    return int(best_idx), rows.loc[best_idx].to_dict()


def _diagnostic_allowed(
    *,
    row: Dict[str, object],
    policy: DoPolicy,
    is_backdoor: bool,
    is_frontdoor: bool,
    authority_level: str,
    estimation_enabled: int,
    adjustment_set_status: str,
    frontdoor_status: str,
    mediators: str,
) -> tuple[bool, List[str], str, str]:
    reasons: List[str] = []
    adj_status = adjustment_set_status
    fd_status = frontdoor_status
    if not policy.allow_diagnostic_estimates:
        return False, ["DIAGNOSTIC_POLICY_STRICT"], adj_status, fd_status
    if policy.require_contract_row_for_diagnostic and not row:
        reasons.append("NO_CONTRACT_FOR_DIAGNOSTIC")
    if not (is_backdoor or is_frontdoor):
        reasons.append("UNSUPPORTED_DIAGNOSTIC_STRATEGY")
    if authority_level == "identified_estimable":
        pass
    elif "identified" in authority_level and policy.allow_needs_estimation_authority_for_diagnostic:
        pass
    else:
        reasons.append("DIAGNOSTIC_REQUIRES_IDENTIFIED_CONTRACT_ROW")
    if not estimation_enabled and not policy.allow_missing_estimation_enabled_for_diagnostic:
        reasons.append("DIAGNOSTIC_REQUIRES_ESTIMATION_ENABLED")
    if is_backdoor:
        if adj_status not in VALID_ADJUSTMENT_STATUSES:
            if policy.allow_missing_adjustment_status_if_adjustment_present and _norm(row.get("adjustment_set")):
                adj_status = "valid_nonempty"
            else:
                reasons.append("DIAGNOSTIC_BACKDOOR_ADJUSTMENT_NOT_VALID")
    if is_frontdoor:
        fd_ok = bool(mediators) and (fd_status in VALID_FRONTDOOR_STATUSES or (policy.allow_frontdoor_chain_lite_diagnostic and fd_status == "frontdoor_chain_complete_lite"))
        if not fd_ok:
            reasons.append("DIAGNOSTIC_FRONTDOOR_NOT_READY")
    return len(reasons) == 0, reasons, adj_status, fd_status


def authorize_do(treatment: str, outcome: str, out_dir: str = "out", contract_df: Optional[pd.DataFrame] = None, contract_path: Optional[str] = None, policy: object = None) -> DoAuthorization:
    """Authorize a strong ``do(treatment)`` estimate from the causal contract."""
    do_policy = get_do_policy(policy)
    contract = load_causal_contract(out_dir, contract_path) if contract_df is None else contract_df
    rows = _match_contract_rows(contract, treatment, outcome)
    idx, row = _select_best_contract_row(rows)
    if not row:
        return DoAuthorization(False, "blocked", _norm(treatment), _norm(outcome), contract_row_present=0, analysis_policy=do_policy.name, diagnostic_estimation_allowed=0, diagnostic_authority_level="blocked_no_contract", reason_codes="NO_CAUSAL_CONTRACT_ROW")

    reasons: List[str] = []
    authority_level = _norm(row.get("authority_level"))
    strategy = _norm(row.get("identification_strategy")).lower()
    adjustment_set_status = _norm(row.get("adjustment_set_status")).lower()
    frontdoor_status = _norm(row.get("frontdoor_status") or row.get("frontdoor_verification_level")).lower()
    mediators = _norm(row.get("mediators"))
    estimation_enabled = 1 if _truthy(row.get("estimation_enabled")) else 0
    id_status = _norm(row.get("id_status") or row.get("identification_status")).lower()
    symbolic_formula_status = _norm(row.get("symbolic_formula_status")).lower()
    hedge_detected = 1 if _truthy(row.get("hedge_detected")) else 0
    recursive_id_status = _norm(row.get("recursive_id_status")).lower()
    c_factor_status = _norm(row.get("c_factor_status")).lower()
    district_status = _norm(row.get("district_status")).lower()
    id_block_reasons = _id_contract_block_reasons(row)
    canonical_id_authority = 1 if has_canonical_id_authority(row) else 0

    if not canonical_id_authority:
        reasons.append("MISSING_CANONICAL_ID_AUTHORITY")
    if id_block_reasons:
        reasons.extend(id_block_reasons)
    if symbolic_formula_status and symbolic_formula_status != "identified_symbolic_formula":
        reasons.append("SYMBOLIC_FORMULA_NOT_IDENTIFIED")
    if not estimation_enabled:
        reasons.append("ESTIMATION_NOT_ENABLED")
    if authority_level != "identified_estimable":
        reasons.append("NOT_IDENTIFIED_ESTIMABLE")
    is_backdoor = strategy in BACKDOOR_DO_SUPPORTED_STRATEGIES or "backdoor" in strategy
    is_frontdoor = strategy in FRONTDOOR_DO_SUPPORTED_STRATEGIES or "frontdoor" in strategy
    if not (is_backdoor or is_frontdoor):
        reasons.append("UNSUPPORTED_IDENTIFICATION_STRATEGY")
    if is_backdoor:
        if adjustment_set_status and adjustment_set_status not in VALID_ADJUSTMENT_STATUSES:
            reasons.append("INVALID_ADJUSTMENT_SET_STATUS")
        if not adjustment_set_status:
            if _norm(row.get("adjustment_set")):
                adjustment_set_status = "valid_nonempty"
            else:
                reasons.append("MISSING_ADJUSTMENT_SET_STATUS")
    if is_frontdoor:
        if not mediators:
            reasons.append("MISSING_FRONTDOOR_MEDIATORS")
        if frontdoor_status and frontdoor_status not in VALID_FRONTDOOR_STATUSES and frontdoor_status != "frontdoor_chain_complete_lite":
            reasons.append("INVALID_FRONTDOOR_STATUS")
        if not frontdoor_status:
            frontdoor_status = "frontdoor_valid" if estimation_enabled else "missing"

    authorized = len(reasons) == 0
    diagnostic_allowed, diagnostic_reasons, adjustment_set_status, frontdoor_status = _diagnostic_allowed(
        row=row,
        policy=do_policy,
        is_backdoor=is_backdoor,
        is_frontdoor=is_frontdoor,
        authority_level=authority_level,
        estimation_enabled=estimation_enabled,
        adjustment_set_status=adjustment_set_status,
        frontdoor_status=frontdoor_status,
        mediators=mediators,
    ) if not authorized else (False, [], adjustment_set_status, frontdoor_status)
    if diagnostic_allowed and not canonical_id_authority:
        diagnostic_allowed = False
        diagnostic_reasons = list(diagnostic_reasons) + ["DIAGNOSTIC_REQUIRES_CANONICAL_ID_AUTHORITY"]

    if authorized and is_frontdoor:
        mode = "identified_frontdoor"
        reason_codes = ["DO_AUTHORIZED_BY_FRONTDOOR_CONTRACT"]
    elif authorized and is_backdoor:
        mode = "identified_backdoor"
        reason_codes = ["DO_AUTHORIZED_BY_BACKDOOR_CONTRACT"]
    elif diagnostic_allowed and is_frontdoor:
        mode = "diagnostic_frontdoor_candidate"
        reason_codes = reasons + ["DIAGNOSTIC_ESTIMATE_ALLOWED_NO_CAUSAL_AUTHORITY"]
    elif diagnostic_allowed and is_backdoor:
        mode = "diagnostic_backdoor_candidate"
        reason_codes = reasons + ["DIAGNOSTIC_ESTIMATE_ALLOWED_NO_CAUSAL_AUTHORITY"]
    else:
        mode = "blocked"
        reason_codes = reasons + diagnostic_reasons

    return DoAuthorization(
        do_authorized=authorized,
        do_mode=mode,
        treatment=_norm(treatment),
        outcome=_norm(outcome),
        authority_level=authority_level,
        identification_strategy=strategy,
        id_status=id_status,
        symbolic_formula_status=symbolic_formula_status,
        hedge_detected=hedge_detected,
        recursive_id_status=recursive_id_status,
        c_factor_status=c_factor_status,
        district_status=district_status,
        adjustment_set=_norm(row.get("adjustment_set")),
        adjustment_set_status=adjustment_set_status,
        mediators=mediators,
        frontdoor_status=frontdoor_status,
        effect_claim_authority=_norm(row.get("effect_claim_authority")),
        estimation_enabled=estimation_enabled,
        contract_row_present=1,
        contract_index=idx,
        canonical_id_authority=canonical_id_authority,
        id_algorithm_level=_norm(row.get("id_algorithm_level")),
        source_artifacts=_norm(row.get("source_artifacts")),
        source_authority=_norm(row.get("source_authority")),
        analysis_policy=do_policy.name,
        diagnostic_estimation_allowed=int(bool(diagnostic_allowed)),
        diagnostic_authority_level=("diagnostic_do_estimate_not_causal_authority" if diagnostic_allowed else ""),
        causal_authority_from_diagnostic=0,
        reason_codes=_join_codes(reason_codes),
    )


def authorize_all_backdoor_do(contract_df: pd.DataFrame, policy: object = None, include_diagnostic: bool = True, include_blocked_audit: bool = True) -> List[DoAuthorization]:
    decisions: List[DoAuthorization] = []
    if contract_df is None or contract_df.empty:
        return decisions
    for _, row in contract_df.iterrows():
        treatment = _norm(row.get("treatment_col") or row.get("source"))
        outcome = _norm(row.get("outcome_col") or row.get("target"))
        strategy = _norm(row.get("identification_strategy")).lower()
        route = _norm(row.get("identification_route")).lower()
        route_text = " ".join([strategy, route])
        if "frontdoor" in route_text and "backdoor" not in route_text:
            continue
        if not treatment or not outcome:
            continue
        decision = authorize_do(treatment, outcome, contract_df=pd.DataFrame([row]), policy=policy)
        if (
            decision.do_mode == "identified_backdoor"
            or (include_diagnostic and decision.do_mode == "diagnostic_backdoor_candidate")
            or (include_blocked_audit and "backdoor" in decision.identification_strategy)
        ):
            decisions.append(decision)
    seen = set()
    out: List[DoAuthorization] = []
    for d in decisions:
        key = (d.treatment, d.outcome, d.do_mode, d.contract_index, d.analysis_policy)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def authorize_all_frontdoor_do(contract_df: pd.DataFrame, policy: object = None, include_diagnostic: bool = True, include_blocked_audit: bool = True) -> List[DoAuthorization]:
    decisions: List[DoAuthorization] = []
    if contract_df is None or contract_df.empty:
        return decisions
    for _, row in contract_df.iterrows():
        strategy = _norm(row.get("identification_strategy")).lower()
        route = _norm(row.get("identification_route")).lower()
        if "frontdoor" not in " ".join([strategy, route]):
            continue
        treatment = _norm(row.get("treatment_col") or row.get("source"))
        outcome = _norm(row.get("outcome_col") or row.get("target"))
        if not treatment or not outcome:
            continue
        decision = authorize_do(treatment, outcome, contract_df=pd.DataFrame([row]), policy=policy)
        if decision.do_authorized or (include_diagnostic and decision.diagnostic_estimation_allowed) or include_blocked_audit:
            decisions.append(decision)
    seen = set()
    out: List[DoAuthorization] = []
    for d in decisions:
        if not include_diagnostic and not d.do_authorized:
            continue
        key = (d.treatment, d.outcome, d.do_mode, d.contract_index, d.analysis_policy)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


__all__ = [
    "DoAuthorization",
    "BACKDOOR_DO_SUPPORTED_STRATEGIES",
    "FRONTDOOR_DO_SUPPORTED_STRATEGIES",
    "STRONG_DO_SUPPORTED_STRATEGIES",
    "VALID_ADJUSTMENT_STATUSES",
    "VALID_FRONTDOOR_STATUSES",
    "load_causal_contract",
    "authorize_do",
    "authorize_all_backdoor_do",
    "authorize_all_frontdoor_do",
    "has_canonical_id_authority",
    "CANONICAL_ID_AUTHORITY_ARTIFACTS",
    "CANONICAL_ID_AUTHORITY_SOURCES",
    "DoPolicy",
    "get_do_policy",
]

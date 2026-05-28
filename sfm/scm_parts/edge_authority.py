from __future__ import annotations

"""SCM edge authority helpers.

SCM edges are candidates. Raw Discovery rows are seed-only, bridge/ranked rows
can be eligible for graphical identification, and estimation is authorized only
by causal_contract.csv.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from ._utils import edge_endpoints_from_row


def _as_str(x) -> str:
    return "" if x is None else str(x)


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if v == v else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _first_present(row, names, default=""):
    for name in names:
        try:
            val = row.get(name, "")
        except (TypeError, ValueError, AttributeError):
            val = ""
        if val is not None and str(val).strip() != "":
            return val
    return default


def _row_has_any(row, names: Iterable[str]) -> bool:
    for name in names:
        try:
            val = row.get(name, "")
        except (TypeError, ValueError, AttributeError):
            val = ""
        if val is not None and str(val).strip() != "":
            return True
    return False


def _source_artifact(row, default: str = "") -> str:
    return _as_str(_first_present(row, ["__source_artifact", "source_artifact", "bridge_source_artifact"], default)).strip()


def _normalized_artifact_name(value: str) -> str:
    return _as_str(value).strip().lower().replace("\\", "/")


def _artifact_has(artifact_l: str, *needles: str) -> bool:
    return any(needle in artifact_l for needle in needles)


def _explicit_pcmci_scm_shape(row) -> bool:
    """True only for explicit PCMCI-SCM metadata, not generic conditioning fields."""
    layer = _as_str(_first_present(row, ["edge_source_layer"], "")).strip().lower()
    if layer and layer != "pcmci_scm_bridge":
        return False
    return _row_has_any(row, ["scm_role_hint", "mci_conditioning_set_used"]) or layer == "pcmci_scm_bridge"


def authority_from_row(row, default_origin: str = "") -> Dict[str, object]:
    artifact = _source_artifact(row, default_origin)
    art_l = _normalized_artifact_name(artifact)
    has_bridge_shape = _row_has_any(row, [
        "bridge_version", "candidate_covariates", "post_treatment_columns",
        "negative_control_col", "discovery_effect_proxy",
    ])
    has_ranked_shape = _row_has_any(row, [
        "insight_id", "discovery_confidence_tier", "mci_q_value",
        "selection_score", "discovery_effect_proxy",
    ])

    # Authority is artifact-first. Generic diagnostic columns such as
    # ``conditioning_set_used`` can appear in bridge/handoff files and must not
    # upgrade an edge to a PCMCI-SCM structural prior by themselves.
    is_scm_input_artifact = _artifact_has(art_l, "scm_input")
    is_pcmci_scm_bridge_artifact = _artifact_has(art_l, "pcmci_scm_bridge")
    is_discovery_estimation_bridge_artifact = _artifact_has(
        art_l,
        "discovery_estimation_bridge",
        "estimation_handoff",
        "causal_contract",
    )

    if is_scm_input_artifact or _row_has_any(row, ["scm_input_edge_kind", "domain_review_status"]):
        edge_kind = _as_str(_first_present(row, ["edge_kind", "scm_input_edge_kind"], "")).strip().lower()
        hard_blocked = edge_kind in {"exogenous_noise", "noise_input", "latent_noise"}
        return {
            "edge_authority_level": "domain_scm_prior",
            "edge_source_artifact": artifact or "scm_input.json",
            "edge_source_layer": "scm_input",
            "bridge_version": _as_str(row.get("bridge_version", "amantia.scm_input.v1")),
            "raw_seed_only": False,
            "eligible_for_identification": not hard_blocked,
            "eligible_for_estimation": False,
            "is_formally_identified": False,
            "authority_rank": 40,
            "authority_reason_codes": "SCM_INPUT_DOMAIN_PRIOR|IDENTIFICATION_REQUIRED_BEFORE_ESTIMATION" + ("|EXOGENOUS_NOISE_NOT_A_DO_QUERY" if hard_blocked else ""),
        }
    if is_discovery_estimation_bridge_artifact:
        return {
            "edge_authority_level": "bridge_candidate",
            "edge_source_artifact": artifact or "out/discovery_estimation_bridge.csv",
            "edge_source_layer": "discovery_bridge",
            "bridge_version": _as_str(row.get("bridge_version", "")),
            "raw_seed_only": False,
            "eligible_for_identification": True,
            "eligible_for_estimation": False,
            "is_formally_identified": False,
            "authority_rank": 30,
            "authority_reason_codes": "BRIDGE_CANDIDATE|NOT_ESTIMATION_AUTHORITY|EXPLICIT_DISCOVERY_ESTIMATION_BRIDGE",
        }
    if is_pcmci_scm_bridge_artifact or _explicit_pcmci_scm_shape(row):
        role_hint = _as_str(row.get("scm_role_hint", "")).strip().lower()
        hard_blocked = role_hint in {"do_not_use_hard_blocked", "discovery_rejected_diagnostic"}
        return {
            "edge_authority_level": "pcmci_scm_structural_prior",
            "edge_source_artifact": artifact or "out/pcmci_scm_bridge.csv",
            "edge_source_layer": "pcmci_scm_bridge",
            "bridge_version": _as_str(row.get("bridge_version", "")),
            "raw_seed_only": False,
            "eligible_for_identification": not hard_blocked,
            "eligible_for_estimation": False,
            "is_formally_identified": False,
            "authority_rank": 35,
            "authority_reason_codes": "PCMCI_SCM_STRUCTURAL_PRIOR|IDENTIFICATION_REQUIRED_BEFORE_ESTIMATION" + ("|ROLE_HINT_NOT_FOR_IDENTIFICATION" if hard_blocked else ""),
        }
    if "bridge" in art_l or has_bridge_shape:
        return {
            "edge_authority_level": "bridge_candidate",
            "edge_source_artifact": artifact or "out/discovery_estimation_bridge.csv",
            "edge_source_layer": "discovery_bridge",
            "bridge_version": _as_str(row.get("bridge_version", "")),
            "raw_seed_only": False,
            "eligible_for_identification": True,
            "eligible_for_estimation": False,
            "is_formally_identified": False,
            "authority_rank": 30,
            "authority_reason_codes": "BRIDGE_CANDIDATE|NOT_ESTIMATION_AUTHORITY",
        }
    if "insights" in art_l or "ranking" in art_l or has_ranked_shape:
        return {
            "edge_authority_level": "ranked_discovery_candidate",
            "edge_source_artifact": artifact or "out/insights_level2.csv",
            "edge_source_layer": "discovery_ranking",
            "bridge_version": "",
            "raw_seed_only": False,
            "eligible_for_identification": True,
            "eligible_for_estimation": False,
            "is_formally_identified": False,
            "authority_rank": 20,
            "authority_reason_codes": "RANKED_DISCOVERY_CANDIDATE|NOT_ESTIMATION_AUTHORITY",
        }
    return {
        "edge_authority_level": "raw_discovery_seed_only",
        "edge_source_artifact": artifact or "out/edges.csv",
        "edge_source_layer": "raw_discovery_seed",
        "bridge_version": "",
        "raw_seed_only": True,
        "eligible_for_identification": False,
        "eligible_for_estimation": False,
        "is_formally_identified": False,
        "authority_rank": 10,
        "authority_reason_codes": "RAW_SEED_ONLY|REQUIRES_BRIDGE_OR_INSIGHT_BEFORE_IDENTIFICATION",
    }


def _safe_int_from_row(row, name: str, default: int = 1) -> int:
    return int(_safe_float(row.get(name, default), default))


def build_scm_edge_row(row, default_origin: str = "") -> Optional[Dict[str, object]]:
    src, tgt = edge_endpoints_from_row(row)
    if not src or not tgt:
        return None
    auth = authority_from_row(row, default_origin=default_origin)
    out: Dict[str, object] = {
        "source": src,
        "target": tgt,
        "treatment_col": src,
        "outcome_col": tgt,
        "lag": _safe_int_from_row(row, "lag", 1),
        "edge_kind": _as_str(_first_present(row, ["edge_kind", "scm_input_edge_kind"], "lagged_structural_candidate")) or "lagged_structural_candidate",
        "confidence_tier": _as_str(_first_present(row, ["discovery_confidence_tier", "discovery_track", "confidence_tier"], "")) or "candidate",
        "mci_q_value": _safe_float(row.get("mci_q_value", ""), float("nan")),
        "selection_score": _safe_float(row.get("selection_score", ""), 0.0),
        "parent_set": _as_str(_first_present(row, ["parent_set", "pc1_parent_set_all", "pc1_parent_set", "candidate_covariates", "suggested_adjustment_set"], "")),
        "conditioning_set_used": _as_str(_first_present(row, ["conditioning_set_used", "mci_conditioning_set_used", "candidate_covariates", "suggested_adjustment_set"], "")),
        "conditioning_set_size": _safe_float(_first_present(row, ["conditioning_set_size", "mci_conditioning_set_size"], ""), 0.0),
        "conditioning_quality": _as_str(_first_present(row, ["conditioning_quality", "mci_conditioning_quality"], "")),
        "mci_status": _as_str(row.get("mci_status", "")),
        "scm_role_hint": _as_str(row.get("scm_role_hint", "")),
        "identification_priority": _safe_float(row.get("identification_priority", row.get("selection_score", "")), 0.0),
        "risk_flags": _as_str(row.get("risk_flags", "")),
        "insight_id": _as_str(row.get("insight_id", "")),
        "discovery_effect_proxy": _as_str(_first_present(row, ["discovery_effect_proxy", "causal_effect"], "")),
        "effect_proxy_semantics": _as_str(row.get("effect_proxy_semantics", "")) or "discovery_screening_proxy_not_causal_claim",
        "candidate_covariates": _as_str(row.get("candidate_covariates", "")),
        "post_treatment_columns": _as_str(row.get("post_treatment_columns", "")),
        "forbidden_adjustment_set": _as_str(row.get("forbidden_adjustment_set", "")),
    }
    out.update(auth)
    return out


def dedupe_edges_by_authority(edge_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in edge_rows:
        key = (_as_str(row.get("source", "")), _as_str(row.get("target", "")), int(_safe_float(row.get("lag", 1), 1)))
        if key not in best or int(row.get("authority_rank", 0)) > int(best[key].get("authority_rank", 0)):
            best[key] = row
        elif int(row.get("authority_rank", 0)) == int(best[key].get("authority_rank", 0)):
            prev = best[key]
            prev_art = _as_str(prev.get("edge_source_artifact", ""))
            art = _as_str(row.get("edge_source_artifact", ""))
            if art and art not in prev_art.split("|"):
                prev["edge_source_artifact"] = "|".join([x for x in [prev_art, art] if x])
    for row in best.values():
        row.pop("authority_rank", None)
    return list(best.values())

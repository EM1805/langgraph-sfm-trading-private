from __future__ import annotations

"""Limited graph-route diagnostics used by the conservative ID layer.

This module keeps backdoor, frontdoor, and hedge-style graphical audits out of
``id_algorithm.py``. They are intentionally conservative helpers: they can
validate the limited adjustment/front-door routes Amantia already supports, or
produce hedge-risk evidence, but they do not claim to be the full recursive ID
algorithm.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

from .admg import ADMG
from .graph_criteria import d_separation_diagnostic, directed_paths
from .id_algorithm_common import _dedupe, _format_component, _format_components, _format_paths, _s


@dataclass(frozen=True)
class BackdoorDiagnostic:
    """Limited graphical audit for a supplied backdoor adjustment set."""

    backdoor_status: str
    backdoor_ok: bool
    adjustment_set: str = ""
    open_paths: str = ""
    descendant_controls: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def backdoor_diagnostic(admg: ADMG, treatment: str, outcome: str, adjustment_set: Sequence[str]) -> BackdoorDiagnostic:
    x = _s(treatment)
    y = _s(outcome)
    z = [v for v in _dedupe(adjustment_set or []) if v in admg.node_set and v not in {x, y}]
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return BackdoorDiagnostic("invalid_query", False, "|".join(z), reason_codes="MISSING_QUERY_NODE")
    descendants = admg.descendants([x]) - {x}
    descendant_controls = sorted(set(z) & descendants)
    if descendant_controls:
        return BackdoorDiagnostic(
            "blocked_descendant_controls",
            False,
            "|".join(z),
            descendant_controls="|".join(descendant_controls),
            reason_codes="ADJUSTMENT_CONTAINS_DESCENDANT_OF_TREATMENT",
        )
    dsep = d_separation_diagnostic(admg, x, y, conditioned_on=z, remove_outgoing_from=[x])
    if dsep.separated:
        return BackdoorDiagnostic(
            "valid_backdoor_adjustment",
            True,
            "|".join(z),
            reason_codes="BACKDOOR_PATHS_BLOCKED_BY_ADJUSTMENT_SET",
        )
    return BackdoorDiagnostic(
        "invalid_backdoor_adjustment_open_paths",
        False,
        "|".join(z),
        open_paths=dsep.open_paths,
        reason_codes="BACKDOOR_PATHS_REMAIN_OPEN",
    )


@dataclass(frozen=True)
class HedgeDiagnostic:
    """Conservative hedge-risk diagnostic.

    This is intentionally not a formal hedge proof. It exposes the evidence a
    future recursive ID implementation needs: ancestral districts of Y and the
    c-component witness connecting treatment/outcome through latent structure.
    """

    hedge_status: str
    possible_hedge: bool
    hedge_witness: str = ""
    ancestral_c_components: str = ""
    treatment_ancestral_district: str = ""
    outcome_ancestral_district: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def hedge_diagnostic(admg: ADMG, treatment: str, outcome: str) -> HedgeDiagnostic:
    """Return conservative hedge/c-component diagnostics for ``P(Y | do(X))``."""
    x = _s(treatment)
    y = _s(outcome)
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return HedgeDiagnostic("invalid_query", False, reason_codes="MISSING_QUERY_NODE")
    if not admg.bidirected_edges:
        return HedgeDiagnostic("not_needed_no_bidirected_edges", False, reason_codes="NO_BIDIRECTED_EDGES")

    ancestral = admg.ancestral_subgraph([y])
    ancestral_components = ancestral.districts()
    components_text = _format_components(ancestral_components)
    x_district: List[str] = []
    y_district: List[str] = []
    for district in ancestral_components:
        if x in district:
            x_district = district
        if y in district:
            y_district = district

    if not x_district:
        return HedgeDiagnostic(
            "not_in_outcome_ancestral_graph",
            False,
            ancestral_c_components=components_text,
            outcome_ancestral_district=_format_component(y_district) if y_district else "",
            reason_codes="TREATMENT_NOT_IN_ANCESTORS_OF_OUTCOME",
        )

    if x_district and y_district and set(x_district) == set(y_district) and len(x_district) > 1:
        witness = _format_component(x_district)
        return HedgeDiagnostic(
            "possible_hedge_same_ancestral_c_component",
            True,
            hedge_witness=witness,
            ancestral_c_components=components_text,
            treatment_ancestral_district=witness,
            outcome_ancestral_district=witness,
            reason_codes="POSSIBLE_HEDGE_X_Y_SAME_ANCESTRAL_C_COMPONENT",
        )

    x_district_desc = admg.descendants(x_district or []) if x_district else set()
    if y in x_district_desc and len(x_district) > 1:
        witness = _format_component(x_district)
        return HedgeDiagnostic(
            "possible_hedge_treatment_c_component_reaches_outcome",
            True,
            hedge_witness=witness,
            ancestral_c_components=components_text,
            treatment_ancestral_district=witness,
            outcome_ancestral_district=_format_component(y_district) if y_district else "",
            reason_codes="POSSIBLE_HEDGE_X_C_COMPONENT_HAS_DIRECTED_ROUTE_TO_Y",
        )

    return HedgeDiagnostic(
        "no_ancestral_hedge_witness_found",
        False,
        ancestral_c_components=components_text,
        treatment_ancestral_district=_format_component(x_district) if x_district else "",
        outcome_ancestral_district=_format_component(y_district) if y_district else "",
        reason_codes="NO_ANCESTRAL_C_COMPONENT_WITNESS",
    )


@dataclass(frozen=True)
class FrontdoorDiagnostic:
    """Pearl-like limited front-door diagnostic for a supplied mediator set."""

    frontdoor_status: str
    frontdoor_ok: bool
    active_mediators: str = ""
    directed_paths_checked: int = 0
    unmediated_directed_paths: str = ""
    x_to_mediator_open_paths: str = ""
    mediator_to_y_open_paths: str = ""
    witness_paths: str = ""
    reason_codes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def frontdoor_diagnostic(admg: ADMG, treatment: str, outcome: str, mediators: Sequence[str]) -> FrontdoorDiagnostic:
    """Verify limited front-door conditions with d-separation diagnostics.

    Conditions implemented here:
    1. every directed X->Y path is intercepted by at least one supplied mediator;
    2. there is no open backdoor path from X to each active mediator;
    3. every backdoor path from each active mediator to Y is blocked by X.

    This is still a limited audit layer, not a full recursive ID proof.
    """
    x = _s(treatment)
    y = _s(outcome)
    meds = [m for m in _dedupe(mediators) if m in admg.node_set and m not in {x, y}]
    if not x or not y or x not in admg.node_set or y not in admg.node_set:
        return FrontdoorDiagnostic("invalid_query", False, reason_codes="MISSING_QUERY_NODE")
    if not meds:
        return FrontdoorDiagnostic("missing_mediators", False, reason_codes="NO_VALID_MEDIATORS_SUPPLIED")

    paths = directed_paths(admg, x, y)
    if not paths:
        return FrontdoorDiagnostic("no_directed_effect", False, reason_codes="NO_DIRECTED_PATH")
    med_set = set(meds)
    unmediated = [p for p in paths if not (set(p[1:-1]) & med_set)]
    path_nodes = set().union(*(set(p[1:-1]) for p in paths)) if paths else set()
    active_meds = sorted(med_set & path_nodes)
    if unmediated:
        return FrontdoorDiagnostic(
            "failed_unmediated_directed_paths",
            False,
            active_mediators="|".join(active_meds),
            directed_paths_checked=len(paths),
            unmediated_directed_paths=_format_paths(unmediated),
            witness_paths=_format_paths(paths),
            reason_codes="FRONTDOOR_CONDITION_1_FAILED_UNMEDIATED_DIRECTED_PATH",
        )
    if not active_meds:
        return FrontdoorDiagnostic(
            "failed_no_active_mediator_on_directed_paths",
            False,
            directed_paths_checked=len(paths),
            witness_paths=_format_paths(paths),
            reason_codes="FRONTDOOR_NO_MEDIATOR_ON_DIRECTED_PATH",
        )

    x_to_z_open: List[str] = []
    for z in active_meds:
        dsep = d_separation_diagnostic(admg, x, z, conditioned_on=[], remove_outgoing_from=[x])
        if not dsep.separated:
            x_to_z_open.append(f"{z}:{dsep.open_paths}")
    if x_to_z_open:
        return FrontdoorDiagnostic(
            "failed_x_to_mediator_backdoor_open",
            False,
            active_mediators="|".join(active_meds),
            directed_paths_checked=len(paths),
            x_to_mediator_open_paths="|".join(x_to_z_open),
            witness_paths=_format_paths(paths),
            reason_codes="FRONTDOOR_CONDITION_2_FAILED_X_TO_MEDIATOR_BACKDOOR_OPEN",
        )

    z_to_y_open: List[str] = []
    for z in active_meds:
        dsep = d_separation_diagnostic(admg, z, y, conditioned_on=[x], remove_outgoing_from=[z])
        if not dsep.separated:
            z_to_y_open.append(f"{z}:{dsep.open_paths}")
    if z_to_y_open:
        return FrontdoorDiagnostic(
            "failed_mediator_to_y_backdoor_open_given_x",
            False,
            active_mediators="|".join(active_meds),
            directed_paths_checked=len(paths),
            mediator_to_y_open_paths="|".join(z_to_y_open),
            witness_paths=_format_paths(paths),
            reason_codes="FRONTDOOR_CONDITION_3_FAILED_MEDIATOR_TO_OUTCOME_BACKDOOR_OPEN",
        )

    return FrontdoorDiagnostic(
        "valid_limited_frontdoor",
        True,
        active_mediators="|".join(active_meds),
        directed_paths_checked=len(paths),
        witness_paths=_format_paths(paths),
        reason_codes="LIMITED_FRONTDOOR_CRITERIA_PASSED",
    )


def _frontdoor_limited_ok(admg: ADMG, treatment: str, outcome: str, mediators: Sequence[str]) -> bool:
    return frontdoor_diagnostic(admg, treatment, outcome, mediators).frontdoor_ok


__all__ = [
    "BackdoorDiagnostic",
    "FrontdoorDiagnostic",
    "HedgeDiagnostic",
    "backdoor_diagnostic",
    "frontdoor_diagnostic",
    "hedge_diagnostic",
    "_frontdoor_limited_ok",
]

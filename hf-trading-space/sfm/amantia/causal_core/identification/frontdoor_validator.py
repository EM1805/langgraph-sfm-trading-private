from __future__ import annotations

"""Conservative online front-door mediator validation for SCM-ID.

This module installs a small runtime wrapper around ``IdentificationEngine``.
It handles the classical front-door shape when callers supply explicit mediator
variables.  It is deliberately narrow: it validates graph structure and returns
machine-readable diagnostics, but it does not claim arbitrary Full-ID coverage.
"""

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _dedupe(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value).strip() if value is not None else ""
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _neighbors_directed(source: str, directed: Sequence[Tuple[str, str]]) -> List[str]:
    return [b for a, b in directed if a == source]


def _directed_paths(source: str, target: str, directed: Sequence[Tuple[str, str]], *, max_depth: int) -> List[List[str]]:
    paths: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(source, [source])]
    while stack:
        node, path = stack.pop()
        if len(path) > max_depth + 1:
            continue
        for nxt in _neighbors_directed(node, directed):
            if nxt in path:
                continue
            candidate = path + [nxt]
            if nxt == target:
                paths.append(candidate)
            else:
                stack.append((nxt, candidate))
    return paths


def _path_text(path: Sequence[str]) -> str:
    return "->".join(path)


def _blocked_result(engine: Any, query: Any, *, strategy: str, reason: str, reason_codes: Sequence[str], raw: Mapping[str, Any]) -> Any:
    return engine._blocked_result(
        strategy=strategy,
        reason=reason,
        reason_codes=list(reason_codes),
        query=query,
        raw=dict(raw),
    )


def _frontdoor_formula(x: str, y: str, mediators: Sequence[str]) -> str:
    m = ",".join(mediators)
    return f"sum_{{{m}}} P({m} | {x}) sum_{{{x}'}} P({y} | {x}',{m}) P({x}')"


def _validate_frontdoor(engine: Any, query: Any) -> Any:
    nodes, directed, bidirected = engine._normalized_graph_components(query.scm_graph)
    x = query.treatment
    y = query.outcome
    mediators = _dedupe(query.mediators)
    max_depth = max(1, int(query.max_depth or 8))

    raw: Dict[str, Any] = {
        "adapter": "online_frontdoor_mediator_validator",
        "nodes_checked": sorted(nodes),
        "directed_edges": list(directed),
        "bidirected_edges": list(bidirected),
        "mediators_supplied": mediators,
        "frontdoor_paths_checked": [],
        "directed_paths_x_to_y": [],
        "open_backdoor_paths_x_to_mediator": [],
        "open_backdoor_paths_mediator_to_y": [],
        "full_id_claim_allowed": 0,
    }

    if len(query.treatments) != 1 or len(query.outcomes) != 1:
        return _blocked_result(
            engine,
            query,
            strategy="blocked_frontdoor_query_scope",
            reason="The online front-door validator only handles one treatment and one outcome.",
            reason_codes=["FRONTDOOR_SINGLE_QUERY_ONLY"],
            raw=raw,
        )
    if not mediators:
        return _blocked_result(
            engine,
            query,
            strategy="blocked_missing_frontdoor_mediator",
            reason="A front-door query requires at least one explicit mediator.",
            reason_codes=["MISSING_FRONTDOOR_MEDIATOR"],
            raw=raw,
        )

    missing_nodes = [node for node in [x, y, *mediators] if node not in nodes]
    if missing_nodes:
        raw["missing_nodes"] = missing_nodes
        return _blocked_result(
            engine,
            query,
            strategy="blocked_invalid_frontdoor_mediator_set",
            reason="Treatment, outcome, and all front-door mediators must appear in the graph.",
            reason_codes=["FRONTDOOR_NODE_NOT_IN_GRAPH"],
            raw=raw,
        )

    bad_mediators = [m for m in mediators if m in {x, y}]
    if bad_mediators:
        raw["invalid_mediators"] = bad_mediators
        return _blocked_result(
            engine,
            query,
            strategy="blocked_invalid_frontdoor_mediator_set",
            reason="A front-door mediator cannot be the treatment or outcome.",
            reason_codes=["FRONTDOOR_MEDIATOR_EQUALS_TREATMENT_OR_OUTCOME"],
            raw=raw,
        )

    mediator_set = set(mediators)
    directed_paths = _directed_paths(x, y, directed, max_depth=max_depth)
    raw["directed_paths_x_to_y"] = [_path_text(path) for path in directed_paths]
    if not directed_paths:
        return _blocked_result(
            engine,
            query,
            strategy="blocked_invalid_frontdoor_mediator_set",
            reason="No directed treatment-to-outcome path was found for the supplied front-door mediator set.",
            reason_codes=["FRONTDOOR_NO_DIRECTED_TREATMENT_OUTCOME_PATH"],
            raw=raw,
        )

    bypassing = [path for path in directed_paths if not mediator_set.intersection(path[1:-1])]
    if bypassing:
        raw["directed_paths_bypassing_mediators"] = [_path_text(path) for path in bypassing]
        return _blocked_result(
            engine,
            query,
            strategy="blocked_invalid_frontdoor_mediator_set",
            reason="The supplied mediators do not intercept every directed path from treatment to outcome.",
            reason_codes=["FRONTDOOR_DIRECT_PATH_BYPASSES_MEDIATOR"],
            raw=raw,
        )

    missing_x_to_m: List[str] = []
    missing_m_to_y: List[str] = []
    for mediator in mediators:
        x_to_m = _directed_paths(x, mediator, directed, max_depth=max_depth)
        m_to_y = _directed_paths(mediator, y, directed, max_depth=max_depth)
        raw["frontdoor_paths_checked"].append({
            "mediator": mediator,
            "x_to_mediator_paths": [_path_text(path) for path in x_to_m],
            "mediator_to_y_paths": [_path_text(path) for path in m_to_y],
        })
        if not x_to_m:
            missing_x_to_m.append(mediator)
        if not m_to_y:
            missing_m_to_y.append(mediator)
    if missing_x_to_m or missing_m_to_y:
        raw["mediators_missing_x_to_m_paths"] = missing_x_to_m
        raw["mediators_missing_m_to_y_paths"] = missing_m_to_y
        return _blocked_result(
            engine,
            query,
            strategy="blocked_invalid_frontdoor_mediator_set",
            reason="Every front-door mediator must lie on a directed treatment-to-outcome chain.",
            reason_codes=["FRONTDOOR_MEDIATOR_NOT_ON_DIRECTED_CHAIN"],
            raw=raw,
        )

    open_x_to_m: List[str] = []
    for mediator in mediators:
        paths = engine._simple_paths(x, mediator, directed, bidirected, max_depth)
        backdoor_paths = [path for path in paths if engine._first_step_is_backdoor(path, x, directed, bidirected)]
        active = [path for path in backdoor_paths if engine._path_is_active(path, set(), directed, bidirected)]
        open_x_to_m.extend(_path_text(path) for path in active)
    raw["open_backdoor_paths_x_to_mediator"] = open_x_to_m
    if open_x_to_m:
        return _blocked_result(
            engine,
            query,
            strategy="blocked_invalid_frontdoor_mediator_set",
            reason="A front-door mediator has an open backdoor path from treatment to mediator.",
            reason_codes=["FRONTDOOR_BACKDOOR_X_TO_MEDIATOR_OPEN"],
            raw=raw,
        )

    open_m_to_y: List[str] = []
    z_block = {x}
    for mediator in mediators:
        paths = engine._simple_paths(mediator, y, directed, bidirected, max_depth)
        backdoor_paths = [path for path in paths if engine._first_step_is_backdoor(path, mediator, directed, bidirected)]
        active = [path for path in backdoor_paths if engine._path_is_active(path, z_block, directed, bidirected)]
        open_m_to_y.extend(_path_text(path) for path in active)
    raw["open_backdoor_paths_mediator_to_y"] = open_m_to_y
    if open_m_to_y:
        return _blocked_result(
            engine,
            query,
            strategy="blocked_invalid_frontdoor_mediator_set",
            reason="Backdoor paths from mediator to outcome are not blocked by treatment.",
            reason_codes=["FRONTDOOR_BACKDOOR_MEDIATOR_TO_OUTCOME_OPEN"],
            raw=raw,
        )

    formula = _frontdoor_formula(x, y, mediators)
    return engine.IdentificationResult(
        identified=True,
        treatment=x,
        outcome=y,
        treatments=[x],
        outcomes=[y],
        conditions=list(query.conditions),
        adjustment_set=list(query.adjustment_set),
        mediators=mediators,
        identification_strategy="validated_frontdoor_mediator",
        identification_tier="identified_frontdoor",
        estimand=formula,
        formula=formula,
        authority_status="validated_graphical_frontdoor_adapter",
        reason="The online front-door validator accepted the mediator set after checking directed-path interception and required backdoor closures.",
        reason_codes=["SCM_ID_IDENTIFIED", "VALID_FRONTDOOR_MEDIATOR_SET"],
        raw_id_result=raw,
        query_id=query.query_id,
        source=query.source,
    )


def install_frontdoor_mediator_validator(engine: Any) -> None:
    """Install a narrow front-door validation route on IdentificationEngine."""

    IdentificationEngine = engine.IdentificationEngine
    IdentificationQuery = engine.IdentificationQuery
    if getattr(IdentificationEngine, "_amantia_step91_frontdoor_validator_installed", False):
        return

    original_identify = IdentificationEngine.identify

    def _identify_with_frontdoor_validator(self, payload):
        query = payload if isinstance(payload, IdentificationQuery) else IdentificationQuery.from_payload(payload)
        wants_frontdoor = bool(query.mediators) or "frontdoor" in str(query.strategy_hint or "").lower()
        if not wants_frontdoor:
            return original_identify(self, query)
        if not engine._has_graph(query.scm_graph) or not query.treatments or not query.outcomes:
            return original_identify(self, query)
        return _validate_frontdoor(engine, query)

    IdentificationEngine.identify = _identify_with_frontdoor_validator
    IdentificationEngine._amantia_step91_frontdoor_validator_installed = True
    IdentificationEngine._amantia_step91_original_identify = original_identify


__all__ = ["install_frontdoor_mediator_validator"]

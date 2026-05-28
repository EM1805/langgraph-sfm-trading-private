from __future__ import annotations

"""Formula AST utilities for Amantia's SCM ID layer.

Step 3 toward Full ID: keep the existing human-readable formulas, but mirror
them into a deterministic, machine-readable abstract syntax tree.  The AST is
deliberately small and dependency-free so future recursive ID steps can carry
non-joint Q inputs without parsing free-text formulas.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


ID_AST_VERSION = "id_ast_v1"


def _s(value: object) -> str:
    raw = "" if value is None else str(value).strip()
    return "" if raw.lower() in {"nan", "none", "null"} else raw


def _dedupe(values: Iterable[object]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for value in values or []:
        item = _s(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _split_vars(value: object) -> Tuple[str, ...]:
    text = _s(value)
    if not text:
        return tuple()
    for ch in "{}[]'\"":
        text = text.replace(ch, "")
    return _dedupe(part.strip() for part in re.split(r"[,|]", text) if part.strip())


@dataclass(frozen=True)
class FormulaAST:
    """Small AST node used by the ID formula layer.

    ``node_type`` is intentionally generic: "probability", "q_factor", "sum",
    "product", "fraction", "do", "hedge_fail", or "placeholder".  Nodes may
    carry children and optional metadata; consumers should ignore unknown keys.
    """

    node_type: str
    variables: Tuple[str, ...] = tuple()
    conditioned_on: Tuple[str, ...] = tuple()
    interventions: Tuple[str, ...] = tuple()
    bound_variables: Tuple[str, ...] = tuple()
    children: Tuple["FormulaAST", ...] = tuple()
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "ast_version": ID_AST_VERSION,
            "node_type": self.node_type,
        }
        if self.variables:
            payload["variables"] = list(self.variables)
        if self.conditioned_on:
            payload["conditioned_on"] = list(self.conditioned_on)
        if self.interventions:
            payload["interventions"] = list(self.interventions)
        if self.bound_variables:
            payload["bound_variables"] = list(self.bound_variables)
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        if self.label:
            payload["label"] = self.label
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def P(variables: Sequence[object], given: Sequence[object] = (), interventions: Sequence[object] = (), *, label: str = "") -> FormulaAST:
    return FormulaAST(
        "probability",
        variables=_dedupe(variables),
        conditioned_on=_dedupe(given),
        interventions=_dedupe(interventions),
        label=label,
    )


def Q(district: Sequence[object], terms: Sequence[FormulaAST] = (), *, q_input: str = "", label: str = "") -> FormulaAST:
    meta = {"q_input": q_input} if q_input else {}
    return FormulaAST("q_factor", variables=_dedupe(district), children=tuple(terms), label=label, metadata=meta)


def Sum(bound_variables: Sequence[object], expr: FormulaAST, *, label: str = "") -> FormulaAST:
    return FormulaAST("sum", bound_variables=_dedupe(bound_variables), children=(expr,), label=label)


def Product(exprs: Sequence[FormulaAST], *, label: str = "") -> FormulaAST:
    items = tuple(expr for expr in exprs if expr is not None)
    if len(items) == 1:
        return items[0]
    return FormulaAST("product", children=items, label=label)


def Fraction(numerator: FormulaAST, denominator: FormulaAST, *, label: str = "") -> FormulaAST:
    return FormulaAST("fraction", children=(numerator, denominator), label=label)


def Do(interventions: Sequence[object], expr: FormulaAST, *, label: str = "") -> FormulaAST:
    return FormulaAST("do", interventions=_dedupe(interventions), children=(expr,), label=label)


def HedgeFail(F: Sequence[object], F_prime: Sequence[object], *, roots: Sequence[object] = (), label: str = "") -> FormulaAST:
    return FormulaAST(
        "hedge_fail",
        variables=_dedupe(F),
        conditioned_on=_dedupe(F_prime),
        bound_variables=_dedupe(roots),
        label=label,
        metadata={"F": list(_dedupe(F)), "F_prime": list(_dedupe(F_prime)), "roots": list(_dedupe(roots))},
    )


def Placeholder(label: str, *, metadata: Optional[Mapping[str, object]] = None) -> FormulaAST:
    return FormulaAST("placeholder", label=_s(label), metadata=dict(metadata or {}))


def ast_from_dict(payload: Mapping[str, object]) -> FormulaAST:
    node_type = _s(payload.get("node_type")) or "placeholder"
    children_payload = payload.get("children", [])
    children: List[FormulaAST] = []
    if isinstance(children_payload, list):
        for child in children_payload:
            if isinstance(child, Mapping):
                children.append(ast_from_dict(child))
            else:
                children.append(Placeholder(str(child)))
    metadata = payload.get("metadata", {})
    return FormulaAST(
        node_type,
        variables=_dedupe(payload.get("variables", [])),
        conditioned_on=_dedupe(payload.get("conditioned_on", [])),
        interventions=_dedupe(payload.get("interventions", [])),
        bound_variables=_dedupe(payload.get("bound_variables", [])),
        children=tuple(children),
        label=_s(payload.get("label")),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def ast_to_dict(ast: FormulaAST) -> dict:
    return ast.to_dict()


def ast_to_json(ast: FormulaAST) -> str:
    return json.dumps(ast.to_dict(), sort_keys=True, separators=(",", ":"))


def parse_factor_term(term: object) -> FormulaAST:
    """Parse a compact probability/Q term into an AST leaf.

    The parser is intentionally permissive and lossy.  Unrecognized terms stay
    as placeholders, so adding AST support cannot create false ID authority.
    """
    text = _s(term)
    if not text:
        return Placeholder("")
    q_match = re.match(r"^Q\[(?P<district>[^\]]+)\](?:\((?P<body>.*)\))?$", text)
    if q_match:
        return Q(_split_vars(q_match.group("district")), label=text)

    p_match = re.match(r"^P(?:_\{do\((?P<do>[^)]*)\)\})?\((?P<body>.*)\)$", text)
    if p_match:
        body = p_match.group("body") or ""
        do_vars = _split_vars(p_match.group("do"))
        if " | " in body:
            left, right = body.split(" | ", 1)
        elif "|" in body:
            left, right = body.split("|", 1)
        else:
            left, right = body, ""
        return P(_split_vars(left), given=_split_vars(right), interventions=do_vars, label=text)
    return Placeholder(text)


def _child_ast_from_subexpression(subexpr: Mapping[str, object]) -> FormulaAST:
    expr = subexpr.get("expression", {})
    if isinstance(expr, Mapping):
        ast_payload = expr.get("formula_ast")
        if isinstance(ast_payload, Mapping):
            return ast_from_dict(ast_payload)
        return payload_to_ast(expr)
    formula = _s(subexpr.get("formula"))
    if formula:
        return parse_factor_term(formula) if formula.startswith(("P(", "P_{", "Q[")) else Placeholder(formula)
    return Placeholder(_s(subexpr.get("status")) or "subexpression")


def ast_from_formula_parts(
    *,
    kind: str,
    y_set: Sequence[object],
    x_set: Sequence[object],
    formula: str = "",
    sum_over: Sequence[object] = (),
    product_terms: Sequence[object] = (),
    subexpressions: Sequence[Mapping[str, object]] = (),
    districts: Sequence[Sequence[object]] = (),
    formal_hedge: Optional[Mapping[str, object]] = None,
    reason_codes: str = "",
) -> FormulaAST:
    kind = _s(kind)
    metadata = {
        "kind": kind,
        "estimand": {"outcome": list(_dedupe(y_set)), "intervention": list(_dedupe(x_set))},
        "formula": _s(formula),
        "reason_codes": _s(reason_codes),
    }

    if formal_hedge:
        return HedgeFail(
            formal_hedge.get("F", []),
            formal_hedge.get("F_prime", []),
            roots=formal_hedge.get("roots_F", []) or formal_hedge.get("roots", []),
            label="formal_hedge_certificate",
        )

    factors: List[FormulaAST] = []
    for term in product_terms or []:
        factors.append(parse_factor_term(term))

    for subexpr in subexpressions or []:
        if isinstance(subexpr, Mapping):
            factors.append(_child_ast_from_subexpression(subexpr))

    if not factors:
        if kind in {"no_intervention", "graphical_zero_effect"}:
            factors.append(P(y_set))
        elif kind == "q_factor_full_district" and districts:
            factors.append(Q(districts[0], label="Q[" + ",".join(_dedupe(districts[0])) + "]"))
        elif _s(formula).startswith(("P(", "P_{", "Q[")):
            factors.append(parse_factor_term(formula))
        elif formula:
            factors.append(Placeholder(formula))
        else:
            factors.append(Placeholder(kind or "empty_formula"))

    expr = Product(factors, label=kind)
    if kind in {"q_factor_full_district", "q_input_subdistrict_recursion"} and districts:
        expr = Q(districts[-1], terms=(expr,), q_input=("Q[" + ",".join(_dedupe(districts[0])) + "]" if len(districts) > 1 else ""), label=kind)

    if sum_over:
        expr = Sum(sum_over, expr, label="sum_over")

    # Top-level do wrapper makes the estimand explicit without altering the
    # inner expression structure.
    if x_set:
        expr = Do(x_set, expr, label="estimand_do")

    return FormulaAST(
        expr.node_type,
        variables=expr.variables,
        conditioned_on=expr.conditioned_on,
        interventions=expr.interventions,
        bound_variables=expr.bound_variables,
        children=expr.children,
        label=expr.label,
        metadata={**dict(expr.metadata), **metadata},
    )


def _estimand_part(payload: Mapping[str, object], key: str) -> Sequence[object]:
    estimand = payload.get("estimand", {})
    if isinstance(estimand, Mapping):
        value = estimand.get(key, [])
        return value if isinstance(value, list) else [value]
    return []


def payload_to_ast(payload: Mapping[str, object]) -> FormulaAST:
    existing = payload.get("formula_ast")
    if isinstance(existing, Mapping):
        return ast_from_dict(existing)

    hedge = payload.get("formal_hedge_candidate")
    if isinstance(hedge, Mapping):
        return ast_from_formula_parts(
            kind="formal_hedge_certificate",
            y_set=_estimand_part(payload, "outcome"),
            x_set=_estimand_part(payload, "intervention"),
            formal_hedge=hedge,
            reason_codes=_s(payload.get("reason_codes")),
        )

    return ast_from_formula_parts(
        kind=_s(payload.get("type")),
        y_set=_estimand_part(payload, "outcome"),
        x_set=_estimand_part(payload, "intervention"),
        formula=_s(payload.get("formula")),
        sum_over=payload.get("sum_over", []) if isinstance(payload.get("sum_over", []), list) else [],
        product_terms=payload.get("product_terms", []) if isinstance(payload.get("product_terms", []), list) else [],
        subexpressions=payload.get("subexpressions", []) if isinstance(payload.get("subexpressions", []), list) else [],
        districts=payload.get("districts", []) if isinstance(payload.get("districts", []), list) else [],
        reason_codes=_s(payload.get("reason_codes")),
    )


def ast_latex(ast: FormulaAST) -> str:
    """Best-effort display serializer for debugging/tests."""
    if ast.node_type == "probability":
        left = ",".join(ast.variables)
        cond = ",".join(ast.conditioned_on)
        do = ",".join(ast.interventions)
        prefix = f"P_{{do({do})}}" if do else "P"
        return f"{prefix}({left} \\mid {cond})" if cond else f"{prefix}({left})"
    if ast.node_type == "q_factor":
        inner = " ".join(ast_latex(c) for c in ast.children)
        base = f"Q[{','.join(ast.variables)}]"
        return f"{base}({inner})" if inner else base
    if ast.node_type == "sum":
        inner = ast_latex(ast.children[0]) if ast.children else ""
        return f"\\sum_{{{','.join(ast.bound_variables)}}} {inner}"
    if ast.node_type == "product":
        return " \\cdot ".join(ast_latex(c) for c in ast.children)
    if ast.node_type == "fraction" and len(ast.children) == 2:
        return f"\\frac{{{ast_latex(ast.children[0])}}}{{{ast_latex(ast.children[1])}}}"
    if ast.node_type == "do":
        inner = ast_latex(ast.children[0]) if ast.children else ""
        return f"do({','.join(ast.interventions)})[{inner}]"
    if ast.node_type == "hedge_fail":
        return f"FAIL(F={{{','.join(ast.variables)}}},F'={{{','.join(ast.conditioned_on)}}})"
    return ast.label or ast.node_type


__all__ = [
    "ID_AST_VERSION",
    "FormulaAST",
    "P",
    "Q",
    "Sum",
    "Product",
    "Fraction",
    "Do",
    "HedgeFail",
    "Placeholder",
    "ast_from_dict",
    "ast_from_formula_parts",
    "payload_to_ast",
    "parse_factor_term",
    "ast_to_dict",
    "ast_to_json",
    "ast_latex",
]

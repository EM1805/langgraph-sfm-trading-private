from __future__ import annotations

"""Formula AST normalization helpers for the SCM ID layer.

Step 54 makes recursive-ID formula ASTs uniform before they are serialized.
The normalizer is intentionally conservative: it does not invent new ID
authority and it does not algebraically simplify probability expressions.  It
only canonicalizes the shape of existing ASTs so downstream estimation/audit
code can consume sums, products, fractions, do-wrappers, Q-factors and hedge
failures without relying on free-text formula parsing.
"""

from typing import Iterable, Mapping, Sequence

from .id_ast import FormulaAST, Placeholder, ast_from_dict


ID_AST_NORMALIZER_VERSION = "id_ast_normalizer_v1_step54"

_COMMUTATIVE_NODES = {"product"}
_WRAPPER_NODES = {"do", "sum"}
_ALLOWED_NODE_TYPES = {
    "probability",
    "q_factor",
    "sum",
    "product",
    "fraction",
    "do",
    "hedge_fail",
    "placeholder",
}


def _dedupe(values: Iterable[object]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        item = "" if value is None else str(value).strip()
        if item and item.lower() not in {"nan", "none", "null"} and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _metadata(ast: FormulaAST, *, extra: Mapping[str, object] | None = None) -> dict[str, object]:
    meta = dict(ast.metadata) if isinstance(ast.metadata, Mapping) else {}
    meta["formula_ast_normalized"] = 1
    meta["formula_ast_normalizer_version"] = ID_AST_NORMALIZER_VERSION
    if extra:
        meta.update(dict(extra))
    return meta


def _node_sort_key(ast: FormulaAST) -> tuple[str, str, str, str]:
    """Stable key used only when a node is explicitly commutative."""
    return (
        ast.node_type,
        "|".join(ast.variables),
        "|".join(ast.conditioned_on),
        ast.label,
    )


def _normalize_children(children: Sequence[FormulaAST]) -> tuple[FormulaAST, ...]:
    return tuple(normalize_formula_ast(child) for child in children or ())


def _flatten_product(children: Sequence[FormulaAST]) -> tuple[FormulaAST, ...]:
    flat: list[FormulaAST] = []
    for child in children:
        if child.node_type == "product":
            flat.extend(child.children)
        elif child.node_type == "placeholder" and (child.label or "").strip() == "1" and child.metadata.get("constant") == 1:
            continue
        else:
            flat.append(child)
    return tuple(sorted(flat, key=_node_sort_key))


def normalize_formula_ast(ast: FormulaAST | Mapping[str, object] | None) -> FormulaAST:
    """Return a normalized ``FormulaAST`` without changing ID semantics.

    Accepted input is either a ``FormulaAST`` instance, a dict returned by
    ``FormulaAST.to_dict()``, or ``None``.  Unknown node types are preserved as
    placeholders, preventing a display-only construct from becoming authority.
    """
    if ast is None:
        return Placeholder("empty_formula_ast", metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION})
    if isinstance(ast, Mapping):
        try:
            ast = ast_from_dict(ast)
        except Exception:  # pragma: no cover - defensive fallback
            return Placeholder("invalid_formula_ast_mapping", metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION})
    if not isinstance(ast, FormulaAST):
        return Placeholder(str(ast), metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION})

    node_type = ast.node_type if ast.node_type in _ALLOWED_NODE_TYPES else "placeholder"
    children = _normalize_children(ast.children)

    # Normalize operator-specific structure.  We deliberately do not collapse
    # single-child product/sum nodes here because downstream code benefits from
    # seeing the operator that the ID branch actually used.
    extra_meta: dict[str, object] = {}
    if node_type == "product":
        children = _flatten_product(children)
        extra_meta["product_flattened_step54"] = 1
        if not children:
            children = (Placeholder("1", metadata={"constant": 1, "formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION}),)
    elif node_type == "fraction":
        if len(children) < 2:
            children = tuple(children) + (Placeholder("missing_denominator", metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION}),)
        elif len(children) > 2:
            # Preserve all extra children inside the denominator product instead
            # of silently dropping structure.
            numerator = children[0]
            denominator = FormulaAST(
                "product",
                children=tuple(children[1:]),
                label="normalized_fraction_denominator_product",
                metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION},
            )
            children = (numerator, normalize_formula_ast(denominator))
        extra_meta["fraction_arity_checked_step54"] = 1
    elif node_type == "sum":
        if not children:
            children = (Placeholder("empty_sum_body", metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION}),)
        elif len(children) > 1:
            children = (FormulaAST(
                "product",
                children=children,
                label="normalized_sum_body_product",
                metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION},
            ),)
            children = _normalize_children(children)
        extra_meta["sum_arity_checked_step54"] = 1
    elif node_type == "do":
        if not children:
            children = (Placeholder("empty_do_body", metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION}),)
        elif len(children) > 1:
            children = (FormulaAST(
                "product",
                children=children,
                label="normalized_do_body_product",
                metadata={"formula_ast_normalized": 1, "formula_ast_normalizer_version": ID_AST_NORMALIZER_VERSION},
            ),)
            children = _normalize_children(children)
        extra_meta["do_arity_checked_step54"] = 1
    elif node_type == "q_factor":
        extra_meta["q_factor_children_normalized_step54"] = 1
    elif node_type == "hedge_fail":
        extra_meta["hedge_fail_normalized_step54"] = 1

    return FormulaAST(
        node_type,
        variables=_dedupe(ast.variables),
        conditioned_on=_dedupe(ast.conditioned_on),
        interventions=_dedupe(ast.interventions),
        bound_variables=_dedupe(ast.bound_variables),
        children=children,
        label=ast.label,
        metadata=_metadata(ast, extra=extra_meta),
    )


def normalized_ast_dict(ast: FormulaAST | Mapping[str, object] | None) -> dict[str, object]:
    return normalize_formula_ast(ast).to_dict()


__all__ = ["ID_AST_NORMALIZER_VERSION", "normalize_formula_ast", "normalized_ast_dict"]

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

try:
    from .action_registry_v2 import _minimal_yaml_load
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    _minimal_yaml_load = None

_CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
_RANK_TO_LABEL = {v: k for k, v in _CONF_RANK.items()}


class OperationalCausalGraph:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec or {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.children: Dict[str, List[str]] = defaultdict(list)
        self.parents: Dict[str, List[str]] = defaultdict(list)
        self.edge_conf: Dict[tuple[str, str], str] = {}
        self.alias_to_node: Dict[str, str] = {}
        self.path_hints: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in (self.spec.get("path_hints", {}) or {}).items()}
        self._build()

    def _norm(self, value: str) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _build(self) -> None:
        for node in self.spec.get("nodes", []) or []:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            self.nodes[node_id] = dict(node)
            self.alias_to_node[self._norm(node_id)] = node_id
            for alias in node.get("aliases", []) or []:
                alias = str(alias).strip()
                if alias:
                    self.alias_to_node[self._norm(alias)] = node_id
        for edge in self.spec.get("edges", []) or []:
            s = self.resolve_node_id(edge.get("source", "")) or str(edge.get("source", "")).strip()
            t = self.resolve_node_id(edge.get("target", "")) or str(edge.get("target", "")).strip()
            if not s or not t:
                continue
            self.children[s].append(t)
            self.parents[t].append(s)
            self.edge_conf[(s, t)] = str(edge.get("confidence", "unknown") or "unknown").lower()

    @classmethod
    def load(cls, path: str | Path = "operational_causal_graph.yaml") -> "OperationalCausalGraph":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if yaml is not None:
            spec = yaml.safe_load(text) or {}
        elif _minimal_yaml_load is not None:
            spec = _minimal_yaml_load(text) or {}
        else:
            raise RuntimeError("PyYAML is unavailable and the fallback YAML loader could not be imported.")
        return cls(spec)

    def resolve_node_id(self, raw: str | None) -> Optional[str]:
        if raw is None:
            return None
        val = str(raw).strip()
        if not val:
            return None
        if val in self.nodes:
            return val
        return self.alias_to_node.get(self._norm(val))

    def path_hint_for_harm(self, harm: str | None) -> Dict[str, Any]:
        harm_id = self.resolve_node_id(harm)
        return dict(self.path_hints.get(harm_id or "", {}))

    def harm_nodes(self) -> List[str]:
        return sorted([nid for nid, n in self.nodes.items() if str(n.get("type", "")) == "harm"])

    def best_path(self, source: str, target: str, max_depth: int = 5) -> Optional[Dict[str, Any]]:
        source = self.resolve_node_id(source) or source
        target = self.resolve_node_id(target) or target
        if source not in self.nodes or target not in self.nodes:
            return None
        best: Optional[Dict[str, Any]] = None
        q = deque([(source, [source], 3)])
        while q:
            cur, path, min_rank = q.popleft()
            if len(path) - 1 >= max_depth:
                continue
            for nxt in self.children.get(cur, []):
                if nxt in path:
                    continue
                edge_rank = _CONF_RANK.get(self.edge_conf.get((cur, nxt), "unknown"), 0)
                path_rank = min(min_rank, edge_rank)
                next_path = path + [nxt]
                if nxt == target:
                    cand = {"nodes": next_path, "confidence_rank": path_rank, "confidence": _RANK_TO_LABEL.get(path_rank, "unknown")}
                    if best is None or cand["confidence_rank"] > best["confidence_rank"] or (cand["confidence_rank"] == best["confidence_rank"] and len(cand["nodes"]) < len(best["nodes"])):
                        best = cand
                elif path_rank > 0:
                    q.append((nxt, next_path, path_rank))
        return best

    def reachable_harms(self, sources: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        harms = self.harm_nodes()
        for src_raw in sorted(set(sources or [])):
            src = self.resolve_node_id(src_raw) or src_raw
            if src not in self.nodes:
                continue
            for harm in harms:
                best = self.best_path(src, harm)
                if best is None:
                    continue
                hint = self.path_hint_for_harm(harm)
                out.append({
                    "source": src,
                    "harm": harm,
                    "path_nodes": best["nodes"],
                    "path_confidence": best["confidence"],
                    "path_length": len(best["nodes"]) - 1,
                    "path_hint": hint,
                    "contrast_key": hint.get("contrast_key", ""),
                    "treated_value": hint.get("treated_value"),
                    "control_value": hint.get("control_value"),
                    "preferred_stratum_keys": list(hint.get("preferred_stratum_keys", []) or []),
                    "adjust_for": list(hint.get("adjust_for", []) or []),
                    "avoid": list(hint.get("avoid", []) or []),
                })
        out.sort(key=lambda x: (-_CONF_RANK.get(x["path_confidence"], 0), x["path_length"], x["harm"]))
        return out

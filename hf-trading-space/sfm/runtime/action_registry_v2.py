from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Keep registry loading stdlib-only. Some mobile/embedded Python setups have
# broken optional YAML wheels that can hang during import; the bundled parser
# supports the registry subset used by Amantia.
yaml = None

DEFAULT_REGISTRY_PATH = "action_registry.yaml"


def _scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_scalar(part.strip()) for part in body.split(",")]
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    """Tiny YAML subset parser for Amantia registry files.

    It supports nested mappings and scalar lists, which is enough for
    action_registry.yaml and dangerous_paths.yaml. It is intentionally
    stdlib-only to keep the runtime safe on constrained/mobile environments.
    """
    rows: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append((len(raw) - len(raw.lstrip(" ")), stripped))

    def parse_scalar(value: str) -> Any:
        return _scalar(value)

    def parse_block(i: int, indent: int):
        if i >= len(rows):
            return {}, i

        # List block.
        if rows[i][0] == indent and rows[i][1].startswith("- "):
            out: List[Any] = []
            while i < len(rows) and rows[i][0] == indent and rows[i][1].startswith("- "):
                item_text = rows[i][1][2:].strip()
                i += 1
                if not item_text:
                    if i < len(rows) and rows[i][0] > indent:
                        item, i = parse_block(i, rows[i][0])
                    else:
                        item = None
                elif ":" in item_text and not item_text.startswith(('"', "'")):
                    key, val = item_text.split(":", 1)
                    item = {key.strip(): parse_scalar(val.strip()) if val.strip() else {}}
                    if i < len(rows) and rows[i][0] > indent:
                        child, i = parse_block(i, rows[i][0])
                        if val.strip():
                            if isinstance(child, dict):
                                item.update(child)
                        else:
                            item[key.strip()] = child
                else:
                    item = parse_scalar(item_text)
                out.append(item)
            return out, i

        # Mapping block.
        out: Dict[str, Any] = {}
        while i < len(rows):
            cur_indent, line = rows[i]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                # Malformed/unsupported indentation; skip it instead of looping.
                i += 1
                continue
            if line.startswith("- "):
                break
            if ":" not in line:
                i += 1
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            i += 1
            if val:
                out[key] = parse_scalar(val)
            elif i < len(rows) and (rows[i][0] > cur_indent or (rows[i][0] == cur_indent and rows[i][1].startswith("- "))):
                out[key], i = parse_block(i, rows[i][0])
            else:
                out[key] = {}
        return out, i

    parsed, _ = parse_block(0, rows[0][0] if rows else 0)
    return parsed if isinstance(parsed, dict) else {}


def load_action_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"actions": {}}
    text = p.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {"actions": {}}
    return _minimal_yaml_load(text) or {"actions": {}}


def get_action_spec(action_name: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    actions = registry.get("actions", {}) or {}
    return dict(actions.get(str(action_name).strip(), {}) or {})

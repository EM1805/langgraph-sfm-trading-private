from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

# Keep path-library loading stdlib-only for runtime safety.
yaml = None

DEFAULT_PATH_LIBRARY = "dangerous_paths.yaml"

def load_path_library(path: str | Path = DEFAULT_PATH_LIBRARY) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"paths": {}}
    text = p.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {"paths": {}}
    from .action_registry_v2 import _minimal_yaml_load
    return _minimal_yaml_load(text) or {"paths": {}}

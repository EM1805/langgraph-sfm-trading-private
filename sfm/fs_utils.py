from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]


def _safe_unlink(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _touch_probe(directory):
    try:
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".amantia_write_probe"
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok")
        _safe_unlink(probe)
        return True
    except OSError:
        return False


def ensure_writable_dir(path, fallback_prefix="amantia_out_"):
    """Return a directory that is guaranteed writable.

    The requested directory is created when possible. If it cannot be created or
    written to, a process-local fallback under the system temp directory is used.
    This keeps CLI commands from crashing in read-only ZIP extracts, CI
    sandboxes, containers, and serverless runtimes.
    """
    requested = Path(path).expanduser()
    if _touch_probe(requested):
        return requested

    stable_id = abs(hash(str(requested))) % 10000000
    fallback_root = Path(tempfile.gettempdir()) / (fallback_prefix + str(stable_id))
    fallback_root.mkdir(parents=True, exist_ok=True)
    if not _touch_probe(fallback_root):
        fallback_root = Path(tempfile.mkdtemp(prefix=fallback_prefix))
    print("[amantia:fs] Output directory is not writable: %s. Using fallback: %s" % (requested, fallback_root))
    return fallback_root


def ensure_writable_parent(file_path, fallback_dir=None):
    """Return a writable file path, preserving the filename when falling back."""
    requested = Path(file_path).expanduser()
    parent = requested.parent if str(requested.parent) else Path(".")
    if _touch_probe(parent):
        return requested

    base_dir = Path(fallback_dir) if fallback_dir is not None else ensure_writable_dir(Path(tempfile.gettempdir()) / "amantia_outputs")
    base_dir.mkdir(parents=True, exist_ok=True)
    fallback = base_dir / requested.name
    print("[amantia:fs] Output path is not writable: %s. Using fallback: %s" % (requested, fallback))
    return fallback


def write_text_safe(path, text, encoding="utf-8"):
    out = ensure_writable_parent(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding=encoding)
    return out

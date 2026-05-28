"""Runtime compatibility checks for Amantia.

Keep this module stdlib-only and import it before heavy scientific dependencies
such as pandas/numpy. Its job is to fail fast with a clear message when the
interpreter is outside the package's supported range.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import sys
from typing import Optional, Tuple

_MIN_PYTHON = (3, 10)
_MAX_PYTHON_EXCLUSIVE = (3, 14)
_RECOMMENDED = "Python 3.11 or 3.12; Python 3.13 is allowed only with fresh wheels"


class RuntimeCompatibilityError(RuntimeError):
    """Raised when Amantia is running on an unsupported runtime."""


def _version_tuple(version: str) -> Tuple[int, ...]:
    parts = []
    for raw in version.replace("-", ".").split("."):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts or [0])


def _installed_version(package: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def assert_supported_python() -> None:
    """Raise a clear error for unsupported Python versions.

    Amantia is validated on Python 3.10-3.12 and is allowed on Python 3.13 when
    the scientific stack is installed with compatible wheels. Python 3.14+ is
    intentionally rejected until pandas/numpy and the package tests are
    validated end-to-end.
    """
    if sys.version_info < _MIN_PYTHON or sys.version_info >= _MAX_PYTHON_EXCLUSIVE:
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise RuntimeCompatibilityError(
            "Unsupported Python runtime for Amantia: "
            f"Python {version}. Use Python >=3.10,<3.14. Python 3.7 is not supported. Recommended: "
            f"{_RECOMMENDED}."
        )


def assert_scientific_stack() -> None:
    """Validate pandas/numpy package versions without importing them.

    This intentionally uses package metadata instead of importing pandas/numpy,
    because broken or mismatched binary wheels can hang or fail during import.
    """
    assert_supported_python()

    numpy_version = _installed_version("numpy")
    pandas_version = _installed_version("pandas")
    missing = [name for name, ver in (("numpy", numpy_version), ("pandas", pandas_version)) if ver is None]
    if missing:
        raise RuntimeCompatibilityError(
            "Missing scientific dependency for Amantia: "
            + ", ".join(missing)
            + ". Run: python -m pip install -r requirements.txt"
        )

    numpy_tuple = _version_tuple(numpy_version or "0")
    pandas_tuple = _version_tuple(pandas_version or "0")

    if sys.version_info >= (3, 13):
        if not ((2, 1, 3) <= numpy_tuple < (2, 4)) or not ((2, 2, 3) <= pandas_tuple < (2, 4)):
            raise RuntimeCompatibilityError(
                "Python 3.13 requires compatible pandas/numpy wheels for Amantia. "
                f"Detected numpy=={numpy_version}, pandas=={pandas_version}. "
                "Run: python -m pip install --upgrade --force-reinstall "
                "'numpy>=2.1.3,<2.4' 'pandas>=2.2.3,<2.4'"
            )
    else:
        if not ((1, 26) <= numpy_tuple < (2, 4)) or not ((2, 2, 3) <= pandas_tuple < (2, 4)):
            raise RuntimeCompatibilityError(
                "Incompatible pandas/numpy versions for Amantia. "
                f"Detected numpy=={numpy_version}, pandas=={pandas_version}. "
                "Run: python -m pip install --upgrade --force-reinstall -r requirements.txt"
            )


__all__ = ["RuntimeCompatibilityError", "assert_supported_python", "assert_scientific_stack"]

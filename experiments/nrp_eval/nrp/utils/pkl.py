"""Versioned pickle save/load.

Pickles are wrapped in a small dict carrying a schema name and version number
so we can detect (and warn about) mismatches on load.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def save_versioned(obj: Any, path: str | Path, schema: str = "evaluation_result") -> None:
    """Save ``obj`` to ``path`` wrapped in a versioned envelope.

    Args:
        obj: The Python object to pickle.
        path: Destination path. Parent directories are created if needed.
        schema: A free-form schema tag (e.g. ``"evaluation_result"``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "version": SCHEMA_VERSION,
        "schema": schema,
        "data": obj,
    }
    with open(path, "wb") as f:
        pickle.dump(wrapper, f)


def load_versioned(path: str | Path, expected_schema: str | None = None) -> Any:
    """Load a versioned pickle from ``path``.

    Args:
        path: Path to the pickle.
        expected_schema: If provided, raise :class:`ValueError` when the
            stored schema does not match.

    Returns:
        The unwrapped object.

    Raises:
        ValueError: If ``expected_schema`` is set and does not match.
    """
    path = Path(path)
    with open(path, "rb") as f:
        wrapper = pickle.load(f)
    if wrapper.get("version") != SCHEMA_VERSION:
        warnings.warn(
            f"Pickle version mismatch (file={wrapper.get('version')}, "
            f"code={SCHEMA_VERSION}). Results may be incompatible.",
            stacklevel=2,
        )
    if expected_schema and wrapper.get("schema") != expected_schema:
        raise ValueError(
            f"Pickle schema mismatch (file='{wrapper.get('schema')}', "
            f"expected='{expected_schema}')"
        )
    return wrapper["data"]

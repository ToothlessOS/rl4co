"""Shared fixtures for learn_decompose_eval tests."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_nrp_eval_on_path():
    """Make sure the nrp_eval sibling experiment is importable."""
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    nrp_eval = here.parent.parent / "nrp_eval"
    if nrp_eval.exists() and str(nrp_eval) not in sys.path:
        sys.path.insert(0, str(nrp_eval))


@pytest.fixture(scope="session")
def lkh_binary() -> str:
    """Path to the patched LKH-3 binary."""
    from pathlib import Path

    here = Path(__file__).resolve().parent
    cand = here.parent / "LKH-3.0.14" / "LKH"
    if cand.exists():
        return str(cand)
    return os.environ.get("LDE_LKH_BINARY") or os.environ.get("NRP_LKH_BINARY") or ""

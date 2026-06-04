"""W&B init helper.

A small wrapper that respects the ``WANDB_DISABLED`` environment variable and
degrades gracefully when ``wandb`` is not installed.
"""
from __future__ import annotations

import os
import warnings
from typing import Any


def init_wandb(
    project: str = "rl4co",
    name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    mode: str | None = None,
) -> Any:
    """Initialise a W&B run, returning the run object (or ``None``).

    Returns ``None`` (and emits a single warning) if ``wandb`` is not installed
    or the ``WANDB_DISABLED`` env var is set to a truthy value.
    """
    try:
        import wandb
    except ImportError:
        warnings.warn("wandb not installed; skipping W&B init.", stacklevel=2)
        return None

    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return None
    if mode is not None:
        os.environ.setdefault("WANDB_MODE", mode)

    return wandb.init(
        project=project,
        name=name,
        group=group,
        tags=tags or [],
        config=config,
    )

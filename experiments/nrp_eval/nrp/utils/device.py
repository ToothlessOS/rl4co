"""Device helpers."""
from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device specifier to a concrete :class:`torch.device`.

    ``None`` or ``"auto"`` resolves to CUDA when available, else CPU.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_device(obj, device: str | torch.device):
    """Move a tensor, TensorDict, or module to ``device``.

    Falls back to returning the object unchanged if it does not expose a
    ``.to`` method.
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj

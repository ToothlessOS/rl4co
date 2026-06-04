"""Warmup-aware timing context manager."""
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager


@contextmanager
def timer(sync_cuda: bool = True):
    """Context manager that yields elapsed wall-clock seconds.

    If sync_cuda is True and CUDA is available, calls torch.cuda.synchronize()
    before and after the block.
    """
    if sync_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
    if sync_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def measure(
    fn: Callable, *args, warmup: int = 1, sync_cuda: bool = True, **kwargs
) -> tuple:
    """Run fn(*args, **kwargs) warmup+1 times; return (last_result, elapsed_s).

    The first `warmup` calls are excluded from timing.
    """
    last_result = None
    for _ in range(warmup):
        last_result = fn(*args, **kwargs)
    with timer(sync_cuda=sync_cuda) as get_elapsed:
        last_result = fn(*args, **kwargs)
    return last_result, get_elapsed()

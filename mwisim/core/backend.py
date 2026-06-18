"""Compute-backend hook (CPU / GPU). Phase 0: CPU/NumPy only — but the indirection is
here so the kernels never hard-code ``numpy`` and a CuPy/PyTorch GPU backend can be
dropped in later without touching call sites (PROJECT_PLAN §11.4).

Usage pattern (later): ``xp = get_array_module(x)`` then use ``xp.fft.fft2`` etc., so the
same code runs on numpy arrays (CPU) or cupy arrays (GPU).
"""
from __future__ import annotations

import numpy as np

#: The active array module. Phase 0 = numpy. Later: switchable to cupy.
xp = np


def get_array_module(*arrays):
    """Return the array module for the given arrays.

    Phase 0 always returns numpy. A future GPU backend will return ``cupy`` when the
    inputs are cupy arrays (CuPy ships ``cupy.get_array_module`` that does exactly this).
    """
    return np

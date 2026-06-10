"""2D free-space scalar Green's function.

Tutorial ref: F1 §2.  Convention e^{+jωt}: G = (1/4j) H0^(2)(k_b R).
"""
from __future__ import annotations

import numpy as np
from scipy.special import hankel2


def green_2d(k_b: complex, R):
    """2D Green's function G(R) = (1/4j) * H0^(2)(k_b * R).

    Parameters
    ----------
    k_b : complex
        Background wavenumber (complex if lossy).
    R : array_like
        Distance(s) |r - r'| > 0. (Self term R=0 is handled separately in mom.py.)

    Returns
    -------
    G : complex ndarray, same shape as R.

    TODO (F1 §2): use scipy.special.hankel2(0, k_b*R). Guard R>0.
    """
    
    G = 1 / 4j * hankel2(0, k_b * R)
    return G.astype(complex)
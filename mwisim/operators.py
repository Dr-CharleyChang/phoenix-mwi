"""Matrix-free forward/adjoint operators (F2 stage — Born operator A).

Not needed for F1.  These power the inversion stage (CGLS/LSQR, DBIM) and the
CG-FFT acceleration; kept here so F1 code can later be wrapped without rewrite.

Tutorial refs: notes 9.6.3.0 (A_op / AH_op), chapter 7 (CG-FFT).
"""
from __future__ import annotations

import numpy as np


def A_op(v: np.ndarray, E_inc: np.ndarray, G_tr, k_b: complex, dS: float) -> np.ndarray:
    """Forward Born operator  A v  (N -> M): contrast -> scattered field.

    A v = k_b^2 * dS * G_tr @ (E_inc * v)   ("equivalent source radiates to rx").
    G_tr may later be an FFT-based callable instead of a dense matrix.

    TODO (F2): implement once F1 works; reuse mom.scattered_field logic.
    """
    return k_b**2 * dS * G_tr @ (E_inc * v)

def AH_op(u: np.ndarray, E_inc: np.ndarray, G_tr, k_b: complex, dS: float) -> np.ndarray:
    """Adjoint operator  A^H u  (M -> N): residual back-propagated to voxels.

    A^H u = k_b^2 * dS * conj(E_inc) * (G_tr^H @ u).

    TODO (F2 / inversion).
    """
    return k_b**2 * dS * np.conj(E_inc) * (G_tr.conj().T @ u)

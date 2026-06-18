"""MoM2D — the 2D Richmond MoM forward solver behind the :class:`ForwardSolver` interface.

Phase 0 refactor: this is a thin *adapter*. It changes no physics — it reuses the
validated F1 functions (`mom`) and the F2 matrix-free engine (`operators.GreenFFT`), and
just re-expresses them as the platform's forward-solver contract. The 15 existing tests
still guard the underlying code; the Phase-0 tests check the adapter is faithful.

Two interchangeable backends, chosen at construction:
  - ``method="dense"``  -> build the N×N matrix and direct-solve (F1).
  - ``method="cgfft"``  -> matrix-free CG-FFT iterative solve (F2).
"""
from __future__ import annotations

import numpy as np

from ..core.interfaces import ForwardSolver, Phantom
from ..core.registry import register
# Reuse the validated building blocks (aliased to avoid name clashes with our methods):
from ..mom import (
    build_D,
    incident_plane_wave,
    solve_total_field as _dense_solve,
    scattered_field as _radiate,
)
from ..operators import GreenFFT


@register("forward", "mom2d")
class MoM2D(ForwardSolver):
    """2D TM Method-of-Moments forward solver (plane-wave incidence along +x).

    Knobs (per-implementation, live in __init__):
        method : "cgfft" (default, matrix-free) or "dense" (direct).
        tol, maxiter, krylov : CG-FFT controls (krylov = "bicgstab" or "gmres").
    """

    def __init__(self, method: str = "cgfft", tol: float = 1e-8,
                 maxiter: int | None = None, krylov: str = "bicgstab"):
        if method not in ("cgfft", "dense"):
            raise ValueError(f"method must be 'cgfft' or 'dense', got {method!r}")
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.krylov = krylov

    # -- ForwardSolver interface ------------------------------------------
    def solve_total_field(self, phantom: Phantom, freq: float, **kwargs):
        centers, dS = phantom.grid()
        d = float(np.sqrt(dS))
        k_b = phantom.background_wavenumber(freq)
        chi = phantom.contrast(freq)
        E_inc = incident_plane_wave(centers, k_b)        # plane wave E0 e^{-j k_b x}

        if self.method == "dense":
            D = build_D(centers, chi, k_b, d)            # N×N coupling matrix (F1)
            E_tot = _dense_solve(D, E_inc)               # np.linalg.solve(I-D, E_inc)
            info = {"method": "dense", "N": int(centers.shape[0])}
        else:  # "cgfft"
            op = GreenFFT(centers, chi, k_b, d)          # matrix-free operator (F2)
            E_tot, info = op.solve_total_field(
                E_inc, tol=self.tol, maxiter=self.maxiter, method=self.krylov
            )
        return E_tot, info

    def scattered_field(self, phantom: Phantom, E_tot, rx, freq: float, **kwargs):
        centers, dS = phantom.grid()
        k_b = phantom.background_wavenumber(freq)
        chi = phantom.contrast(freq)
        return _radiate(rx, centers, chi, E_tot, k_b, dS)   # F1 §6 radiation sum

"""Matrix-free forward operators — F2 stage (CG-FFT acceleration).

YOUR TASK (F2): implement the TODO-marked methods of ``GreenFFT`` so that
``tests/test_f2.py`` (T9-T14) goes green.  Tutorial:
``docs/F2_Tutorial_CG-FFT-matrix-free-solver.md``.

The idea: on a regular grid the Richmond MoM matrix is block-Toeplitz with
Toeplitz blocks (BTTB) — entries depend only on the displacement r_m - r_n.
A BTTB matvec is a 2D convolution, evaluated in O(N log N) with an FFT and a
circulant embedding (zero-padding to kill wrap-around).  The dense
(I - D) E = E_inc solve becomes a matrix-free iterative solve.

Kernel to reproduce — exactly ``mom.build_D`` (including the F1 self-cell fix!):

    off-diagonal  g(rho) = pref * J1(k_b a) * H0^(2)(k_b rho)      (rho > 0)
    self-cell     g(0)   = pref * H1^(2)(k_b a) - 1
    with          pref   = -(j*pi*k_b*a/2),  a = d/sqrt(pi)

and D_mn = g(r_m - r_n) * chi_n.   Convention e^{+jwt} / H^(2), as in mom.py.

Tutorial refs: F2 tutorial §1-§3; A_op/AH_op note 9.6.3.0.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, hankel2
from scipy.fft import fft2, ifft2, next_fast_len
import scipy.sparse.linalg as spla


def infer_grid_shape(centers: np.ndarray) -> tuple[int, int]:
    """Recover (Ny, Nx) from a meshgrid-raveled ``centers`` array.  (GIVEN)

    ``grid.make_grid`` builds ``centers`` row-major (C order) so that flat index
    ``n = iy * Nx + ix``.  Read this carefully — it encodes the ravel convention
    your kernel axes must match (axis 0 = y!).
    """
    nx = np.unique(np.round(centers[:, 0], 12)).size
    ny = np.unique(np.round(centers[:, 1], 12)).size
    if nx * ny != centers.shape[0]:
        raise ValueError(
            f"centers ({centers.shape[0]}) is not a full {ny}x{nx} grid — "
            "the FFT operator needs a regular grid."
        )
    return ny, nx


class GreenFFT:
    """Matrix-free Richmond MoM operator via circulant-embedded FFT.

    Plan: precompute the displacement kernel g and its FFT once in __init__;
    every matvec is then two FFTs of the padded field.  Never form NxN.
    """

    def __init__(self, centers: np.ndarray, chi: np.ndarray, k_b: complex, d: float):
        self.ny, self.nx = infer_grid_shape(centers)
        self.N = self.ny * self.nx
        self.k_b = complex(k_b)
        self.d = float(d)
        self.chi = np.asarray(chi, dtype=complex).reshape(self.ny, self.nx)

        # TODO (F2 §3, step 1): build the displacement kernel g.
        #   1. a = d/sqrt(pi); pref = -(1j*pi*k_b*a/2).
        #   2. displacement ranges: dyr = np.arange(-(ny-1), ny), same for dxr;
        #      meshgrid with indexing="ij"; rho = d*sqrt(DX^2 + DY^2).
        #   3. off-diagonal g via J1/H0^(2); evaluate inside
        #      np.errstate(invalid="ignore") — hankel2(0, 0) is NaN.
        #   4. self term g(0) = pref*H1^(2)(k_b a) - 1   (the F1 "-1"!).
        #      Combine with np.where(rho > 0, ...).
        #
        # TODO (F2 §3, step 2): circulant embedding.
        #   5. self.py / self.px = next_fast_len(2*n - 1) per axis.
        #   6. g_pad = zeros((py, px)); place displacement (dy, dx) at index
        #      (dy % py, dx % px) — np.mod + np.ix_ scatter the block.
        #   7. self.G_hat = fft2(g_pad).   Precompute ONCE here.
        a = self.d / np.sqrt(np.pi)
        pref = -(1j * np.pi * self.k_b * a / 2)

        dyr = np.arange(-(self.ny - 1), self.ny)
        dxr = np.arange(-(self.nx - 1), self.nx)
        DY, DX = np.meshgrid(dyr, dxr, indexing = "ij")
        rho = self.d * np.sqrt(DX**2 + DY**2)

        with np.errstate(invalid="ignore"):
            g_off = pref * jv(1, self.k_b * a) * hankel2(0, self.k_b * rho)
        g_self = pref * hankel2(1, self.k_b * a) - 1
        g = np.where(rho > 0, g_off, g_self)
        
        self.py = next_fast_len(2 * self.ny - 1)
        self.px = next_fast_len(2 * self.nx - 1)
        g_pad = np.zeros((self.py, self.px), dtype=complex)
        g_pad[np.mod(DY, self.py), np.mod(DX, self.px)] = g
        self.G_hat = fft2(g_pad)
        
    # -- core matvecs ------------------------------------------------------
    def _conv(self, v_grid: np.ndarray) -> np.ndarray:
        """2D *linear* convolution g * v via the padded FFT.

        TODO (F2 §3, step 3): zero-pad v_grid into the top-left (ny, nx) corner
        of a (py, px) array; out = ifft2(self.G_hat * fft2(vp)); return the
        [:ny, :nx] slice.  The wrap-around garbage lives in what you discard.
        """
        vp = np.zeros((self.py, self.px), dtype=complex)
        vp[:self.ny, :self.nx] = v_grid
        return ifft2(self.G_hat * fft2(vp))[:self.ny, :self.nx]

    def apply_D(self, x: np.ndarray) -> np.ndarray:
        """Matrix-free ``D @ x`` (flat in, flat out).

        TODO (F2 §3, step 4): reshape x to (ny, nx), multiply by chi FIRST
        (chi_n indexes the *source* cell — column index), convolve, ravel.
        Target: T9 — match dense build_D @ x on a RANDOM vector to <1e-12.
        """
        x_grid = x.reshape(self.ny, self.nx)
        x_grid= self.chi * x_grid
        return self._conv(x_grid).ravel()

    def apply_IminusD(self, x: np.ndarray) -> np.ndarray:
        """Matrix-free ``(I - D) @ x`` — the forward-solve operator.

        TODO: one line.
        """
        return x - self.apply_D(x)

    def as_linear_operator(self) -> spla.LinearOperator:
        """SciPy ``LinearOperator`` for ``(I - D)``.  (GIVEN — boilerplate.)"""
        return spla.LinearOperator(
            shape=(self.N, self.N), matvec=self.apply_IminusD, dtype=complex
        )

    # -- forward solve -----------------------------------------------------
    def solve_total_field(
        self,
        E_inc: np.ndarray,
        tol: float = 1e-8,
        maxiter: int | None = None,
        method: str = "bicgstab",
    ) -> tuple[np.ndarray, dict]:
        """Solve ``(I - D) E = E_inc`` matrix-free.  Returns (E_tot, info).

        TODO (F2 §3, step 5):
          1. A = self.as_linear_operator(); pick scipy.sparse.linalg.bicgstab
             (method="bicgstab") or gmres (method="gmres").
          2. SciPy >=1.12 renamed `tol` -> `rtol`: call with rtol=..., and on
             TypeError fall back to tol=... .
          3. gmres: pass callback_type="pr_norm" explicitly (else a
             DeprecationWarning, and "legacy" counts outer iters only).
          4. Count iterations with a callback; compute the final relative
             residual ||b - A E|| / ||b|| yourself — don't trust `status` alone.
          5. info dict: iters, status, rel_residual, method, N.
        Targets: T11/T12 — match the F1 direct solve to <1e-7 at tol=1e-10.
        """
        A = self.as_linear_operator()
    
        iters = 0
        def _cb(_):
            nonlocal iters
            iters += 1
        
        if method == "bicgstab":
            solver, extra = spla.bicgstab, {}
        elif method == "gmres":
            solver, extra = spla.gmres, {"callback_type": "pr_norm"}
        else:
            raise ValueError(f"Unknown method: {method}")
        
        try:
            E_tot, status = solver(A, E_inc, rtol=tol, maxiter=maxiter, callback=_cb, **extra)
        except TypeError:
            E_tot, status = solver(A, E_inc, tol=tol, maxiter=maxiter, callback=_cb, **extra)   
        
        rel_residual = np.linalg.norm(E_inc - A @ E_tot) / np.linalg.norm(E_inc)

        info = {
            "iters": iters,
            "status": status,
            "rel_residual": rel_residual,
            "method": method,
            "N": self.N,
        }
        return E_tot, info


# --------------------------------------------------------------------------
# Born operators (inversion stage; already implemented pre-F2). Dense form is
# fine for small problems; DBIM will later reuse GreenFFT for the in-domain
# Green action.
# --------------------------------------------------------------------------
def A_op(v: np.ndarray, E_inc: np.ndarray, G_tr, k_b: complex, dS: float) -> np.ndarray:
    """Forward Born operator ``A v`` (contrast -> scattered field at rx).

    ``A v = k_b^2 * dS * G_tr @ (E_inc * v)``.  ``G_tr`` is the rx-by-grid Green
    matrix (small M, dense is fine).
    """
    return k_b**2 * dS * G_tr @ (E_inc * v)


def AH_op(u: np.ndarray, E_inc: np.ndarray, G_tr, k_b: complex, dS: float) -> np.ndarray:
    """Adjoint operator ``A^H u`` (residual back-propagated to voxels)."""
    return k_b**2 * dS * np.conj(E_inc) * (G_tr.conj().T @ u)

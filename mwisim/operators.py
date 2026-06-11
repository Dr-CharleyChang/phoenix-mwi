"""Matrix-free forward operators — F2 stage (CG-FFT acceleration).

On a regular grid the Richmond MoM coupling matrix is **block-Toeplitz with
Toeplitz blocks (BTTB)**: its entries depend only on the *displacement*
``r_m - r_n``, not on the absolute positions.  A BTTB matrix-vector product is a
2D convolution, evaluated in ``O(N log N)`` with an FFT and a circulant
embedding (zero-padding to kill wrap-around).  This turns the dense
``(I - D) E = E_inc`` solve (``O(N^3)`` factorisation, ``O(N^2)`` storage) into a
matrix-free iterative solve (``O(N log N)`` per iteration, ``O(N)`` storage),
which is what makes large / 3D MWI problems tractable.

Convention matches ``mom.py``: ``e^{+jwt}`` / ``H^(2)``, equal-area cell radius
``a = d/sqrt(pi)``.  The kernel reproduced here is exactly ``mom.build_D``:

    off-diagonal  g(rho) = pref * J1(k_b a) * H0^(2)(k_b rho)      (rho > 0)
    self-cell     g(0)   = pref * H1^(2)(k_b a) - 1
    with          pref   = -(j*pi*k_b*a/2)

and ``D_mn = g(r_m - r_n) * chi_n``.

Tutorial refs: F2 (CG-FFT), notes chapter 7; A_op/AH_op note 9.6.3.0.
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, hankel2
from scipy.fft import fft2, ifft2, next_fast_len
import scipy.sparse.linalg as spla


def infer_grid_shape(centers: np.ndarray) -> tuple[int, int]:
    """Recover (Ny, Nx) from a meshgrid-raveled ``centers`` array.

    ``grid.make_grid`` builds ``centers`` row-major (C order) so that flat index
    ``n = iy * Nx + ix``.  We count unique coordinates to recover the shape.
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

    Parameters
    ----------
    centers : (N, 2) float
        Cell centers from ``grid.make_grid`` (regular grid, row-major).
    chi : (N,) complex
        Contrast per cell.
    k_b : complex
        Background wavenumber.
    d : float
        Cell side length.

    Notes
    -----
    The first-column kernel ``g`` of the BTTB matrix is precomputed and FFT'd
    once; every matvec is then two FFTs of the padded field.  ``apply_D`` and
    ``apply_IminusD`` are matrix-free and never form an ``N x N`` array.
    """

    def __init__(self, centers: np.ndarray, chi: np.ndarray, k_b: complex, d: float):
        self.ny, self.nx = infer_grid_shape(centers)
        self.N = self.ny * self.nx
        self.k_b = complex(k_b)
        self.d = float(d)
        self.chi = np.asarray(chi, dtype=complex).reshape(self.ny, self.nx)

        a = d / np.sqrt(np.pi)
        pref = -(1j * np.pi * k_b * a / 2)

        # displacement-index grid: dy in [-(Ny-1), Ny-1], dx in [-(Nx-1), Nx-1]
        dyr = np.arange(-(self.ny - 1), self.ny)
        dxr = np.arange(-(self.nx - 1), self.nx)
        DY, DX = np.meshgrid(dyr, dxr, indexing="ij")
        rho = d * np.sqrt(DX.astype(float) ** 2 + DY.astype(float) ** 2)

        with np.errstate(invalid="ignore"):
            g_off = pref * jv(1, k_b * a) * hankel2(0, k_b * rho)
        g_self = pref * hankel2(1, k_b * a) - 1.0
        g = np.where(rho > 0, g_off, g_self).astype(complex)

        # circulant embedding: pad to >= (2N-1) in each axis, next fast FFT len
        self.py = next_fast_len(2 * self.ny - 1)
        self.px = next_fast_len(2 * self.nx - 1)

        g_pad = np.zeros((self.py, self.px), dtype=complex)
        # place displacement (dy, dx) at index (dy % py, dx % px): negatives wrap
        iy = np.mod(dyr, self.py)
        ix = np.mod(dxr, self.px)
        g_pad[np.ix_(iy, ix)] = g
        self.G_hat = fft2(g_pad)

    # -- core matvecs ------------------------------------------------------
    def _conv(self, v_grid: np.ndarray) -> np.ndarray:
        """2D linear convolution g * v via circulant-embedded FFT."""
        vp = np.zeros((self.py, self.px), dtype=complex)
        vp[: self.ny, : self.nx] = v_grid
        out = ifft2(self.G_hat * fft2(vp))
        return out[: self.ny, : self.nx]

    def apply_D(self, x: np.ndarray) -> np.ndarray:
        """Matrix-free ``D @ x`` (flat in, flat out). ``D_mn = g(r_m-r_n) chi_n``."""
        v = (self.chi * x.reshape(self.ny, self.nx))
        return self._conv(v).ravel()

    def apply_IminusD(self, x: np.ndarray) -> np.ndarray:
        """Matrix-free ``(I - D) @ x`` — the forward-solve operator."""
        return x - self.apply_D(x)

    def as_linear_operator(self) -> spla.LinearOperator:
        """SciPy ``LinearOperator`` for ``(I - D)`` (for bicgstab/gmres)."""
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
        """Solve ``(I - D) E = E_inc`` matrix-free.

        Returns ``(E_tot, info)`` carrying iteration count and the final relative
        residual.  ``method`` is ``"bicgstab"`` (classic CG-FFT pairing) or
        ``"gmres"`` (more robust for strong scattering).
        """
        A = self.as_linear_operator()
        b = np.asarray(E_inc, dtype=complex)

        iters = {"n": 0}

        def _cb(xk):
            iters["n"] += 1

        solver = {"bicgstab": spla.bicgstab, "gmres": spla.gmres}[method]
        kw = dict(maxiter=maxiter, callback=_cb)
        if method == "gmres":
            kw["callback_type"] = "pr_norm"  # count inner iters, silence warning
        # SciPy >=1.12 renamed `tol`->`rtol`; support both.
        try:
            E, status = solver(A, b, rtol=tol, atol=0.0, **kw)
        except TypeError:
            E, status = solver(A, b, tol=tol, atol=0.0, **kw)

        res = np.linalg.norm(b - A.matvec(E)) / np.linalg.norm(b)
        info = {"iters": iters["n"], "status": int(status),
                "rel_residual": float(res), "method": method, "N": self.N}
        return E, info


# --------------------------------------------------------------------------
# Born operators (inversion stage). Dense form retained for small problems;
# the in-domain Green action can reuse GreenFFT in DBIM later.
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

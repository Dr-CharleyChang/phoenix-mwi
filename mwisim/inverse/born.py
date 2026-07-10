"""Born linear inversion for the first contrast-map reconstruction stage.

Convention e^{+jωt}/H^(2), single frequency.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ..core.interfaces import Inverter
from ..core.registry import register
from ..green import green_2d
from ..operators import A_op, AH_op

C0 = 299_792_458.0


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def green_matrix(rx: np.ndarray, centers: np.ndarray, k_b: complex) -> np.ndarray:
    """Receiver×cell Green matrix ``G_tr`` with shape ``(M, N)``.

    ``G_tr[m, n] = (1/4j) H0^(2)(k_b * |rx_m - centers_n|)`` — reuse ``green_2d``.
    Receivers are OUTSIDE the object, so every distance > 0 (no self-cell term).
    
    """
    rho = np.sqrt(((rx[:, None, :] - centers[None, :, :])**2).sum(axis=-1))
    G_tr = green_2d(k_b, rho)
    return G_tr


def plane_wave_incidences(centers: np.ndarray, k_b: complex, angles) -> np.ndarray:
    """Incident plane-wave field on the grid for each direction. Shape ``(N_v, N)``.

    For direction θ, unit vector k̂=(cosθ, sinθ):  E_inc(r) = exp(-j k_b (k̂·r)).
    (θ = 0 reproduces the plane wave e^{-j k_b x}.)
    """
    angles = np.atleast_1d(np.asarray(angles, dtype=float))     # (N_v,)
    k_hat = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (N_v, 2) unit directions
    phase = k_hat @ centers.T          # (N_v,2) @ (2,N) -> (N_v, N):  k̂_i · r_n
    E_inc = np.exp(-1j * k_b * phase)  # (N_v, N)
    return E_inc


# --------------------------------------------------------------------------
# The stacked multiview Born operator
# --------------------------------------------------------------------------
class BornOperator:
    """Matrix-free multiview Born operator A and its adjoint Aᴴ.

    A maps a contrast vector χ ``(N,)`` to stacked scattered data ``(N_v·M,)`` over all
    incidences; Aᴴ maps a residual ``(N_v·M,)`` back to the grid ``(N,)``. Built once from
    the geometry; never formed densely.
    """

    def __init__(self, centers, rx, E_inc_set, k_b, dS):
        self.centers = np.asarray(centers)
        self.rx = np.asarray(rx)
        self.E_inc_set = np.atleast_2d(np.asarray(E_inc_set, dtype=complex))  # (N_v, N)
        self.k_b = complex(k_b)
        self.dS = float(dS)
        self.G_tr = green_matrix(self.rx, self.centers, self.k_b)  # (M, N), built once
        self.N = self.centers.shape[0]
        self.M = self.rx.shape[0]
        self.Nv = self.E_inc_set.shape[0]

    def matvec(self, chi: np.ndarray) -> np.ndarray:
        """A χ -> stacked data ``(N_v·M,)``.
        """
        d = np.zeros(self.Nv * self.M, dtype=complex)
        for i in range(self.Nv):
            d[i * self.M:(i + 1) * self.M] = A_op(chi, self.E_inc_set[i], self.G_tr, self.k_b, self.dS)
        return d

    def rmatvec(self, u: np.ndarray) -> np.ndarray:
        """Aᴴ u -> grid ``(N,)``.
        """
        chi = np.zeros(self.N, dtype=complex)
        for i in range(self.Nv):
            chi += AH_op(u[i * self.M:(i + 1) * self.M], self.E_inc_set[i], self.G_tr, self.k_b, self.dS)
        return chi

    def as_linear_operator(self) -> spla.LinearOperator:
        """SciPy ``LinearOperator`` wrapping ``matvec`` and ``rmatvec``."""
        return spla.LinearOperator(
            shape=(self.Nv * self.M, self.N),
            matvec=self.matvec, rmatvec=self.rmatvec, dtype=complex,
        )


# --------------------------------------------------------------------------
# The inverter
# --------------------------------------------------------------------------
@register("inverter", "born")
class BornInverter(Inverter):
    """Born linear inversion: regularized least squares via LSMR/LSQR.

    Knobs (per-implementation, in __init__):
        mu       : Tikhonov weight (damp = sqrt(mu)).
        iter_lim : max least-squares iterations.
        solver   : "lsmr" (default) or "lsqr".
    """

    def __init__(self, mu: float = 1e-2, iter_lim: int = 200, solver: str = "lsmr"):
        self.mu = float(mu)
        self.iter_lim = int(iter_lim)
        self.solver = solver

    def reconstruct(self, data: dict, forward=None, x0=None, **kwargs):
        """Reconstruct χ̂ from ``data``. Returns ``(chi_hat, info)``.

        ``data`` carries the geometry + measurements (see ``make_born_problem``):
        keys ``centers, dS, k_b, rx, E_inc_set, d`` (and ``chi_true`` for tests).
        ``forward`` is unused for linear Born — it's the hook DBIM/I2 will use.
        """
        op = BornOperator(data["centers"], data["rx"], data["E_inc_set"],
                          data["k_b"], data["dS"])
        A = op.as_linear_operator()
        d = data["d"]                       # (N_v·M,)
        damp = np.sqrt(self.mu)             # Tikhonov: min ||A x - d||^2 + mu||x||^2

        # lsmr/lsqr are FUNCTIONS (self.solver is a string); both return a TUPLE whose
        # first three entries are (x, istop, itn). They differ in the iteration kwarg.
        if self.solver == "lsmr":
            out = spla.lsmr(A, d, damp=damp, maxiter=self.iter_lim)
        elif self.solver == "lsqr":
            out = spla.lsqr(A, d, damp=damp, iter_lim=self.iter_lim)
        else:
            raise ValueError(f"solver must be 'lsmr' or 'lsqr', got {self.solver!r}")

        chi_hat, istop, itn = out[0], out[1], out[2]
        info = {"iters": int(itn), "istop": int(istop),
                "mu": self.mu, "solver": self.solver, "N": op.N}
        return chi_hat, info


# --------------------------------------------------------------------------
# Synthetic-problem assembler
# --------------------------------------------------------------------------
def make_born_problem(eps_r=1.1, n_per_lambda=12, n_views=16, n_rx=40, f=1e9,
                      R_obs_factor=3.0, mode="physical") -> dict:
    """Assemble an I1 test problem as a ``data`` dict.

    mode="crime"    -> data = A χ_true via the Born operator (no model error; unit test).
    mode="physical" -> data from the FULL forward solve per incidence (real Born error).

    Returns dict: centers, dS, k_b, rx, E_inc_set, d, chi_true, f.
    """
    from ..phantoms.circle import CirclePhantom
    from ..mom import build_D, solve_total_field, scattered_field

    lam0 = C0 / f
    R_cyl = 0.3 * lam0
    lam1 = lam0 / np.sqrt(np.real(eps_r))
    d = lam1 / n_per_lambda

    ph = CirclePhantom(R_cyl=R_cyl, eps_r=eps_r, d=d)
    centers, dS = ph.grid()
    k_b = ph.background_wavenumber(f)
    chi_true = ph.contrast()

    angles = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    E_inc_set = plane_wave_incidences(centers, k_b, angles)            # (N_v, N)

    a = np.linspace(0, 2 * np.pi, n_rx, endpoint=False)
    rx = np.column_stack([R_obs_factor * R_cyl * np.cos(a),
                          R_obs_factor * R_cyl * np.sin(a)])

    if mode == "crime":
        op = BornOperator(centers, rx, E_inc_set, k_b, dS)
        d_data = op.matvec(chi_true)
    elif mode == "physical":
        D = build_D(centers, chi_true, k_b, d)                        # dense, once
        blocks = []
        for E_inc_i in E_inc_set:
            E_tot_i = solve_total_field(D, E_inc_i)                   # (I - D) E = E_inc_i
            blocks.append(scattered_field(rx, centers, chi_true, E_tot_i, k_b, dS))
        d_data = np.concatenate(blocks)
    else:
        raise ValueError(f"mode must be 'crime' or 'physical', got {mode!r}")

    return dict(centers=centers, dS=dS, k_b=k_b, rx=rx, E_inc_set=E_inc_set,
                d=d_data, chi_true=chi_true, f=f)

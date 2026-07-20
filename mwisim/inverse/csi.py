"""CSI (Contrast Source Inversion) for nonlinear contrast reconstruction.

Convention e^{+jwt}/H^(2), single frequency.
"""
from __future__ import annotations

import numpy as np

from ..core.interfaces import Inverter
from ..core.registry import register
from ..mom import build_D
from .born import BornInverter, green_matrix, make_born_problem

C0 = 299_792_458.0


def receiver_operator(rx: np.ndarray, centers: np.ndarray, k_b: complex, dS: float) -> np.ndarray:
    """Return the homogeneous receiver operator S with shape (M, N).

    S = k_b**2 * dS * G_tr, where G_tr[m, n] propagates a source at grid/target
    cell n to receiver m.  Here dS is the cell area/integration weight.
    """
    return (k_b**2 * dS) * green_matrix(rx, centers, k_b)


def domain_green_matrix(centers: np.ndarray, k_b: complex, d: float) -> np.ndarray:
    """Return the dense grid-to-grid Richmond interaction kernel G with shape (N, N).

    This is ``build_D`` with chi == 1, so weighting its columns by a real contrast
    gives the usual MoM domain operator D(chi) = G @ diag(chi).  It includes the
    equal-area-cell/self-cell discretization, not only the raw continuous Green function.
    """
    ones = np.ones(centers.shape[0], dtype=complex)
    return build_D(centers, ones, k_b, d)


def simulate_csi_data(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Stack receiver data S w_i over all incidences into shape (N_v*M,)."""
    return np.concatenate([S @ w_i for w_i in W], axis=0)


def update_contrast_sources(
    chi: np.ndarray,
    E_inc_set: np.ndarray,
    d_data: np.ndarray,
    S: np.ndarray,
    G_dom: np.ndarray,
    mu_w: float,
    xi: float,
) -> np.ndarray:
    """Solve the CSI w-subproblem for all incidences.

    For fixed chi, each view solves the regularized least-squares problem

        min_w ||S w - d_i||^2 + xi ||(I - XG) w - X E_inc_i||^2 + mu_w ||w||^2

    where X = diag(chi).
    """
    Nv, N = E_inc_set.shape
    M = S.shape[0]
    I = np.eye(N, dtype=complex)
    XG = chi[:, None] * G_dom
    alpha = np.sqrt(float(xi))
    beta = np.sqrt(float(mu_w))
    B = np.vstack([S, alpha * (I - XG), beta * I])

    # B is identical for every view, so solve all N_v right-hand sides at once.  This is
    # mathematically the same least-squares problem as a Python loop, but NumPy reuses the
    # matrix factorization and the Phase-1 benchmark is substantially faster.
    data_blocks = np.asarray(d_data, dtype=complex).reshape(Nv, M)
    rhs = np.vstack([
        data_blocks.T,
        alpha * (chi[None, :] * E_inc_set).T,
        np.zeros((N, Nv), dtype=complex),
    ])
    return np.linalg.lstsq(B, rhs, rcond=None)[0].T


def update_contrast(
    W: np.ndarray,
    E_inc_set: np.ndarray,
    G_dom: np.ndarray,
    mu_chi: float,
    project_real: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the CSI chi-subproblem in closed form, cell by cell."""
    Nv, N = W.shape
    E_tot_set = np.zeros((Nv, N), dtype=complex)
    numer = np.zeros(N, dtype=complex)
    denom = np.full(N, float(mu_chi), dtype=float)

    for i in range(Nv):
        E_tot_i = E_inc_set[i] + G_dom @ W[i]
        E_tot_set[i] = E_tot_i
        numer += np.conj(E_tot_i) * W[i]
        denom += np.abs(E_tot_i) ** 2

    chi = numer / denom
    if project_real:
        chi = np.maximum(chi.real, 0.0).astype(complex)
    return chi, E_tot_set


@register("inverter", "csi")
class CSIInverter(Inverter):
    """Contrast Source Inversion with alternating updates of w and chi."""

    def __init__(
        self,
        mu_chi: float = 1e-2,
        mu_w: float = 1e-3,
        xi: float = 1.0,
        max_outer: int = 20,
        tol: float = 1e-3,
        step: float = 1.0,
        init: str = "born",
        init_iter_lim: int = 300,
        project_real: bool = True,
    ):
        self.mu_chi = float(mu_chi)
        self.mu_w = float(mu_w)
        self.xi = float(xi)
        self.max_outer = int(max_outer)
        self.tol = float(tol)
        self.step = float(step)
        self.init = init
        self.init_iter_lim = int(init_iter_lim)
        self.project_real = bool(project_real)

    def reconstruct(self, data: dict, forward=None, x0=None, **kwargs):
        """Reconstruct chi_hat from data. Returns (chi_hat, info)."""
        centers = np.asarray(data["centers"])
        rx = np.asarray(data["rx"])
        E_inc_set = np.atleast_2d(np.asarray(data["E_inc_set"], dtype=complex))
        d_data = np.asarray(data["d"], dtype=complex)
        k_b = complex(data["k_b"])
        dS = float(data["dS"])
        d = float(np.sqrt(dS))

        S = receiver_operator(rx, centers, k_b, dS)
        G_dom = domain_green_matrix(centers, k_b, d)

        if x0 is not None:
            chi = np.asarray(x0, dtype=complex).copy()
        elif self.init == "born":
            chi, _ = BornInverter(mu=self.mu_chi, iter_lim=self.init_iter_lim).reconstruct(data)
        elif self.init == "zero":
            chi = np.zeros(centers.shape[0], dtype=complex)
        else:
            raise ValueError(f"init must be 'born' or 'zero', got {self.init!r}")

        W = chi[None, :] * E_inc_set
        data_res_history = []
        state_res_history = []

        for n in range(self.max_outer):
            W = update_contrast_sources(chi, E_inc_set, d_data, S, G_dom, self.mu_w, self.xi)
            chi_new, E_tot_set = update_contrast(
                W, E_inc_set, G_dom, self.mu_chi, project_real=self.project_real
            )
            chi = (1.0 - self.step) * chi + self.step * chi_new

            d_sim = simulate_csi_data(S, W)
            data_res = float(np.linalg.norm(d_data - d_sim) / np.linalg.norm(d_data))
            state_gap = W - chi[None, :] * E_tot_set
            state_res = float(np.linalg.norm(state_gap) / max(np.linalg.norm(W), 1e-12))
            data_res_history.append(data_res)
            state_res_history.append(state_res)

            if data_res < self.tol:
                break

        info = {
            "outer_iters": n + 1,
            # CSI-specific source-data residual ||d - S W|| / ||d||, not the
            # full nonlinear forward residual ||d - F(chi)|| / ||d||.
            "data_res_history": data_res_history,
            "state_res_history": state_res_history,
            "mu_chi": self.mu_chi,
            "mu_w": self.mu_w,
            "xi": self.xi,
            "step": self.step,
            "tol": self.tol,
            "init": self.init,
            "N": centers.shape[0],
        }
        return chi, info


def make_csi_problem(
    eps_r=1.5,
    n_per_lambda=12,
    n_views=16,
    n_rx=40,
    f=1e9,
    R_obs_factor=3.0,
) -> dict:
    """Assemble a CSI problem using full-forward physical data."""
    return make_born_problem(
        eps_r=eps_r,
        n_per_lambda=n_per_lambda,
        n_views=n_views,
        n_rx=n_rx,
        f=f,
        R_obs_factor=R_obs_factor,
        mode="physical",
    )

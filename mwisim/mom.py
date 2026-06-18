"""Method-of-Moments (Richmond) forward solver for 2D TM scattering.

Tutorial refs: F1 §3 (D matrix + self-cell), §4 (incident field),
§5 (solve total field), §6 (scattered field at receivers).

Convention e^{+jωt} / H^(2). Equal-area circular cell radius a = d/sqrt(pi).
"""
from __future__ import annotations

import numpy as np
from scipy.special import jv, hankel2

from .green import green_2d

def build_D(centers: np.ndarray, chi: np.ndarray, k_b: complex, d: float) -> np.ndarray:
    """Richmond coupling matrix D (N x N), so the system is (I - D) E = E_inc.

    Off-diagonal (m != n):
        D_mn = -chi_n * (j*pi*k_b*a/2) * J1(k_b a) * H0^(2)(k_b * rho_mn)
    Self  (m == n):
        D_nn = -chi_n * (j*pi*k_b*a/2) * H1^(2)(k_b a)
    with a = d/sqrt(pi), rho_mn = |r_m - r_n|.

    TODO (F1 §3.4):
      1. pairwise distances rho_mn (use broadcasting on centers).
      2. fill off-diagonal with J1*H0^(2) form.
      3. overwrite diagonal with H1^(2) self term.
      4. multiply each column n by chi_n and the prefactor.
    scipy: jv(1,.), hankel2(0,.), hankel2(1,.).
    """
    """
    a = d / np.sqrt(np.pi)
    N = centers.shape[0]
    D = np.zeros((N, N), dtype=complex)
    for n in range(N):
        for m in range(N):
            rho_mn = np.linalg.norm(centers[m] - centers[n])
            if n != m:
                D[m, n] = -chi[n] * (1j * np.pi * k_b * a / 2) * jv(1, k_b * a) * hankel2(0, k_b * rho_mn)
            else:
                D[m, n] = -chi[n] * (1j * np.pi * k_b * a / 2) * hankel2(1, k_b * a)
    return D
    """
    a = d / np.sqrt(np.pi)
    pref = -(1j * np.pi * k_b * a / 2)
    diff = centers[:, None, :] - centers[None, :, :]
    rho = np.sqrt((diff**2).sum(axis=-1))
    D = pref * jv(1, k_b * a) * hankel2(0, k_b * rho)
    np.fill_diagonal(D, pref * hankel2(1, k_b * a) - 1)   # -1 → -chi_n after column-multiply (self-cell lower-limit term)
    D = D * chi[None, :]
    return D

def incident_plane_wave(centers: np.ndarray, k_b: complex, E0: complex = 1.0) -> np.ndarray:
    """Plane wave E_inc(r) = E0 * exp(-j k_b x), evaluated at cell centers.

    Returns (N,) complex.  TODO (F1 §4): use centers[:, 0] as x.
    """
    E_inc = E0 * np.exp(-1j * k_b * centers[:, 0])
    return E_inc.astype(complex)


def solve_total_field(D: np.ndarray, E_inc: np.ndarray) -> np.ndarray:
    """Solve (I - D) E_tot = E_inc for the in-domain total field.

    Returns (N,) complex.  F1: direct solve is fine (np.linalg.solve).
    TODO (F1 §5).
    """
    I = np.eye(D.shape[0])
    return np.linalg.solve(I - D, E_inc)


def scattered_field(
    rx_points: np.ndarray,
    centers: np.ndarray,
    chi: np.ndarray,
    E_tot: np.ndarray,
    k_b: complex,
    dS: float,
) -> np.ndarray:
    """Scattered field at exterior observation points (the G_tr action).

    E_sc(r_r) = k_b^2 * sum_n G(r_r, r_n) * chi_n * E_tot_n * dS,
    with G = (1/4j) H0^(2)(k_b |r_r - r_n|).  rx_points are OUTSIDE the cylinder
    so all distances > 0 (no singularity).

    Parameters
    ----------
    rx_points : (Nrx, 2) float
    Returns
    -------
    E_sc : (Nrx,) complex

    TODO (F1 §6): distances r_r -> r_n, Green, weighted sum.
    """
    pref = k_b**2 * dS
    rho = np.sqrt(((rx_points[:, None, :] - centers[None, :, :])**2).sum(axis=-1))
    G = green_2d(k_b, rho)   # (1/4j) H0^(2)(k_b rho); rx outside cylinder => rho>0, no self-cell
    E_sc = pref * (G @ (chi * E_tot))
    return E_sc
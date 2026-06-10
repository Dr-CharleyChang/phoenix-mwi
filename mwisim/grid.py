"""Grid generation and contrast assignment.

Tutorial refs: F1 §1 (physics setup), §3.1 (discretization).
"""
from __future__ import annotations

import numpy as np
import math

def make_grid(domain_size: float, d: float) -> tuple[np.ndarray, float]:
    """Uniform square grid of cell centers covering [-L/2, L/2]^2.

    Parameters
    ----------
    domain_size : float
        Side length L of the square imaging domain (meters). Must comfortably
        contain the cylinder (e.g. >= 2.5 * diameter).
    d : float
        Cell side length (meters). Pick d <= lambda_1 / 15 where
        lambda_1 = lambda_0 / sqrt(eps_r) is the *in-medium* wavelength.

    Returns
    -------
    centers : (N, 2) float ndarray
        (x, y) coordinates of each cell center.
    dS : float
        Cell area = d**2.

    TODO (F1 §3.1): build a regular grid of cell centers. Origin at cylinder
    center. Tip: np.meshgrid + reshape to (N, 2).
    """
    N_cells = math.ceil(domain_size / d)
    if N_cells % 2 == 0:
        x_ = np.linspace(-(N_cells // 2) * d + d / 2, (N_cells // 2) * d - d / 2, N_cells)
        y_ = x_
    else:
        x_ = np.linspace(-(N_cells // 2) * d, (N_cells // 2) * d, N_cells)
        y_ = x_
    X, Y = np.meshgrid(x_, y_, indexing="xy")
    centers = np.column_stack([X.ravel(), Y.ravel()])  # centers: (Ny*Nx, 2)
    dS = d**2
    return centers, dS


def assign_contrast(
    centers: np.ndarray, R_cyl: float, eps_r: complex, eps_b: complex = 1.0
) -> np.ndarray:
    """Contrast function chi_n for each cell.

    Inside the cylinder (|r_n| <= R_cyl): chi = eps_r/eps_b - 1.
    Outside: chi = 0.

    Returns
    -------
    chi : (N,) complex ndarray

    TODO (F1 §1): radial test on centers, fill complex array.
    """
    r = np.hypot(centers[:, 0], centers[:, 1])       # (N,) distance of each cell to the origin
    chi = np.where(r <= R_cyl, eps_r / eps_b - 1, 0.0)
    return chi.astype(complex)
"""Validation metrics and convergence study."""
from __future__ import annotations

import numpy as np


def rel_l2_error(approx: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error ||approx - ref|| / ||ref|| (complex-aware)."""
    approx = np.asarray(approx)
    ref = np.asarray(ref)
    denom = np.linalg.norm(ref)
    if denom == 0:
        raise ValueError("reference has zero norm")
    return float(np.linalg.norm(approx - ref) / denom)


def convergence_study(d_list, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run the full MoM-vs-Mie pipeline for several cell sizes d.

    For each d: build grid, contrast, D, solve E_tot, scattered field at the
    receiver ring; compare to Mie; record rel_l2_error.

    Returns
    -------
    n_per_lambda : (K,) float    in-medium cells-per-wavelength for each d
    errs : (K,) float
        Relative L2 error for each d.

    ``params`` holds ``f``, ``eps_r``, ``eps_b``, ``R_cyl``, ``R_obs``,
    ``N_rx``, ``domain_size``, and related setup values.
    """
    from .grid import make_grid, assign_contrast
    from .mom import build_D, incident_plane_wave, solve_total_field, scattered_field
    from .mie import mie_scattered
    f = params["f"]
    eps_r = params["eps_r"]
    eps_b = params["eps_b"]
    R_cyl = params["R_cyl"]
    R_obs = params["R_obs"]
    N_rx = params["N_rx"]
    domain_size = params["domain_size"]

    C0 = 299_792_458
    lam0 = C0 / f
    k_b = 2 * np.pi * f / C0 * np.sqrt(eps_b)
    k_1 = 2 * np.pi * f / C0 * np.sqrt(eps_r)   # FIX: f/C0 = 1/lam0 (was f/lam0 = f^2/C0)
    lam1 = lam0 / np.sqrt(eps_r)

    ang = np.linspace(0, 2 * np.pi, N_rx, endpoint=False)
    centers_obs = np.column_stack((R_obs * np.cos(ang), R_obs * np.sin(ang)))
    E_mie = mie_scattered(centers_obs, k_b, k_1, R_cyl)

    errs = []
    n_per_lambda = []
    for d in d_list:
        centers, dS = make_grid(domain_size, d)
        chi = assign_contrast(centers, R_cyl, eps_r, eps_b)
        m = chi != 0
        centers, chi = centers[m], chi[m]        
        D = build_D(centers, chi, k_b, d)
        E_inc = incident_plane_wave(centers, k_b)
        E_tot= solve_total_field(D, E_inc)
        E_sc = scattered_field(centers_obs, centers, chi, E_tot, k_b, dS)
        errs.append(rel_l2_error(E_sc, E_mie))
        n_per_lambda.append(lam1 / d)
    return np.array(n_per_lambda), np.array(errs)

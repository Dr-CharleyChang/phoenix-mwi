"""F2 self-test checklist (T9-T14): CG-FFT matrix-free acceleration.

The contract for F2 is simple and strict: the fast path must reproduce the slow
path. Every test compares the matrix-free FFT operator / iterative solve against
the dense F1 ground truth (or, for the operator itself, against ``build_D``).

Unlike F1's ``_setup`` (which masks to cylinder cells), F2 uses the **full
regular grid** — the BTTB/FFT structure requires it. The chi=0 exterior cells
cost nothing and are carried naturally.

Run:  pytest tests/test_f2.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.grid import make_grid, assign_contrast
from mwisim.mom import build_D, incident_plane_wave, solve_total_field, scattered_field
from mwisim.mie import mie_scattered
from mwisim.metrics import rel_l2_error
from mwisim.operators import GreenFFT, infer_grid_shape

C0 = 299_792_458.0


def _full_setup(eps_r=2.0, n_per_lambda=12, f=1e9, domain_factor=2.0, eps_b=1.0):
    """Full-grid setup (no masking) so the grid stays regular for the FFT op."""
    lam0 = C0 / f
    k_b = 2 * np.pi / lam0 * np.sqrt(eps_b)
    k_1 = 2 * np.pi / lam0 * np.sqrt(eps_r)
    R_cyl = 0.3 * lam0
    lam1 = lam0 / np.sqrt(eps_r)
    d = lam1 / n_per_lambda
    centers, dS = make_grid(domain_factor * 2 * R_cyl, d)
    chi = assign_contrast(centers, R_cyl, eps_r, eps_b)
    return dict(centers=centers, chi=chi, dS=dS, d=np.sqrt(dS),
                k_b=k_b, k_1=k_1, R_cyl=R_cyl, lam0=lam0)


def _ring(R_obs, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([R_obs * np.cos(a), R_obs * np.sin(a)])


# ---------- T9: matrix-free D matches dense build_D ----------
@pytest.mark.parametrize("eps_r", [2.0, 8.0])
def test_T9_apply_D_matches_dense(eps_r):
    s = _full_setup(eps_r=eps_r)
    D = build_D(s["centers"], s["chi"], s["k_b"], s["d"])
    op = GreenFFT(s["centers"], s["chi"], s["k_b"], s["d"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal(D.shape[1]) + 1j * rng.standard_normal(D.shape[1])
    assert rel_l2_error(op.apply_D(x), D @ x) < 1e-12


# ---------- T10: (I - D) matvec matches dense ----------
def test_T10_IminusD_matvec():
    s = _full_setup(eps_r=8.0)
    D = build_D(s["centers"], s["chi"], s["k_b"], s["d"])
    op = GreenFFT(s["centers"], s["chi"], s["k_b"], s["d"])
    rng = np.random.default_rng(1)
    x = rng.standard_normal(D.shape[1]) + 1j * rng.standard_normal(D.shape[1])
    ref = x - D @ x
    assert rel_l2_error(op.apply_IminusD(x), ref) < 1e-12


# ---------- T11: BiCGStab CG-FFT solve matches direct solve ----------
@pytest.mark.parametrize("eps_r", [2.0, 8.0])
def test_T11_bicgstab_matches_direct(eps_r):
    s = _full_setup(eps_r=eps_r)
    D = build_D(s["centers"], s["chi"], s["k_b"], s["d"])
    E_inc = incident_plane_wave(s["centers"], s["k_b"])
    E_dir = solve_total_field(D, E_inc)
    op = GreenFFT(s["centers"], s["chi"], s["k_b"], s["d"])
    E_fft, info = op.solve_total_field(E_inc, tol=1e-10, method="bicgstab")
    assert info["rel_residual"] < 1e-8
    assert rel_l2_error(E_fft, E_dir) < 1e-7


# ---------- T12: GMRES solve also matches direct ----------
def test_T12_gmres_matches_direct():
    s = _full_setup(eps_r=8.0)
    D = build_D(s["centers"], s["chi"], s["k_b"], s["d"])
    E_inc = incident_plane_wave(s["centers"], s["k_b"])
    E_dir = solve_total_field(D, E_inc)
    op = GreenFFT(s["centers"], s["chi"], s["k_b"], s["d"])
    E_fft, info = op.solve_total_field(E_inc, tol=1e-10, method="gmres")
    assert rel_l2_error(E_fft, E_dir) < 1e-7


# ---------- T13: end-to-end scattered field via fast path == slow path == Mie ----------
def test_T13_end_to_end_scattered_field():
    s = _full_setup(eps_r=2.0, n_per_lambda=20)
    D = build_D(s["centers"], s["chi"], s["k_b"], s["d"])
    E_inc = incident_plane_wave(s["centers"], s["k_b"])
    E_dir = solve_total_field(D, E_inc)
    op = GreenFFT(s["centers"], s["chi"], s["k_b"], s["d"])
    E_fft, _ = op.solve_total_field(E_inc, tol=1e-10)
    rx = _ring(3 * s["R_cyl"])
    E_slow = scattered_field(rx, s["centers"], s["chi"], E_dir, s["k_b"], s["dS"])
    E_fast = scattered_field(rx, s["centers"], s["chi"], E_fft, s["k_b"], s["dS"])
    E_mie = mie_scattered(rx, s["k_b"], s["k_1"], s["R_cyl"])
    # fast path reproduces slow path to solver tolerance ...
    assert rel_l2_error(E_fast, E_slow) < 1e-7
    # ... and both still validate against the analytic Mie series
    assert rel_l2_error(E_fast, E_mie) < 0.05


# ---------- T14: grid-shape inference ----------
def test_T14_infer_grid_shape():
    centers, _ = make_grid(1.0, 0.1)
    ny, nx = infer_grid_shape(centers)
    assert ny * nx == centers.shape[0]
    with pytest.raises(ValueError):
        infer_grid_shape(centers[:-1])  # drop a cell -> no longer a full grid

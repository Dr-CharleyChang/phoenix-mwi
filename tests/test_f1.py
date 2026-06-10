"""F1 self-test checklist (T1-T8) as executable tests.

These are RED until you implement the stubs in mwisim/. Make them green one by
one — that is your F1 progress bar. Run:  pytest -q

Maps to docs/F1 tutorial §11 (自测清单).
"""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.grid import make_grid, assign_contrast
from mwisim.green import green_2d
from mwisim.mom import build_D, incident_plane_wave, solve_total_field, scattered_field
from mwisim.mie import mie_scattered
from mwisim.metrics import rel_l2_error

C0 = 299_792_458.0


# ---------- small fixtures ----------
def _setup(eps_r=2.0, n_per_lambda=15, f=1e9, eps_b=1.0):
    lam0 = C0 / f
    k_b = 2 * np.pi / lam0 * np.sqrt(eps_b)
    k_1 = 2 * np.pi / lam0 * np.sqrt(eps_r)
    R_cyl = 0.5 * lam0
    lam1 = lam0 / np.sqrt(eps_r)
    d = lam1 / n_per_lambda
    centers, dS = make_grid(2.5 * 2 * R_cyl, d)
    chi = assign_contrast(centers, R_cyl, eps_r, eps_b)
    m = chi != 0
    return dict(centers=centers[m], chi=chi[m], dS=dS, k_b=k_b, k_1=k_1,
                R_cyl=R_cyl, lam0=lam0)


def _ring(R_obs, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([R_obs * np.cos(a), R_obs * np.sin(a)])


# ---------- metrics helper (already implemented) ----------
def test_rel_l2_error_helper():
    x = np.array([1 + 1j, 2 - 1j])
    assert rel_l2_error(x, x) == pytest.approx(0.0)


# ---------- T2: incident field unit magnitude ----------
def test_T2_incident_unit_magnitude():
    s = _setup()
    E_inc = incident_plane_wave(s["centers"], s["k_b"])
    assert np.allclose(np.abs(E_inc), 1.0)


# ---------- T3: Green depends only on distance (kernel symmetry) ----------
def test_T3_green_symmetric_in_distance():
    k_b = 2 * np.pi
    assert green_2d(k_b, 0.3) == pytest.approx(green_2d(k_b, 0.3))
    # two equal distances -> equal values (sanity on vectorization)
    R = np.array([0.2, 0.2, 0.5])
    g = green_2d(k_b, R)
    assert g[0] == pytest.approx(g[1])


# ---------- T4: weak-scatterer sanity (E_tot ~ E_inc) ----------
def test_T4_weak_scatterer_total_field():
    s = _setup(eps_r=1.01)
    D = build_D(s["centers"], s["chi"], s["k_b"], np.sqrt(s["dS"]))
    E_inc = incident_plane_wave(s["centers"], s["k_b"])
    E_tot = solve_total_field(D, E_inc)
    assert rel_l2_error(E_tot, E_inc) < 0.05


# ---------- T5: Mie self-convergence ----------
def test_T5_mie_self_convergence():
    s = _setup()
    rx = _ring(3 * s["R_cyl"])
    e_small = mie_scattered(rx, s["k_b"], s["k_1"], s["R_cyl"], Nmax=8)
    e_large = mie_scattered(rx, s["k_b"], s["k_1"], s["R_cyl"], Nmax=25)
    assert rel_l2_error(e_small, e_large) < 1e-3


# ---------- T6/T7: MoM matches Mie (weak scatterer) ----------
def test_T6_mom_matches_mie_weak():
    s = _setup(eps_r=2.0, n_per_lambda=20)
    D = build_D(s["centers"], s["chi"], s["k_b"], np.sqrt(s["dS"]))
    E_inc = incident_plane_wave(s["centers"], s["k_b"])
    E_tot = solve_total_field(D, E_inc)
    rx = _ring(3 * s["R_cyl"])
    E_mom = scattered_field(rx, s["centers"], s["chi"], E_tot, s["k_b"], s["dS"])
    E_mie = mie_scattered(rx, s["k_b"], s["k_1"], s["R_cyl"])
    assert rel_l2_error(E_mom, E_mie) < 0.05


# ---------- T8: still matches for strong scatterer ----------
def test_T8_mom_matches_mie_strong():
    s = _setup(eps_r=8.0, n_per_lambda=25)
    D = build_D(s["centers"], s["chi"], s["k_b"], np.sqrt(s["dS"]))
    E_inc = incident_plane_wave(s["centers"], s["k_b"])
    E_tot = solve_total_field(D, E_inc)
    rx = _ring(3 * s["R_cyl"])
    E_mom = scattered_field(rx, s["centers"], s["chi"], E_tot, s["k_b"], s["dS"])
    E_mie = mie_scattered(rx, s["k_b"], s["k_1"], s["R_cyl"])
    assert rel_l2_error(E_mom, E_mie) < 0.05

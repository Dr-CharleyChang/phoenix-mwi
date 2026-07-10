"""I1 self-test checklist (I1.1–I1.5): Born linear inversion.

Run:  python -m pytest tests/test_i1.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.metrics import rel_l2_error
from mwisim.phantoms.circle import CirclePhantom
from mwisim.core.registry import build, available
from mwisim.inverse.born import (
    green_matrix, plane_wave_incidences, BornOperator, BornInverter, make_born_problem,
)

C0 = 299_792_458.0


def _small_geometry(eps_r=2.0, n_per_lambda=10, f=1e9, n_views=4, n_rx=24):
    """A small problem's geometry (centers, rx, E_inc_set, k_b, dS) for operator tests."""
    lam0 = C0 / f
    R_cyl = 0.3 * lam0
    lam1 = lam0 / np.sqrt(eps_r)
    d = lam1 / n_per_lambda
    ph = CirclePhantom(R_cyl=R_cyl, eps_r=eps_r, d=d)
    centers, dS = ph.grid()
    k_b = ph.background_wavenumber(f)
    angles = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    E_inc_set = plane_wave_incidences(centers, k_b, angles)
    a = np.linspace(0, 2 * np.pi, n_rx, endpoint=False)
    rx = np.column_stack([3 * R_cyl * np.cos(a), 3 * R_cyl * np.sin(a)])
    return dict(centers=centers, dS=dS, k_b=k_b, rx=rx, E_inc_set=E_inc_set, R_cyl=R_cyl)


# ---------- I1.1: green_matrix sanity ----------
def test_I1_1_green_matrix():
    g = _small_geometry()
    G = green_matrix(g["rx"], g["centers"], g["k_b"])
    assert G.shape == (g["rx"].shape[0], g["centers"].shape[0])
    # depends only on distance: two receivers equidistant from a cell give equal entries
    c0 = g["centers"][0]
    rx2 = np.array([c0 + [0.5, 0.0], c0 + [0.0, 0.5], c0 + [-0.5, 0.0]])
    G2 = green_matrix(rx2, c0[None, :], g["k_b"])
    assert G2[0, 0] == pytest.approx(G2[1, 0])
    assert G2[0, 0] == pytest.approx(G2[2, 0])


# ---------- I1.2: adjoint test (the gate) ----------
def test_I1_2_adjoint():
    g = _small_geometry()
    op = BornOperator(g["centers"], g["rx"], g["E_inc_set"], g["k_b"], g["dS"])
    rng = np.random.default_rng(0)
    chi = rng.standard_normal(op.N) + 1j * rng.standard_normal(op.N)
    u = rng.standard_normal(op.Nv * op.M) + 1j * rng.standard_normal(op.Nv * op.M)
    lhs = np.vdot(op.matvec(chi), u)     # <A χ, u>
    rhs = np.vdot(chi, op.rmatvec(u))    # <χ, Aᴴ u>
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


# ---------- I1.3: inverse crime — the reconstruction must FIT its own data ----------
def test_I1_3_inverse_crime_fits_data():
    # Born's A is rank-deficient (single-frequency ring => limited resolution), so even
    # its OWN data does not pin down chi_true uniquely. What an inverse crime guarantees
    # is that the reconstruction explains the data: ||A chi_hat - d|| / ||d|| -> 0.
    prob = make_born_problem(eps_r=1.1, n_per_lambda=10, n_views=16, n_rx=40, mode="crime")
    op = BornOperator(prob["centers"], prob["rx"], prob["E_inc_set"], prob["k_b"], prob["dS"])
    inv = BornInverter(mu=1e-12, iter_lim=2000)
    chi_hat, info = inv.reconstruct(prob)
    data_residual = np.linalg.norm(op.matvec(chi_hat) - prob["d"]) / np.linalg.norm(prob["d"])
    assert data_residual < 1e-4


# ---------- I1.4: physical recovery (real Born model error) ----------
def test_I1_4_physical_recovery():
    prob = make_born_problem(eps_r=1.1, n_per_lambda=12, n_views=16, n_rx=40, mode="physical")
    inv = BornInverter(mu=1e-2, iter_lim=400)
    chi_hat, info = inv.reconstruct(prob)
    # localization: reconstructed |χ| peaks inside the true cylinder (centred at origin)
    lam0 = C0 / prob["f"]
    R_cyl = 0.3 * lam0
    peak = prob["centers"][np.argmax(np.abs(chi_hat))]
    assert np.hypot(*peak) < R_cyl          # peak lands within the object support
    # quantitative: loose threshold (Born model error is real)
    assert rel_l2_error(chi_hat, prob["chi_true"]) < 0.6


# ---------- I1.5: select by name ----------
def test_I1_5_build_by_name():
    assert "born" in available("inverter")
    inv = build("inverter", "born", mu=1e-2)
    assert isinstance(inv, BornInverter) and inv.mu == 1e-2

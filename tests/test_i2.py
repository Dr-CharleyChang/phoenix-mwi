"""I2 self-test checklist (I2.1–I2.5) as executable tests — DBIM.

RED until you implement the TODOs in ``mwisim/inverse/dbim.py``:
    simulate_scattered_data  ·  DBIMInverter.reconstruct
(the GIVEN helpers distorted_green_matrix / build_frechet_operator make I2.2/I2.5
runnable already). Tutorial: docs/I2_Tutorial_Distorted-Born-iterative-method.md.

Run:  python -m pytest tests/test_i2.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.mom import build_D, solve_total_field, scattered_field
from mwisim.metrics import rel_l2_error
from mwisim.core.registry import build
from mwisim.inverse.born import BornInverter
from mwisim.inverse.dbim import (
    simulate_scattered_data,
    distorted_green_matrix,
    build_frechet_operator,
    DBIMInverter,
    make_dbim_problem,
)


# ---------- independent full-forward helpers (built from validated F1 code, ----------
# ---------- so they don't depend on the TODO simulate_scattered_data) ----------------
def _forward(data, chi):
    """Full nonlinear forward F(χ): stacked scattered data at the receivers."""
    centers, dS, k_b, rx = data["centers"], data["dS"], data["k_b"], data["rx"]
    d_side = np.sqrt(dS)
    D = build_D(centers, chi, k_b, d_side)
    blocks = [scattered_field(rx, centers, chi, solve_total_field(D, E_inc), k_b, dS)
              for E_inc in data["E_inc_set"]]
    return np.concatenate(blocks)


def _total_fields(data, chi):
    """Per-incidence interior total fields (N_v, N) at contrast χ."""
    centers, k_b = data["centers"], data["k_b"]
    d_side = np.sqrt(data["dS"])
    D = build_D(centers, chi, k_b, d_side)
    return np.stack([solve_total_field(D, E_inc) for E_inc in data["E_inc_set"]])


# ---------- I2.1: the forward re-simulation is wired correctly ----------
def test_I2_1_simulate_matches_measured():
    data = make_dbim_problem(eps_r=1.4, n_per_lambda=8, n_views=6, n_rx=20)
    d_side = np.sqrt(data["dS"])
    d_sim, E_tot_set = simulate_scattered_data(
        data["centers"], data["chi_true"], data["k_b"], d_side, data["dS"],
        data["E_inc_set"], data["rx"])
    assert d_sim.shape == data["d"].shape
    assert E_tot_set.shape == data["E_inc_set"].shape
    assert rel_l2_error(d_sim, data["d"]) < 1e-10   # same full-forward path as the data


# ---------- I2.2: the distorted-Born adjoint gate (given operator wiring) ----------
def test_I2_2_distorted_born_adjoint():
    data = make_dbim_problem(eps_r=1.4, n_per_lambda=8, n_views=4, n_rx=16)
    d_side = np.sqrt(data["dS"])
    chi = data["chi_true"]
    E_tot_set = _total_fields(data, chi)
    G_tr_dist = distorted_green_matrix(data["centers"], chi, data["k_b"], d_side,
                                       data["dS"], data["rx"], distorted=True)
    J = build_frechet_operator(data["centers"], data["rx"], E_tot_set,
                               data["k_b"], data["dS"], G_tr_dist)
    A = J.as_linear_operator()
    rng = np.random.default_rng(0)
    N = data["centers"].shape[0]
    Md = A.shape[0]
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    u = rng.standard_normal(Md) + 1j * rng.standard_normal(Md)
    lhs = np.vdot(A.matvec(v), u)      # <J v, u>
    rhs = np.vdot(v, A.rmatvec(u))     # <v, Jᴴ u>
    assert abs(lhs - rhs) <= 1e-9 * (abs(lhs) + abs(rhs))


# ---------- I2.3 (headline): DBIM explains the nonlinear data far better than Born ----------
def test_I2_3_dbim_beats_born():
    data = make_dbim_problem(eps_r=1.5, n_per_lambda=8, n_views=12, n_rx=32)
    chi_born, _ = BornInverter(mu=1e-2).reconstruct(data)
    chi_dbim, info = DBIMInverter(mu=1e-2, max_outer=12, tol=1e-3).reconstruct(data)

    res_born = rel_l2_error(_forward(data, chi_born), data["d"])   # ||F(χ̂)−d|| / ||d||
    res_dbim = rel_l2_error(_forward(data, chi_dbim), data["d"])
    assert res_dbim < 0.3 * res_born     # DBIM fits the TRUE nonlinear data

    err_born = rel_l2_error(chi_born, data["chi_true"])
    err_dbim = rel_l2_error(chi_dbim, data["chi_true"])
    assert err_dbim < err_born           # ...and the χ-map is closer to truth


# ---------- I2.4: the outer residual actually decreases / converges ----------
def test_I2_4_residual_decreases():
    data = make_dbim_problem(eps_r=1.5, n_per_lambda=8, n_views=12, n_rx=32)
    _, info = DBIMInverter(mu=1e-2, max_outer=12, tol=1e-3).reconstruct(data)
    hist = info["res_history"]
    assert len(hist) >= 2
    assert hist[-1] < 0.3 * hist[0]      # overall decreasing
    assert hist[-1] < 1e-2               # reached a small residual


# ---------- I2.5: build-by-name ----------
def test_I2_5_build_by_name():
    inv = build("inverter", "dbim", mu=1e-2)
    assert isinstance(inv, DBIMInverter)
    assert hasattr(inv, "reconstruct")

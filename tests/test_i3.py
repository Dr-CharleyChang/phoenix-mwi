"""I3 self-tests: CSI (Contrast Source Inversion)."""
from __future__ import annotations

import numpy as np

from mwisim.core.registry import build
from mwisim.metrics import rel_l2_error
from mwisim.inverse.born import BornInverter
from mwisim.inverse.csi import (
    CSIInverter,
    receiver_operator,
    domain_green_matrix,
    simulate_csi_data,
    update_contrast_sources,
    update_contrast,
    make_csi_problem,
)
from mwisim.inverse.dbim import simulate_scattered_data


def _full_forward_residual(data, chi):
    d_side = np.sqrt(data["dS"])
    d_sim, _ = simulate_scattered_data(
        data["centers"], chi, data["k_b"], d_side, data["dS"], data["E_inc_set"], data["rx"]
    )
    return rel_l2_error(d_sim, data["d"])


def test_I3_1_csi_helpers_have_expected_shapes():
    data = make_csi_problem(eps_r=1.3, n_per_lambda=6, n_views=4, n_rx=12)
    S = receiver_operator(data["rx"], data["centers"], data["k_b"], data["dS"])
    G_dom = domain_green_matrix(data["centers"], data["k_b"], np.sqrt(data["dS"]))

    Nv, N = data["E_inc_set"].shape
    M = data["rx"].shape[0]
    W = np.zeros((Nv, N), dtype=complex)

    assert S.shape == (M, N)
    assert G_dom.shape == (N, N)
    assert simulate_csi_data(S, W).shape == (Nv * M,)


def test_I3_2_source_and_contrast_updates_are_dimensionally_consistent():
    data = make_csi_problem(eps_r=1.3, n_per_lambda=6, n_views=4, n_rx=12)
    chi0 = np.zeros(data["centers"].shape[0], dtype=complex)
    S = receiver_operator(data["rx"], data["centers"], data["k_b"], data["dS"])
    G_dom = domain_green_matrix(data["centers"], data["k_b"], np.sqrt(data["dS"]))

    W = update_contrast_sources(
        chi0, data["E_inc_set"], data["d"], S, G_dom, mu_w=1e-3, xi=1.0
    )
    chi1, E_tot_set = update_contrast(
        W, data["E_inc_set"], G_dom, mu_chi=1e-2, project_real=True
    )

    assert W.shape == data["E_inc_set"].shape
    assert E_tot_set.shape == data["E_inc_set"].shape
    assert chi1.shape == (data["centers"].shape[0],)
    assert np.all(chi1.real >= -1e-12)


def test_I3_3_csi_beats_born_on_stronger_scatterer():
    data = make_csi_problem(eps_r=1.5, n_per_lambda=6, n_views=8, n_rx=20)
    chi_born, _ = BornInverter(mu=1e-2, iter_lim=250).reconstruct(data)
    chi_csi, info = CSIInverter(
        mu_chi=1e-2, mu_w=1e-3, xi=1.0, max_outer=10, tol=5e-3, step=1.0
    ).reconstruct(data)

    res_born = _full_forward_residual(data, chi_born)
    res_csi = _full_forward_residual(data, chi_csi)
    err_born = rel_l2_error(chi_born, data["chi_true"])
    err_csi = rel_l2_error(chi_csi, data["chi_true"])

    assert info["outer_iters"] >= 1
    assert res_csi < res_born
    assert err_csi < err_born


def test_I3_4_csi_build_by_name():
    inv = build("inverter", "csi", mu_chi=1e-2)
    assert isinstance(inv, CSIInverter)
    assert hasattr(inv, "reconstruct")

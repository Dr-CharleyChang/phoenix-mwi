"""Phase 0 self-tests (P1-P6): the interface layer + registry + faithful refactor.

The contract for Phase 0 is "scaffolding that changes no physics": the new ABCs/registry
behave correctly, and the MoM2D / CirclePhantom adapters reproduce the validated F1/F2
results exactly. Run:  python -m pytest tests/test_phase0.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.core.interfaces import ForwardSolver
from mwisim.core.registry import register, build, available, REGISTRY
from mwisim.grid import make_grid, assign_contrast
from mwisim.mom import build_D, incident_plane_wave, solve_total_field, scattered_field
from mwisim.mie import mie_scattered
from mwisim.metrics import rel_l2_error
from mwisim.phantoms.circle import CirclePhantom   # importing registers "phantom/circle"
from mwisim.forward.mom2d import MoM2D             # importing registers "forward/mom2d"

C0 = 299_792_458.0


def _problem(eps_r=2.0, n_per_lambda=12, f=1e9, domain_factor=2.0, eps_b=1.0):
    """A small circular-cylinder problem (mirrors test_f2._full_setup)."""
    lam0 = C0 / f
    R_cyl = 0.3 * lam0
    lam1 = lam0 / np.sqrt(eps_r)
    d = lam1 / n_per_lambda
    ph = CirclePhantom(R_cyl=R_cyl, eps_r=eps_r, d=d, eps_b=eps_b,
                       domain_factor=domain_factor)
    return ph, f


def _ring(R_obs, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([R_obs * np.cos(a), R_obs * np.sin(a)])


# ---------- P1: ABCs are enforced ----------
def test_P1_abc_enforced():
    # cannot instantiate an abstract interface
    with pytest.raises(TypeError):
        ForwardSolver()

    # a subclass missing a method is still abstract -> cannot instantiate
    class Incomplete(ForwardSolver):
        def solve_total_field(self, phantom, freq, **kw):
            return None, {}
        # scattered_field NOT implemented

    with pytest.raises(TypeError):
        Incomplete()

    # a complete subclass instantiates fine
    class Complete(ForwardSolver):
        def solve_total_field(self, phantom, freq, **kw):
            return None, {}
        def scattered_field(self, phantom, E_tot, rx, freq, **kw):
            return None

    Complete()  # no error


# ---------- P2: registry register/build roundtrip ----------
def test_P2_registry_roundtrip():
    @register("inverter", "_dummy_p2")
    class Dummy:
        def __init__(self, x=1):
            self.x = x

    assert "_dummy_p2" in available("inverter")
    obj = build("inverter", "_dummy_p2", x=5)
    assert isinstance(obj, Dummy) and obj.x == 5

    # duplicate name is rejected
    with pytest.raises(KeyError):
        @register("inverter", "_dummy_p2")
        class Dummy2:
            pass

    # cleanup so re-runs in the same session don't trip the duplicate guard
    del REGISTRY["inverter"]["_dummy_p2"]


# ---------- P3: CirclePhantom reproduces the raw F1 grid/contrast ----------
def test_P3_phantom_matches_raw():
    ph, _ = _problem()
    centers_raw, dS_raw = make_grid(ph.domain_size, ph.d)
    chi_raw = assign_contrast(centers_raw, ph.R_cyl, ph.eps_r, ph.eps_b)
    centers, dS = ph.grid()
    assert np.allclose(centers, centers_raw)
    assert dS == pytest.approx(dS_raw)
    assert np.allclose(ph.contrast(), chi_raw)


# ---------- P4: MoM2D(dense) reproduces the raw F1 pipeline, and matches Mie ----------
def test_P4_mom2d_dense_faithful_and_matches_mie():
    ph, f = _problem(eps_r=2.0, n_per_lambda=20)   # 20/λ for a clean Mie comparison
    fs = MoM2D(method="dense")
    E_tot, info = fs.solve_total_field(ph, f)

    # (a) adapter == calling the raw functions directly
    centers, dS = ph.grid(); d = np.sqrt(dS)
    k_b = ph.background_wavenumber(f); chi = ph.contrast()
    E_inc = incident_plane_wave(centers, k_b)
    E_ref = solve_total_field(build_D(centers, chi, k_b, d), E_inc)
    assert rel_l2_error(E_tot, E_ref) < 1e-12

    # (b) scattered field validates against the analytic Mie series
    rx = _ring(3 * ph.R_cyl)
    E_sc = fs.scattered_field(ph, E_tot, rx, f)
    E_mie = mie_scattered(rx, k_b, ph.inside_wavenumber(f), ph.R_cyl)
    assert rel_l2_error(E_sc, E_mie) < 0.05


# ---------- P5: cgfft backend matches dense backend ----------
@pytest.mark.parametrize("eps_r", [2.0, 8.0])
def test_P5_cgfft_matches_dense(eps_r):
    ph, f = _problem(eps_r=eps_r, n_per_lambda=12)
    E_dense, _ = MoM2D(method="dense").solve_total_field(ph, f)
    E_cg, info = MoM2D(method="cgfft", tol=1e-10).solve_total_field(ph, f)
    assert info["iters"] >= 1
    assert rel_l2_error(E_cg, E_dense) < 1e-7


# ---------- P6: select implementations by name through the registry ----------
def test_P6_build_by_name():
    assert "mom2d" in available("forward")
    assert "circle" in available("phantom")

    ph = build("phantom", "circle", R_cyl=0.03, eps_r=2.0, d=0.005)
    fs = build("forward", "mom2d", method="dense")
    assert isinstance(fs, MoM2D)
    E_tot, info = fs.solve_total_field(ph, 1e9)
    assert E_tot.shape[0] == ph.grid()[0].shape[0]   # one field value per cell

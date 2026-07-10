"""I2 driver: DBIM — reconstruct a strong-scatterer contrast map and plot it.

Run:
    python scripts/run_i2.py

Outputs:
    docs/fig_i2_chi.png        true χ | one-step Born χ̂ | DBIM χ̂   (real part)
    docs/fig_i2_residual.png   data residual ‖d − F(χ)‖/‖d‖ vs outer iteration
                               (with the single-step Born residual as a reference line)

The orchestration is complete; the physics/algorithm lives in mwisim/inverse/dbim.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mwisim.inverse.born import BornInverter
from mwisim.inverse.dbim import make_dbim_problem, DBIMInverter, simulate_scattered_data
from mwisim.metrics import rel_l2_error
from mwisim.core.registry import build  # noqa: F401  (shows build-by-name is available)

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def _forward_residual(prob, chi):
    """Full nonlinear data residual ‖d − F(χ)‖ / ‖d‖ (the honest DBIM score)."""
    d_side = np.sqrt(prob["dS"])
    d_sim, _ = simulate_scattered_data(prob["centers"], chi, prob["k_b"], d_side,
                                       prob["dS"], prob["E_inc_set"], prob["rx"])
    return rel_l2_error(d_sim, prob["d"])


def main():
    # A scatterer strong enough that a single Born step is poor (ε_r = 1.5).
    prob = make_dbim_problem(eps_r=1.5, n_per_lambda=12, n_views=16, n_rx=40)

    # --- one-step Born (I1) for comparison ---
    chi_born, _ = BornInverter(mu=1e-2, iter_lim=400).reconstruct(prob)

    # --- DBIM (I2) ---
    inv = DBIMInverter(mu=1e-2, max_outer=12, inner_iter=200, tol=1e-3)
    chi_dbim, info = inv.reconstruct(prob)

    # scores
    err_born = rel_l2_error(chi_born, prob["chi_true"])
    err_dbim = rel_l2_error(chi_dbim, prob["chi_true"])
    res_born = _forward_residual(prob, chi_born)
    res_dbim = _forward_residual(prob, chi_dbim)
    print(f"[I2] outer_iters={info.get('outer_iters')}  final data-res={res_dbim:.3%}")
    print(f"     χ-err:  Born={err_born:.1%}   DBIM={err_dbim:.1%}")
    print(f"     data-res(full forward):  Born={res_born:.1%}   DBIM={res_dbim:.1%}")

    # ---- figure 1: true vs Born vs DBIM χ-maps ----
    n_side = int(round(np.sqrt(prob["chi_true"].size)))
    imgs = [prob["chi_true"], chi_born, chi_dbim]
    titles = ["true χ (Re)", f"Born χ̂  err={err_born:.0%}", f"DBIM χ̂  err={err_dbim:.0%}"]
    vmax = float(np.max(prob["chi_true"].real))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, img, t in zip(ax, imgs, titles):
        im = a.imshow(img.reshape(n_side, n_side).real, cmap="viridis", vmin=0, vmax=vmax)
        a.set_title(t)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("I2: DBIM vs one-step Born", fontweight="bold")
    fig.tight_layout()
    out1 = os.path.join(DOCS, "fig_i2_chi.png")
    fig.savefig(out1, dpi=130)
    print("saved", out1)

    # ---- figure 2: outer residual history ----
    hist = info.get("res_history", [])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(range(len(hist)), hist, "o-", label="DBIM data residual")
    ax2.axhline(res_born, ls="--", color="C3", label=f"one-step Born ({res_born:.0%})")
    ax2.set_xlabel("outer iteration")
    ax2.set_ylabel(r"$\|d - F(\chi)\| / \|d\|$")
    ax2.set_title("I2: DBIM convergence")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    out2 = os.path.join(DOCS, "fig_i2_residual.png")
    fig2.savefig(out2, dpi=130)
    print("saved", out2)


if __name__ == "__main__":
    main()

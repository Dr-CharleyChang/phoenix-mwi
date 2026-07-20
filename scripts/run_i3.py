"""I3 driver: CSI — reconstruct a strong-scatterer contrast map and plot it."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mwisim.core.registry import build  # noqa: F401
from mwisim.inverse.born import BornInverter
from mwisim.inverse.csi import CSIInverter, make_csi_problem
from mwisim.inverse.dbim import simulate_scattered_data
from mwisim.metrics import rel_l2_error

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def _forward_residual(prob, chi):
    d_side = np.sqrt(prob["dS"])
    d_sim, _ = simulate_scattered_data(
        prob["centers"], chi, prob["k_b"], d_side, prob["dS"], prob["E_inc_set"], prob["rx"]
    )
    return rel_l2_error(d_sim, prob["d"])


def main():
    prob = make_csi_problem(eps_r=1.5, n_per_lambda=8, n_views=12, n_rx=32)

    chi_born, _ = BornInverter(mu=1e-2, iter_lim=300).reconstruct(prob)
    inv = CSIInverter(mu_chi=1e-2, mu_w=1e-3, xi=1.0, max_outer=12, tol=1e-3)
    chi_csi, info = inv.reconstruct(prob)

    err_born = rel_l2_error(chi_born, prob["chi_true"])
    err_csi = rel_l2_error(chi_csi, prob["chi_true"])
    res_born = _forward_residual(prob, chi_born)
    res_csi = _forward_residual(prob, chi_csi)
    print(f"[I3] outer_iters={info.get('outer_iters')}  final data-res={res_csi:.3%}")
    print(f"     chi-err:  Born={err_born:.1%}   CSI={err_csi:.1%}")
    print(f"     data-res(full forward):  Born={res_born:.1%}   CSI={res_csi:.1%}")

    n_side = int(round(np.sqrt(prob["chi_true"].size)))
    imgs = [prob["chi_true"], chi_born, chi_csi]
    titles = ["true chi (Re)", f"Born chi_hat  err={err_born:.0%}", f"CSI chi_hat  err={err_csi:.0%}"]
    vmax = float(np.max(prob["chi_true"].real))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, img, title in zip(ax, imgs, titles):
        im = a.imshow(img.reshape(n_side, n_side).real, cmap="viridis", vmin=0, vmax=vmax)
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("I3: CSI vs one-step Born", fontweight="bold")
    fig.tight_layout()
    out1 = os.path.join(DOCS, "fig_i3_chi.png")
    fig.savefig(out1, dpi=130)
    print("saved", out1)

    hist = info.get("data_res_history", [])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.semilogy(range(len(hist)), hist, "o-", label="CSI data residual")
    ax2.axhline(res_born, ls="--", color="C3", label=f"one-step Born ({res_born:.0%})")
    ax2.set_xlabel("outer iteration")
    ax2.set_ylabel(r"$\|d - S w\| / \|d\|$")
    ax2.set_title("I3: CSI convergence")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    out2 = os.path.join(DOCS, "fig_i3_residual.png")
    fig2.savefig(out2, dpi=130)
    print("saved", out2)


if __name__ == "__main__":
    main()

"""I1 driver: Born linear inversion — reconstruct a contrast map and plot it.

Run AFTER implementing the TODOs in mwisim/inverse/born.py:
    python scripts/run_i1.py

Output:
    docs/fig_i1_chi.png   true χ (real part) next to reconstructed χ̂

The orchestration is complete; the physics/algorithm lives in mwisim/inverse/born.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mwisim.inverse.born import make_born_problem, BornInverter
from mwisim.metrics import rel_l2_error
from mwisim.core.registry import build  # noqa: F401  (shows build-by-name is available)

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def main():
    # weak scatterer so the Born approximation is valid
    prob = make_born_problem(eps_r=1.1, n_per_lambda=12, n_views=16, n_rx=40,
                             mode="physical")

    inv = BornInverter(mu=1e-2, iter_lim=400)        # or: build("inverter","born", mu=1e-2)
    chi_hat, info = inv.reconstruct(prob)

    err = rel_l2_error(chi_hat, prob["chi_true"])
    print(f"[I1] solver={info.get('solver')}, iters={info.get('iters')}, "
          f"mu={info.get('mu')}, rel L2 err={err:.3%}")

    # reshape flat χ back to the 2-D grid for display
    n_side = int(round(np.sqrt(prob["chi_true"].size)))
    true_img = prob["chi_true"].reshape(n_side, n_side).real
    rec_img = chi_hat.reshape(n_side, n_side).real

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    im0 = ax[0].imshow(true_img, cmap="viridis"); ax[0].set_title("true χ (Re)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(rec_img, cmap="viridis"); ax[1].set_title(f"Born χ̂ (Re)  err={err:.1%}")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.suptitle("I1: Born linear inversion", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(DOCS, "fig_i1_chi.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

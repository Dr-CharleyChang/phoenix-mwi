"""F2 driver: CG-FFT vs dense — correctness, scaling, and the figure.

Produces docs/fig_f2_scaling.png and prints a benchmark table.  The point of F2
is that the matrix-free FFT operator turns an O(N^2)-memory / O(N^3)-solve dense
problem into an O(N)-memory / O(N log N)-per-iteration one, so problems that
blow up RAM densely run comfortably matrix-free.

Run:  python scripts/run_f2.py
"""
from __future__ import annotations

import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mwisim.grid import make_grid, assign_contrast
from mwisim.mom import build_D, incident_plane_wave, solve_total_field
from mwisim.metrics import rel_l2_error
from mwisim.operators import GreenFFT

C0 = 299_792_458.0


def setup(n_side_target: int, eps_r: float = 2.0, f: float = 1e9):
    """Build a problem whose grid is ~n_side_target cells per side."""
    lam0 = C0 / f
    k_b = 2 * np.pi / lam0
    R_cyl = 0.4 * lam0
    L = 1.2 * 2 * R_cyl
    d = L / n_side_target
    centers, dS = make_grid(L, d)
    chi = assign_contrast(centers, R_cyl, eps_r)
    return centers, chi, dS, k_b


def bench(n_side, eps_r=2.0, do_dense=True):
    centers, chi, dS, k_b = setup(n_side, eps_r)
    N = centers.shape[0]
    d = np.sqrt(dS)
    E_inc = incident_plane_wave(centers, k_b)

    row = {"N": N, "n_side": int(round(np.sqrt(N)))}

    # --- CG-FFT path ---
    t0 = time.perf_counter()
    op = GreenFFT(centers, chi, k_b, d)
    t_build_fft = time.perf_counter() - t0
    t0 = time.perf_counter()
    E_fft, info = op.solve_total_field(E_inc, tol=1e-8, method="bicgstab")
    t_solve_fft = time.perf_counter() - t0
    row.update(t_fft=t_build_fft + t_solve_fft, iters=info["iters"],
               mem_fft_MB=(op.G_hat.nbytes + 4 * N * 16) / 1e6)

    # --- dense path (optional; memory blows up) ---
    if do_dense:
        t0 = time.perf_counter()
        D = build_D(centers, chi, k_b, d)
        E_dir = solve_total_field(D, E_inc)
        t_dense = time.perf_counter() - t0
        row.update(t_dense=t_dense, mem_dense_MB=D.nbytes / 1e6,
                   match=rel_l2_error(E_fft, E_dir))
        del D, E_dir; gc.collect()
    else:
        row.update(t_dense=np.nan, mem_dense_MB=(N * N * 16) / 1e6, match=np.nan)
    return row


def main():
    # dense feasible only for small N (build_D allocates an (N,N,2) temp too)
    dense_sides = [16, 24, 32, 48, 64, 80]
    fft_only_sides = [112, 160, 224, 320]

    rows = []
    print(f"{'N':>7} {'iters':>5} {'t_fft(s)':>9} {'t_dense(s)':>11} "
          f"{'mem_fft(MB)':>11} {'mem_dense(MB)':>13} {'match':>9}")
    for s in dense_sides:
        r = bench(s, do_dense=True); rows.append(r)
        print(f"{r['N']:>7} {r['iters']:>5} {r['t_fft']:>9.3f} {r['t_dense']:>11.3f} "
              f"{r['mem_fft_MB']:>11.2f} {r['mem_dense_MB']:>13.1f} {r['match']:>9.1e}")
    for s in fft_only_sides:
        r = bench(s, do_dense=False); rows.append(r)
        print(f"{r['N']:>7} {r['iters']:>5} {r['t_fft']:>9.3f} {'--':>11} "
              f"{r['mem_fft_MB']:>11.2f} {r['mem_dense_MB']:>13.1f} {'--':>9}")

    N = np.array([r["N"] for r in rows], float)
    t_fft = np.array([r["t_fft"] for r in rows], float)
    t_dense = np.array([r.get("t_dense", np.nan) for r in rows], float)
    mem_fft = np.array([r["mem_fft_MB"] for r in rows], float)
    mem_dense = np.array([r["mem_dense_MB"] for r in rows], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # -- timing --
    md = ~np.isnan(t_dense)
    ax1.loglog(N[md], t_dense[md], "s-", color="#c0392b", label="dense (build+solve)")
    ax1.loglog(N, t_fft, "o-", color="#2471a3", label="CG-FFT (build+solve)")
    # reference slopes anchored to the dense/fft data
    Nref = np.array([N.min(), N.max()])
    ax1.loglog(Nref, t_dense[md][0] * (Nref / N[md][0]) ** 3, "k--", lw=1,
               alpha=0.6, label=r"$\propto N^3$ (dense solve)")
    ax1.loglog(Nref, t_fft[0] * (Nref / N[0]) * np.log2(Nref) / np.log2(N[0]),
               "k:", lw=1, alpha=0.6, label=r"$\propto N\log N$")
    ax1.set_xlabel("N (unknowns)"); ax1.set_ylabel("wall time (s)")
    ax1.set_title("Solve time: dense vs CG-FFT"); ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # -- memory --
    ax2.loglog(N, mem_dense, "s--", color="#c0392b", label=r"dense $D$: $16N^2$ B")
    ax2.loglog(N, mem_fft, "o-", color="#2471a3", label="CG-FFT: O(N)")
    ax2.axhline(8000, color="gray", ls=":", lw=1)
    ax2.text(N.min(), 9000, "8 GB", fontsize=8, color="gray")
    ax2.set_xlabel("N (unknowns)"); ax2.set_ylabel("operator memory (MB)")
    ax2.set_title("Memory footprint"); ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("F2: CG-FFT acceleration of the 2D MoM forward solver", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "fig_f2_scaling.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")

    # headline numbers
    big = rows[-1]
    print(f"\nLargest case: N={big['N']} solved matrix-free in {big['t_fft']:.2f}s "
          f"({big['iters']} iters); dense D alone would need "
          f"{big['mem_dense_MB']/1e3:.1f} GB.")


if __name__ == "__main__":
    main()

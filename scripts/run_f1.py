"""F1 driver: 2D MoM forward vs Mie analytic, produce the two result figures.

Run after implementing the stubs in mwisim/:
    python scripts/run_f1.py

Outputs:
    docs/fig_pointwise.png      MoM (dots) vs Mie (lines), Re/Im over rx angle
    docs/fig_convergence.png    rel L2 error vs in-medium cells-per-wavelength

The orchestration is complete; the physics lives in mwisim/ (which you implement).
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable without `pip install` (offline-friendly).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from mwisim.grid import make_grid, assign_contrast
from mwisim.mom import build_D, incident_plane_wave, solve_total_field, scattered_field
from mwisim.mie import mie_scattered
from mwisim.metrics import rel_l2_error, convergence_study

C0 = 299_792_458.0
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")

# ---- F1 parameters (first run: weak scatterer, easy to match) ----
P = dict(
    f=1e9,            # frequency [Hz]
    eps_b=1.0,        # background (vacuum)
    eps_r=2.0,        # cylinder permittivity (start weak; later 4..8)
    R_cyl=None,       # set below = 0.5 * lambda0
    R_obs=None,       # set below = 3 * R_cyl
    N_rx=72,          # receivers around the ring
    domain_factor=2.5,  # domain side = domain_factor * (2 R_cyl)
)


def _derived(P):
    # 定义一个函数：输入参数字典 P，计算并返回一组“派生量”（由基础参数推出来的量）

    lam0 = C0 / P["f"]
    # 自由空间波长 λ0 = c0 / f，其中 C0 是光速常量，P["f"] 是频率（Hz）

    k_b = 2 * np.pi / lam0 * np.sqrt(P["eps_b"])
    # 背景介质（b=background）的波数 k_b = k0 * sqrt(eps_b)，其中 k0=2π/λ0
    # 这里默认非磁性（μr=1），eps_b 是背景相对介电常数

    k_1 = 2 * np.pi / lam0 * np.sqrt(P["eps_r"])
    # 圆柱介质（1=inside cylinder）的波数 k_1 = k0 * sqrt(eps_r)
    # eps_r 是圆柱的相对介电常数（可能是实数或复数）

    R_cyl = P["R_cyl"] or 0.5 * lam0
    # 圆柱半径 R_cyl：如果 P["R_cyl"] 提供了（且为“真值”），就用它；
    # 否则用默认值 0.5*λ0。这里用 `or` 实现“未指定就用默认值”。

    R_obs = P["R_obs"] or 3 * R_cyl
    # 观测圆环半径 R_obs：如果 P["R_obs"] 提供了就用它；
    # 否则默认取 3*R_cyl（让接收点在圆柱外一定距离）

    lam1 = lam0 / np.sqrt(P["eps_r"].real if hasattr(P["eps_r"], "real") else P["eps_r"])
    # 介质内波长 λ1 = λ0 / sqrt(eps_r)
    # 若 eps_r 是复数/带 .real 属性，则取其实部 eps_r.real 来算“有效波长”（便于设定网格分辨率）；
    # 否则直接用 eps_r（通常为普通实数）

    return lam0, lam1, k_b, k_1, R_cyl, R_obs
    # 返回这些派生量，供后续计算（网格尺度、入射/散射计算、接收环位置等）使用


def rx_ring(R_obs, N_rx):
    ang = np.linspace(0, 2 * np.pi, N_rx, endpoint=False)  # ang is the receiver angles with shape (N_rx,)
    return np.column_stack([R_obs * np.cos(ang), R_obs * np.sin(ang)]), ang  # rx is the receiver positions with shape (N_rx, 2); ang is the receiver angles with shape (N_rx,)


def run_pointwise(P, n_per_lambda=15):
    lam0, lam1, k_b, k_1, R_cyl, R_obs = _derived(P)
    d = lam1 / n_per_lambda
    centers, dS = make_grid(P["domain_factor"] * 2 * R_cyl, d)  # centers is the cell centers with shape (N, 2); dS is the cell area d**2
    chi = assign_contrast(centers, R_cyl, P["eps_r"], P["eps_b"])
    # keep only scatterer cells (chi != 0) to shrink the system
    mask = chi != 0
    centers, chi = centers[mask], chi[mask]

    D = build_D(centers, chi, k_b, d)  # D is the MoM matrix with shape (N, N)
    E_inc = incident_plane_wave(centers, k_b)  # E_inc is the incident plane wave with shape (N,)
    E_tot = solve_total_field(D, E_inc)  # E_tot is the total field with shape (N,)

    rx, ang = rx_ring(R_obs, P["N_rx"])  # rx is the receiver positions with shape (N_rx, 2)
    # ang is the receiver angles with shape (N_rx,)
    E_mom = scattered_field(rx, centers, chi, E_tot, k_b, dS)  # E_mom is the scattered field with shape (N_rx,)
    E_mie = mie_scattered(rx, k_b, k_1, R_cyl)  # E_mie is the scattered field with shape (N_rx,)

    err = rel_l2_error(E_mom, E_mie)  # err is the relative L2 error between MoM and Mie
    print(f"[pointwise] cells/lambda={n_per_lambda}, N={len(chi)}, rel L2 err={err:.3%}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    deg = np.degrees(ang)
    ax[0].plot(deg, E_mie.real, "-", label="Mie")
    ax[0].plot(deg, E_mom.real, ".", label="MoM")
    ax[0].set(title="Re(E_sc)", xlabel="rx angle [deg]"); ax[0].legend()
    ax[1].plot(deg, E_mie.imag, "-", label="Mie")
    ax[1].plot(deg, E_mom.imag, ".", label="MoM")
    ax[1].set(title="Im(E_sc)", xlabel="rx angle [deg]"); ax[1].legend()
    fig.suptitle(f"MoM vs Mie  (eps_r={P['eps_r']}, rel L2={err:.2%})")
    fig.tight_layout()
    out = os.path.join(DOCS, "fig_pointwise.png")
    fig.savefig(out, dpi=130); print("saved", out)


def run_convergence(P, d_list_per_lambda=(8, 10, 15, 20, 30)):
    lam0, lam1, k_b, k_1, R_cyl, R_obs = _derived(P)
    d_list = [lam1 / n for n in d_list_per_lambda]
    params = dict(
        f=P["f"], eps_r=P["eps_r"], eps_b=P["eps_b"],
        R_cyl=R_cyl, R_obs=R_obs, N_rx=P["N_rx"],
        domain_size=P["domain_factor"] * 2 * R_cyl,   # 真实边长，对齐 convergence_study 的键
    )
    n_per_lambda, errs = convergence_study(d_list, params)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(n_per_lambda, errs, "o-")
    ax.set(xlabel="cells per (in-medium) wavelength", ylabel="rel L2 error",
           title="F1 convergence: MoM -> Mie")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(DOCS, "fig_convergence.png")
    fig.savefig(out, dpi=130); print("saved", out)


if __name__ == "__main__":
    run_pointwise(P)
    run_convergence(P)

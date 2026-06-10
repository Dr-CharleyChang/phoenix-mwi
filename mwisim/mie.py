"""Analytic Mie series for a dielectric circular cylinder (TM / E-polarization).

Tutorial ref: F1 §7.  This is the GROUND TRUTH used to validate the MoM solver.
Convention e^{+jωt} / H^(2).  Get this RIGHT FIRST — if the truth is wrong,
everything downstream is wrong.
"""
from __future__ import annotations
import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp


def mie_an(n: int, k_b: complex, k_1: complex, R_cyl: float) -> complex:
    """Scattering coefficient a_n for order n.

    a_n = - [k1 J'_n(k1 R) J_n(kb R)  - kb J_n(k1 R) J'_n(kb R)]
            / [k1 J'_n(k1 R) H^(2)_n(kb R) - kb J_n(k1 R) H^(2)'_n(kb R)]

    scipy.special: jv(n,x), jvp(n,x), hankel2(n,x), h2vp(n,x).
    TODO (F1 §7).
    """
    x1, xb = k_1 * R_cyl, k_b * R_cyl
    num_ = k_1 * jvp(n, x1) * jv(n, xb)      - k_b * jv(n, x1) * jvp(n, xb)
    den_ = k_1 * jvp(n, x1) * hankel2(n, xb) - k_b * jv(n, x1) * h2vp(n, xb)
    return -num_ / den_


def mie_scattered(
    rx_points: np.ndarray,
    k_b: complex,
    k_1: complex,
    R_cyl: float,
    Nmax: int | None = None,
) -> np.ndarray:
    """Analytic scattered field at exterior points (plane wave along +x, E0=1).

    E_sc(rho, phi) = sum_{n=-Nmax}^{Nmax} (-j)^n a_n H^(2)_n(k_b rho) e^{j n phi}

    Parameters
    ----------
    rx_points : (Nrx, 2) float   (must be outside the cylinder)
    Nmax : int, optional         default ~ ceil(|k_b| R_cyl) + 10

    Returns
    -------
    E_sc_mie : (Nrx,) complex

    TODO (F1 §7):
      1. rho, phi = polar(rx_points).
      2. default Nmax if None.
      3. sum series over n in [-Nmax, Nmax].
      4. sanity: increase Nmax until result stops changing.
    """
    rho = np.hypot(rx_points[:, 0], rx_points[:, 1])
    phi = np.arctan2(rx_points[:, 1], rx_points[:, 0])
    if Nmax is None:
        Nmax = int(np.ceil(np.abs(k_b) * R_cyl)) + 10
    E_sc_mie = np.zeros(len(rx_points), dtype=complex)
    for n in range(-Nmax, Nmax + 1):
        a_n = (-1j)**n * mie_an(n, k_b, k_1, R_cyl)
        H_n = hankel2(n, k_b * rho)
        phi_n = np.exp(1j * n * phi)
        E_sc_mie += a_n * H_n * phi_n
    return E_sc_mie
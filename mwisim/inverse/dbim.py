"""I2 — DBIM (Distorted Born Iterative Method): the nonlinear χ-map.

YOUR TASK (I2): implement the two TODO-marked pieces so ``tests/test_i2.py``
(I2.1–I2.5) goes green. Tutorial: ``docs/I2_Tutorial_Distorted-Born-iterative-method.md``.

What you implement (TODO):  simulate_scattered_data  ·  DBIMInverter.reconstruct
What is GIVEN (don't change unless you want to):  distorted_green_matrix ·
                            build_frechet_operator · make_dbim_problem ·
                            DBIMInverter.__init__ + registration.

Reuse: ``build_D`` / ``solve_total_field`` / ``scattered_field`` (mom.py),
``green_matrix`` / ``BornOperator`` (born.py, I1), ``A_op``/``AH_op`` (operators.py).
Convention e^{+jωt}/H^(2), single frequency, REAL background wavenumber k_b.

DBIM in one line: I1 in an outer loop — at each step re-solve the full forward
problem for the current χ, linearize around it (the *distorted* Born operator),
take one regularized LSMR step on the data residual, repeat.  See tutorial §3–§5.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from ..core.interfaces import Inverter
from ..core.registry import register
from ..mom import build_D, solve_total_field, scattered_field
from .born import BornOperator, green_matrix, plane_wave_incidences

C0 = 299_792_458.0


# --------------------------------------------------------------------------
# The full nonlinear forward map  F(χ)  (TODO — I2 §2, §6)
# --------------------------------------------------------------------------
def simulate_scattered_data(centers, chi, k_b, d, dS, E_inc_set, rx):
    """Full forward map F(χ): stacked scattered data + the interior total fields.

    For each incidence i:  solve (I − D(χ)) E_i = E_inc_i, then radiate E_i to rx.
    Return the stacked data ``(N_v·M,)`` AND the per-view total fields ``(N_v, N)``
    (the outer loop needs the total fields to build the Fréchet operator).

    Parameters
    ----------
    centers : (N,2)      cell centers
    chi     : (N,)       current contrast estimate
    k_b     : complex    background wavenumber (real magnitude here)
    d       : float      cell side  (note: build_D takes the side d, not the area)
    dS      : float      cell area  (= d**2)
    E_inc_set : (N_v,N)  incident field per direction
    rx      : (M,2)      receiver positions

    Returns
    -------
    d_sim     : (N_v·M,) complex   stacked scattered field at receivers
    E_tot_set : (N_v,N)  complex   total interior field per incidence

    TODO (I2 §2):
      1. D = build_D(centers, chi, k_b, d)                      # once, reused for all views
      2. for each E_inc_i in E_inc_set:
             E_tot_i = solve_total_field(D, E_inc_i)
             block_i = scattered_field(rx, centers, chi, E_tot_i, k_b, dS)
      3. d_sim = concatenate(blocks);  E_tot_set = stack(E_tot_i)
    Target tests: I2.1 (consistency), and it powers I2.3/I2.4 via reconstruct.
    """
    D = build_D(centers, chi, k_b, d)
    d_sim = np.zeros(rx.shape[0]*E_inc_set.shape[0], dtype=complex)
    E_tot_set = np.zeros_like(E_inc_set, dtype=complex)
    for i, E_inc_i in enumerate(E_inc_set):
        E_tot_i = solve_total_field(D, E_inc_i)
        block_i = scattered_field(rx, centers, chi, E_tot_i, k_b, dS)
        E_tot_set[i] = E_tot_i
        d_sim[i*rx.shape[0]:(i+1)*rx.shape[0]] = block_i
    return d_sim, E_tot_set


# --------------------------------------------------------------------------
# The distorted Born (Fréchet) operator  (GIVEN — I2 §3, §4)
# --------------------------------------------------------------------------
def distorted_green_matrix(centers, chi, k_b, d, dS, rx, distorted=True):
    """Distorted receiver×cell Green matrix G_tr^dist  (M, N).   [GIVEN — study, don't rewrite]

    Homogeneous receiver operator  S = k_b^2 dS G_tr  is bent *through the current
    object* χ by the multiple-scattering factor (I − D)^{-1} (tutorial §3–§4):

        S^dist = S + S diag(χ) (I − D)^{-1} G           (G = grid-to-grid Green op)

    computed one receiver-row at a time by RECIPROCITY (reuse one factorization):

        (I − D)^T Z = diag(χ) S^T            # M right-hand sides, shared transposed factor
        S^dist      = S + (G Z)^T            # G symmetric  ⇒  z^T G = (G z)^T

    Returns G_tr^dist = S^dist / (k_b^2 dS)  (BornOperator re-multiplies by k_b^2 dS).
    With ``distorted=False`` returns the plain homogeneous G_tr → this reduces DBIM to
    BIM (field updated, Green operator not).  Tutorial §4 "BIM vs DBIM".
    """
    N = centers.shape[0]
    G_tr = green_matrix(rx, centers, k_b)                 # (M, N) homogeneous, I1's helper
    if not distorted:
        return G_tr                                       # BIM: skip the object-bending term
    S = (k_b**2 * dS) * G_tr                              # (M, N) receiver operator 𝒮
    G_dom = build_D(centers, np.ones(N, dtype=complex), k_b, d)   # (N, N) grid-to-grid 𝒢 (χ≡1)
    D = build_D(centers, chi, k_b, d)                     # (N, N) domain operator D = 𝒢 diag(χ)
    ImD_T = (np.eye(N) - D).T
    lu, piv = sla.lu_factor(ImD_T)                        # factor once
    RHS = chi[:, None] * S.T                              # (N, M): column m = χ ⊙ 𝒮[m,:]
    Z = sla.lu_solve((lu, piv), RHS)                      # (N, M): (I−D)^T Z = RHS
    S_dist = S + (G_dom @ Z).T                            # (M, N): row m = 𝒮[m,:] + (𝒢 z_m)^T
    return S_dist / (k_b**2 * dS)                         # back out G_tr^dist


def build_frechet_operator(centers, rx, E_tot_set, k_b, dS, G_tr_dist):
    """Fréchet operator J of the forward map at the current χ.   [GIVEN — reuses I1]

    J is I1's Born operator with two substitutions (tutorial §3):
    homogeneous G_tr → distorted G_tr^dist, and incident field → current TOTAL field.
    We reuse ``BornOperator`` and overwrite its ``G_tr`` — matvec/rmatvec stay exact
    adjoints, so the I2.2 adjoint gate passes by construction.
    """
    op = BornOperator(centers, rx, E_tot_set, k_b, dS)    # builds a homogeneous G_tr internally
    op.G_tr = np.asarray(G_tr_dist)                       # ...overwrite it with the distorted one
    return op


# --------------------------------------------------------------------------
# The inverter (reconstruct is TODO — I2 §5, §6)
# --------------------------------------------------------------------------
@register("inverter", "dbim")
class DBIMInverter(Inverter):
    """Distorted Born Iterative Method: outer Gauss–Newton loop over I1 steps.

    Knobs (per-implementation, in __init__):
        mu        : Tikhonov weight on the UPDATE Δχ  (damp = sqrt(mu)).
        max_outer : max outer (re-linearization) iterations.
        inner_iter: max LSMR iterations for each Δχ solve.
        step      : Gauss–Newton step γ ∈ (0,1]  (χ ← χ + γ Δχ).
        tol       : stop when data residual ‖Δd‖/‖d‖ < tol.
        distorted : True = DBIM (update Green op); False = BIM (field only).
        solver    : "lsmr" (default) or "lsqr".
    """

    def __init__(self, mu: float = 1e-2, max_outer: int = 12, inner_iter: int = 200,
                 step: float = 1.0, tol: float = 1e-3, distorted: bool = True,
                 solver: str = "lsmr"):
        self.mu = float(mu)
        self.max_outer = int(max_outer)
        self.inner_iter = int(inner_iter)
        self.step = float(step)
        self.tol = float(tol)
        self.distorted = bool(distorted)
        self.solver = solver

    def reconstruct(self, data: dict, forward=None, x0=None, **kwargs):
        """Reconstruct χ̂ from ``data`` by DBIM. Returns ``(chi_hat, info)``.

        ``data`` keys (see make_dbim_problem / make_born_problem):
            centers, dS, k_b, rx, E_inc_set, d   (and chi_true for tests).
        ``forward`` is unused (we re-simulate with MoM here); it's the platform hook.
        ``x0`` optional warm start (e.g. the I1 Born estimate); default zeros.

        TODO (I2 §5):
          d_side = sqrt(dS);  chi = zeros(N) (or x0.copy())
          res_history = []
          for n in range(max_outer):
              d_sim, E_tot_set = simulate_scattered_data(centers, chi, k_b, d_side, dS, E_inc_set, rx)
              delta_d = d_meas - d_sim
              res = ||delta_d|| / ||d_meas||;  res_history.append(res)
              if res < tol: break
              G_tr_dist = distorted_green_matrix(centers, chi, k_b, d_side, dS, rx, self.distorted)
              J = build_frechet_operator(centers, rx, E_tot_set, k_b, dS, G_tr_dist)
              A = J.as_linear_operator()
              # LSMR (maxiter=) / LSQR (iter_lim=) on the RESIDUAL delta_d, damp=sqrt(mu):
              delta_chi = (spla.lsmr(A, delta_d, damp=sqrt(mu), maxiter=inner_iter))[0]  # or lsqr
              chi = chi + self.step * delta_chi
          return chi, {"outer_iters":..., "res_history":res_history, "mu":..., ...}
        Targets: I2.3 (beats Born), I2.4 (residual decreases/converges).
        """
        d_side = np.sqrt(data["dS"]); 
        chi = x0.copy() if x0 is not None else np.zeros(data["centers"].shape[0], dtype=complex)
        res_history = []
        for n in range(self.max_outer):
            d_sim, E_tot_set = simulate_scattered_data(
                data["centers"], chi, data["k_b"], d_side, data["dS"],
                data["E_inc_set"], data["rx"])
            delta_d = data["d"] - d_sim
            res = np.linalg.norm(delta_d) / np.linalg.norm(data["d"])
            res_history.append(res)
            if res < self.tol: break
            G_tr_dist = distorted_green_matrix(
                data["centers"], chi, data["k_b"], d_side, data["dS"], data["rx"], self.distorted)
            J = build_frechet_operator(
                data["centers"], data["rx"], E_tot_set, data["k_b"], data["dS"], G_tr_dist)
            A = J.as_linear_operator()
            delta_chi = (spla.lsmr(A, delta_d, damp=np.sqrt(self.mu), maxiter=self.inner_iter))[0]  # or lsqr
            chi = chi + self.step * delta_chi
        return chi, {"outer_iters":n+1, "res_history":res_history, "mu":self.mu, "step":self.step, "tol":self.tol, "distorted":self.distorted, "solver":self.solver}


# --------------------------------------------------------------------------
# Synthetic-problem assembler (GIVEN — reuses I1's physical forward data)
# --------------------------------------------------------------------------
def make_dbim_problem(eps_r=1.5, n_per_lambda=12, n_views=16, n_rx=40, f=1e9,
                      R_obs_factor=3.0) -> dict:
    """Assemble an I2 test problem: a scatterer strong enough that Born struggles.

    Reuses I1's ``make_born_problem`` in ``mode="physical"`` (data from the FULL forward
    solve per incidence — the honest nonlinear measurements DBIM must fit).  Default
    ε_r=1.5 is beyond the comfortable Born regime, so DBIM should clearly beat one Born step.
    Returns the same dict schema: centers, dS, k_b, rx, E_inc_set, d, chi_true, f.
    """
    from .born import make_born_problem
    return make_born_problem(eps_r=eps_r, n_per_lambda=n_per_lambda, n_views=n_views,
                             n_rx=n_rx, f=f, R_obs_factor=R_obs_factor, mode="physical")

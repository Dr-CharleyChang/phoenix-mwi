---
title: "I1 Milestone: Born linear inversion — the first χ-map"
tags: [MWI, MWT, milestone, I1, inversion, Born, LSQR, LSMR, adjoint, build-journal]
status: milestone — I1 complete
date: 2026-06-18
related: "[[PROJECT_PLAN]], [[I1_Tutorial_Born-linear-inversion]], [[CODE_GUIDE_codebase-and-algorithm-from-zero]], [[PHASE0_milestone]]"
---

# I1 Milestone — Born linear inversion (the first image)

> **In one line:** turned the corner from *forward* to *inverse* — from measured scattered fields, reconstruct the unknown contrast map χ̂ by solving one regularized linear least-squares problem with the matrix-free Born operator and its exact adjoint (`A_op`/`AH_op`) driven by LSMR. This is the project's first *image*, and the engine DBIM (I2) will drive. `pytest`: I1 **5/5**, full suite **28**.

## 1. What was built

All in `mwisim/inverse/born.py`, behind the Phase-0 `Inverter` interface:

| Piece | Role |
|---|---|
| `green_matrix(rx, centers, k_b)` | receiver×cell Green matrix `G_tr` `(M,N)` (reuses `green_2d`) |
| `plane_wave_incidences(centers, k_b, angles)` | incident field per direction `(N_v,N)`: $e^{-jk_b(\hat k_i\cdot r)}$ |
| `BornOperator.matvec` | $\mathbf A\chi$: per-view `A_op`, concatenate `(N_v·M,)` |
| `BornOperator.rmatvec` | $\mathbf A^H u$: per-view `AH_op`, **sum** over views `(N,)` |
| `BornInverter.reconstruct` | LSMR/LSQR with `damp=√μ`; returns `(χ̂, info)` |
| `make_born_problem` (given) | proto data schema: crime (`d=Aχ`) or physical (full forward per view) |
| `scripts/run_i1.py` (given) | reconstruct + plot true vs χ̂ → `docs/fig_i1_chi.png` |

`A_op`/`AH_op` (the per-view Born forward + adjoint) were already in `operators.py`; I1 stacks them over multiple incidences and wraps them as a SciPy `LinearOperator`.

## 2. Validation results

`tests/test_i1.py`: **5 passed**.

- **I1.1** — `green_matrix` shape `(M,N)` and depends only on distance.
- **I1.2 — the adjoint gate** — $\langle\mathbf A\chi,u\rangle=\langle\chi,\mathbf A^H u\rangle$ to **<1e-10** on random complex vectors. Nothing else can be trusted until this passes; it confirms `rmatvec` is the exact Hermitian adjoint of `matvec`.
- **I1.3 — inverse crime (data-fit)** — reconstruction explains its own data: $\lVert\mathbf A\hat\chi-\mathbf d\rVert/\lVert\mathbf d\rVert\approx$ **8e-6**.
- **I1.4 — physical recovery** — data from the full `MoM2D` forward solve (ε_r=1.1, multiview): χ̂ peak lands inside the true cylinder, relative error within the loose Born threshold (<0.6).
- **I1.5** — `build("inverter","born")` works.

Full suite after I1: **28 passed** (F1 8 incl. the new convergence test, F2 8, Phase-0 7, I1 5).

## 3. The key scientific insight: the inverse crime does *not* recover χ exactly

A natural expectation is "feed the inversion its own data (`d=Aχ_true`) and it returns χ_true." It doesn't — and that is **correct physics, not a bug**. The Born operator $\mathbf A$ (single frequency, a ring of receivers) is **rank-deficient**: many different χ produce the *same* data (limited resolution ≈ λ/2; high-spatial-frequency detail radiates weakly). So the least-squares solver returns the **minimum-norm** member of that solution set — which fits the data to ~1e-6 but is not χ_true (here χ-error ≈ 0.33). The honest inverse-crime check is therefore **data-fit**, not χ-equality; χ-quality is judged separately (I1.4), against a loose threshold. This limitation is exactly what regularization (`μ`), multi-frequency, and the nonlinear DBIM loop (I2) push back against.

## 4. Bug caught during this stage (review pays off)

While checking the convergence figure, found a wavenumber bug in `metrics.convergence_study`: `k_1 = 2π·f/lam0·√εr`, but `f/lam0 = f²/C0`, so `k_1` was off by a factor ~$10^9$ → the Mie ground truth was garbage and the F1 convergence curve sat **flat at ~93%** (never matching the milestone's "monotone" claim). Fixed to `k_1 = 2π·f/C0·√εr`; convergence now falls **~15% → ~1.4%** under refinement (overall-decreasing, with small staircasing bumps). Added the regression test `test_T8b_convergence_decreases` (no test had covered `convergence_study` — that gap let it through), and corrected the F1 milestone wording.

## 5. Debug war story (the two I1 bugs)

- **`plane_wave_incidences` shapes** — `k_hat=[cos,sin]` came out `(2,N_v)` and `np.dot(k_hat, centers)` demanded `N_v==N` → `ValueError`. Fix: `k_hat (N_v,2) @ centers.T (2,N)` → phase `(N_v,N)`.
- **`reconstruct` solver call** — `self.solver` is a *string* (not callable), and `lsmr`/`lsqr` return an 8-tuple, not `(x, info)`; also `lsqr` uses `iter_lim=` while `lsmr` uses `maxiter=`. Fix: dispatch on the string, unpack `out[0:3]=(x, istop, itn)`, build `info`.
- **The adjoint was right the first time** — `matvec`/`rmatvec` (per-view `A_op`/`AH_op`, sum on the adjoint) passed I1.2 immediately, which is the hard part.

## 6. Transferable lessons

1. **The adjoint test is the gate.** Make $\langle Av,u\rangle=\langle v,A^Hu\rangle$ pass before any reconstruction; a wrong adjoint makes Krylov LS silently fail.
2. **Know what your test actually guarantees.** "Inverse crime" guarantees *data-fit*, not parameter recovery, for a rank-deficient operator. Asserting the wrong thing wastes hours.
3. **Untested code drifts.** The `k_1` bug survived because nothing exercised `convergence_study`. Every claim in a milestone should have a test behind it.

## 7. What it unlocks — DBIM (I2)

DBIM is **I1 wrapped in an outer loop**: given the current χ̂, (a) solve the *full* forward field with `MoM2D` (drop the Born approximation), (b) re-linearize (update the incident → total field, rebuild the operator), (c) take a regularized Born/LSQR step on the residual, repeat. Same `A_op`/`AH_op`, same LSMR — now handling strong contrast and beating the resolution/accuracy limits seen here. The `forward` argument of `reconstruct` is the hook for exactly that.

## 8. Next steps

- **I2 — DBIM** (the nonlinear loop; the headline quantitative result).
- **CSI** as a second `Inverter` implementation (proves the abstraction; no per-iteration forward solve).
- **Evaluation** layer: SSIM, εr-RMSE, localization error for χ-maps.

---

*I1 closed 2026-06-18 · "the first picture of an unknown object" · adjoint-gated, data-fit-honest, 28/28.*

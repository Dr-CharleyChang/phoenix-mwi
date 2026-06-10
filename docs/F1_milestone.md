---
title: "F1 milestone: 2D MoM forward solver + Mie analytic validation (passed)"
tags: [MWI, milestone, F1, MoM, Mie, build-journal]
status: done
date: 2026-06-10
related: "F1_Tutorial_2D-MoM-and-Mie-validation.md"
---

# F1 milestone: the first analytically validated 2D MWI forward solver

> **In one line:** built a 2D TM microwave scattering forward solver (Richmond MoM /
> Lippmann–Schwinger) from scratch and validated it rigorously against the analytic
> Mie series — 3.15% pointwise error, monotone convergence, `pytest 7/7`. This is the
> foundation for the whole MWI project.

## 1. What was built

A complete "forward + validation" pipeline (`mwisim/`):

| Module | Functions | Purpose |
|---|---|---|
| `grid.py` | `make_grid`, `assign_contrast` | square-domain midpoint grid + contrast $\chi$ |
| `green.py` | `green_2d` | 2D free-space Green's function $\frac{1}{4j}H_0^{(2)}(k_bR)$ |
| `mom.py` | `build_D`, `incident_plane_wave`, `solve_total_field`, `scattered_field` | Richmond MoM discretization, plane-wave excitation, solve $(\mathbf I-\mathbf D)\mathbf E=\mathbf E_{inc}$, scattered field at receivers |
| `mie.py` | `mie_an`, `mie_scattered` | analytic series for a dielectric cylinder (ground truth) |
| `metrics.py` | `rel_l2_error`, `convergence_study` | error metric + grid-refinement study |

Physical setup: a plane wave $e^{-jk_bx}$ illuminates a dielectric cylinder
($\varepsilon_r=2\sim8$); the $e^{j\omega t}$/$H^{(2)}$ convention is used consistently
throughout.

## 2. Validation results

- **Pointwise** (`fig_pointwise.png`): on the receiver ring the MoM (dots) sits on
  the Mie (line), matching in both real and imaginary parts, relative $L_2$ error
  **3.15%** (weak scattering, `npl=15`).
- **Convergence** (`fig_convergence.png`): error decreases monotonically with
  cells-per-wavelength, log-log slope $\approx 1\sim2$ — proving the residual error is
  discretization that vanishes under refinement, not a bug.
- **Tests**: `pytest -q` → **7 passed**, covering weak-scatter sanity (T4), Mie
  self-convergence (T5), and MoM-vs-Mie at weak/strong contrast (T6 $\varepsilon_r=2$ /
  T8 $\varepsilon_r=8$).

## 3. Debug war story: the missing self-cell term (the AI was wrong, the physics was right)

> Worth recording separately — the most valuable lesson of this stage.

**Symptom**: weak scattering passed, but at $\varepsilon_r=2/8$ the MoM and Mie
disagreed by ~0.8–1.1 relative error.

**Triangulation, three moves**:
1. **Not a convention issue** — conjugate / sign-flip / complex-scale all made it
   worse, ruling out a global "$H^{(2)}\leftrightarrow H^{(1)}$" type error.
2. **Refining the grid left the error stuck at 0.80** — so the MoM converged
   self-consistently to a *stable wrong answer*; the bug was in a formula, not the
   resolution.
3. **Weak-limit three-way comparison** (MoM / Mie / **Born**) — at
   $\varepsilon_r\to1.01$ all three agreed (1.6–1.9% error). That pinned the bug to
   something that only shows under strong scattering = the **diagonal self term**
   (it only matters when multiple scattering dominates).

**Root cause**: the self-cell integral
$\int_0^a uH_0^{(2)}(k_bu)\,du=\big[uH_1^{(2)}(u)\big]_0^{k_ba}$ had its **lower limit**
treated as 0. In fact $\lim_{u\to0}uH_1^{(2)}(u)=\frac{2j}{\pi}\neq0$ (because $Y_1$
diverges at the origin). Restored, the correct self term is

$$D_{nn}=-\chi_n\Big[\tfrac{j\pi k_ba}{2}H_1^{(2)}(k_ba)+\underbrace{1}_{\text{the missing term}}\Big]$$

The fix is a single line: `np.fill_diagonal(D, pref*hankel2(1,k_b*a) - 1)`. After that,
**7/7 green**.

**Why it was invisible in the weak limit**: there $\mathbf D$ is tiny, so
$(\mathbf I-\mathbf D)\approx\mathbf I$ and the diagonal error is negligible; under
strong scattering multiple scattering amplifies it and the reconstruction breaks.

## 4. Transferable lessons

1. **The gold standard must be independent and made correct first.** Insisting on
   "validate Mie to self-convergence first, then use it to validate MoM" is the only
   reason the bug was catchable — if both were wrong you'd never know.
2. **Weak-limit three-way comparison is the killer diagnostic.** The Born
   approximation is unambiguous; using it as a third referee instantly separates
   "global convention error" from "local formula error".
3. **The convergence curve is a diagnostic, not just a result figure.** Error
   "stuck, not decreasing" points straight at a formula bug rather than insufficient
   resolution.
4. **The 2D MoM self-cell lower-limit singularity ($Y_1$ at the origin) is a classic
   trap.** The intuition transfers to 3D Green's functions and iterative-solver
   preconditioners.
5. **Verify AI-provided formulas too.** Here it was the tutorial formula that dropped
   a term, caught by physical diagnostics — "the AI writes the code, the human checks
   the physics."

## 5. Next steps

- **F2**: CG-FFT acceleration (matrix-free + Toeplitz FFT), paving the way for large /
  3D problems, with the FFT core converging with Zenith-Radar.
- **I1–I4**: inversion (Born → BIM/DBIM → CGLS/LSQR → PnP-DBIM) — from forward to imaging.
- **F3**: UWCEM phantom + Cole-Cole multi-frequency, closer to real tissue.

---

*F1 closed 2026-06-10 · from "muddling through" to "an analytically validated forward solver" · every bug logged.*

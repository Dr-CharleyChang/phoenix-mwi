# Phoenix Phase-1 benchmark report

> Generated automatically at `2026-07-19T18:56:39-07:00` by `python scripts/run_phase1_benchmark.py`.

## Problem

All methods used the same full-wave synthetic data: frequency `1 GHz`, grid `9 x 9` (`81` cells), `8` plane-wave views, `20` receivers, and maximum true relative permittivity `1.5` for the Phase-1 vacuum background.

DBIM and CSI use the already-computed Born map as a warm start. Their total runtime includes that shared Born time; `refine time` shows only the nonlinear refinement.

## Quantitative results

| Method | chi rel-L2 | eps_r RMSE | SSIM | Localization [mm] | Support IoU | Contrast recovery | Full data residual | Total time [s] | Refine time [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Born | 0.6146 | 0.1231 | 0.7100 | 0.000 | 0.6190 | 66.0% | 3.5004e-01 | 0.002 | 0.002 |
| DBIM | 0.3792 | 0.0760 | 0.8536 | 0.000 | 1.0000 | 85.7% | 1.6231e-03 | 0.201 | 0.200 |
| CSI | 0.4207 | 0.0843 | 0.7788 | 0.000 | 1.0000 | 74.3% | 1.1436e-01 | 0.064 | 0.062 |

DAS is a normalized qualitative energy image, not an estimate of chi, so RMSE and contrast recovery are intentionally not reported for it.

| Imager | SSIM | Localization [mm] | Support IoU | Runtime [s] |
| --- | ---: | ---: | ---: | ---: |
| DAS | 0.6175 | 0.000 | 0.6923 | 0.0008 |

![Phase-1 method comparison](fig_phase1_methods.png)

![Phase-1 residual comparison](fig_phase1_residuals.png)

## Acceptance gates

- [x] DBIM full-forward data fit is better than Born
- [x] CSI full-forward data fit is better than Born
- [x] DAS centroid lands inside the true object
- [x] All reconstructed arrays are finite

## How to read the residual figure

The left panel is the fair comparison: every method is re-simulated through the same nonlinear forward map and scored as `||F(chi_hat)-d||/||d||`. The right panel exposes each algorithm's own convergence history. DBIM's history is already a full-forward residual, while CSI's history is `||d-SW||/||d||`; those two curves diagnose their own algorithms but must not be compared point-for-point.

## Scope

This is a reproducible 2-D, single-frequency, synthetic-data platform benchmark. It validates software wiring and controlled inverse behavior; it is not evidence of clinical diagnostic performance. Phase 2 must add realistic dispersive tissue, calibration, noise/artifacts, and a public measured-data benchmark.

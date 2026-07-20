# P2-B UM-BMID measured-imaging benchmark report

**Scope:** UM-BMID Gen-One breast-phantom radar imaging benchmark; qualitative reflectivity only, not a clinical or quantitative-permittivity claim.

The benchmark used calibration scan IDs [1, 104, 135] and disjoint held-out evaluation scan IDs [13, 25, 36, 37, 117, 136, 147, 269]. It selected one global homogeneous propagation speed of 2.60×10⁸ m/s from the declared candidate grid; it did not tune speed per scan. Frequencies were restricted to 2.0–8.0 GHz and deterministically decimated to 51 points. The phase-centre radial offset was fixed at 0.0 mm.

## Held-out aggregate results

| Method | n | Median LE (cm) | Mean LE (cm) | Localized fraction | Median SCR (dB) |
|---|---:|---:|---:|---:|---:|
| DAS: empty reference | 8 | 4.09 | 4.24 | 0.0% | -3.34 |
| DAS: empty + angular mean | 8 | 2.18 | 2.96 | 62.5% | 0.34 |
| DAS: empty + rank-1 SVD | 8 | 3.75 | 4.21 | 25.0% | -1.05 |
| DAS: adipose reference | 8 | 4.80 | 4.13 | 25.0% | -2.87 |
| DAS: healthy reference | 8 | 1.58 | 2.09 | 75.0% | 1.54 |
| ORR: healthy reference | 8 | 2.31 | 2.44 | 62.5% | 0.10 |

## Acceptance gate

The predeclared spatial gate is evaluated on `das_empty_plus_angular_mean` over held-out scans: median localization error ≤ 3.0 cm and localized fraction ≥ 50%. Result: **PASS**.

## Interpretation boundary

This report validates a coordinate-aware qualitative radar reconstruction workflow on controlled breast phantoms. Healthy-reference subtraction is an experimental oracle because it uses a matched scan of the same phantom without the tumour; it is not available in ordinary clinical use. ORR reconstructs a radar reflectivity proxy under a homogeneous, primary-scatter model, not the dielectric contrast χ and not a diagnosis. A lower localization error or higher SCR here must not be translated into clinical sensitivity or specificity.

## Reproduction

Run `python scripts/run_p2b_measured_imaging.py`. The script verifies the pinned Gen-One archive, uses metadata-linked references, records every parameter in JSON, and regenerates this report and both figures.

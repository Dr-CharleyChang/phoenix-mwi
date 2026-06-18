# phoenix-mwi

2D/3D microwave imaging (MWI) forward & inverse simulation, aimed at breast and
bone-density detection research. Companion theory notes live in `docs/`.

**Current stage: F2 — CG-FFT acceleration ✅ done. Matrix-free `GreenFFT` in
`mwisim/operators.py` solves `(I - D)E = E_inc` in O(N log N); validated against the
F1 dense solve and Mie (`tests/test_f2.py`, T9-T14). Next: I1 — Born inversion.**

> Built in public · physics-first · every stage validated against an analytic
> solution or a public benchmark.

## Repository layout

```
phoenix-mwi/
├── mwisim/              # core library
│   ├── grid.py         # grid + contrast chi            (F1 §1, §3.1)
│   ├── green.py        # 2D Green's function G          (F1 §2)
│   ├── mom.py          # MoM forward: D matrix / incident field / solve / scattered field (F1 §3-§6)
│   ├── operators.py    # matrix-free GreenFFT + CG-FFT solve; A_op/AH_op (F2)
│   ├── mie.py          # analytic Mie series (ground truth)  (F1 §7)
│   ├── metrics.py      # error metric + convergence study    (F1 §8)
│   └── inverse/        # inversion stage: Born/BIM/DBIM/PnP (placeholder)
├── scripts/
│   ├── run_f1.py       # F1 driver: two figures + convergence curve
│   └── run_f2.py       # F2 driver: CG-FFT vs dense scaling benchmark + figure
├── tests/
│   ├── test_f1.py      # self-test checklist T1-T8
│   └── test_f2.py      # F2 checklist T9-T14 (FFT op vs dense, CG-FFT vs direct)
├── phantoms/           # UWCEM phantoms etc. (placeholder)
├── hls/                # Zynq HLS stage (placeholder; reuses Zenith-Radar FFT core)
└── docs/               # tutorial + figures
```

## Install (offline-friendly)

Only `numpy scipy matplotlib pytest` are required. **No `pip install -e .` needed** —
`scripts/run_f1.py` adds the repo root to `sys.path`, and tests run with `python -m pytest`.

```bash
python -c "import numpy, scipy, matplotlib, pytest; print('ok')"
# if missing:
#   conda install numpy scipy matplotlib pytest      (recommended)
#   pip install numpy scipy matplotlib pytest
```

## Running F1

```bash
python -m pytest -q          # T1-T8 self-tests; all green == F1 passes
python scripts/run_f1.py     # writes docs/fig_pointwise.png and docs/fig_convergence.png
```

`fig_pointwise.png` overlays the MoM scattered field (dots) on the analytic Mie
solution (line); `fig_convergence.png` shows relative L2 error vs cells-per-wavelength
on a log-log axis (monotone decreasing).

## Running F2

```bash
python -m pytest tests/test_f2.py -q   # T9-T14; all green == F2 passes
python scripts/run_f2.py               # writes docs/fig_f2_scaling.png
```

`fig_f2_scaling.png` plots wall-clock time and memory vs N for the matrix-free CG-FFT
solver against the dense MoM solve. The dense curve cuts off where the N×N matrix stops
fitting in RAM; the CG-FFT curve keeps going (N=102400 in ~0.5 s, 7 iterations).

## Roadmap

| Stage | Content | Status |
|---|---|---|
| **F1** | 2D MoM forward + Mie validation | ✅ done (pytest 7/7, 3.15% pointwise, monotone convergence) |
| **F2** | CG-FFT acceleration (matrix-free + Toeplitz) | ✅ done (pytest 8/8; N=102400 in 0.47 s / 7 iters; dense would need 167 GB) |
| F3 | UWCEM phantom + Cole-Cole multi-frequency | ⏳ |
| I1–I4 | Inversion: Born → BIM/DBIM → CGLS/LSQR → PnP-DBIM | ⏳ |
| HLS | Zynq-7020 FFT-core acceleration (reuses Zenith-Radar) | ⏳ |

## Validation philosophy

Rather than chasing agreement with real measurements (no phantom / no clinical data
here), each stage is checked against an **analytic solution** (Mie) and **public
benchmarks** (UWCEM phantoms, Institut Fresnel measured datasets). This yields
credible "validated" evidence from simulation alone.

## How to cite

If you use phoenix-mwi in academic work, please cite it. GitHub shows a "Cite this
repository" button generated from `CITATION.cff`; or cite as:

> Chang, C. (2026). *phoenix-mwi: a pluggable platform for 2D/3D microwave imaging and
> tomography* (Version 0.1.0) [Software]. https://github.com/Dr-CharleyChang/phoenix-mwi

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`). Free for academic, research, and
commercial use; you must retain the copyright and attribution notices when
redistributing. Academic citation is requested via `CITATION.cff` (see "How to cite").

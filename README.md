# phoenix-mwi

Phoenix-MWI is a pluggable Python platform for 2-D/3-D microwave imaging and tomography research, aimed ultimately at breast-cancer and bone-density applications.

**Current stage: Phase-1 hardening complete; Phase-2A measured-data foundation complete.** Phoenix now adds an axis-aware measured-data schema, checksum-pinned UM-BMID Gen-One ingestion, explicit pickle trust boundaries, complex-gain calibration, metadata-matched reference subtraction, and a reproduced public reference-subtraction/ICZT workflow. The locally verified suite is **78/78 passing**, including 16 Phase-2 tests.

> Built in public · physics-first · reproducible evidence · honest limitations

## What is implemented

- Dense 2-D method of moments and matrix-free CG-FFT forward solvers, validated against the analytic Mie solution.
- Born, DBIM, and CSI quantitative inversion behind one `Inverter` contract.
- Frequency-domain DAS qualitative imaging.
- Centered, off-centre, dual-target, and nested heterogeneous circular phantoms.
- Full-wave synthetic measurements with exact finite-sample complex SNR and controlled receiver-position error.
- YAML `SceneBuilder`, registry-driven algorithm construction, and an end-to-end `Phase1Pipeline`.
- Unified image/data metrics, connected-component checks, multi-seed mean and sample-standard-deviation statistics.
- Reproducible figures, JSON, Markdown reports, publication notebook, and GitHub Actions CI for Python 3.10/3.11.
- Named-axis `MeasurementSet` records with aligned coordinates/geometry/metadata, provenance, deterministic fingerprints, and pickle-free NPZ storage.
- UM-BMID Gen-One MAT/raw/explicitly trusted-pickle ingestion with checksum verification, safe ZIP extraction, canonical metadata, and SI-unit geometry aliases.
- Composable `ComplexGainCalibrator`, ID-matched `ReferenceSubtract`, and measured-data `PreprocessingPipeline`.
- Reproduced public UM-BMID sample-minus-empty-reference and 0–6 ns ICZT sinogram with floating-point-precision numerical gates.

## Repository layout

```text
phoenix-mwi/
├── mwisim/
│   ├── core/            # ABC contracts, registry, backend hook
│   ├── phantoms/        # single-circle and composite-circle scenes
│   ├── forward/         # MoM2D ForwardSolver adapter
│   ├── data/            # synthetic data + measured schema and UM-BMID ingest
│   ├── preprocessing/   # complex gain, reference subtraction, stage pipeline
│   ├── imaging/         # DAS qualitative imager
│   ├── inverse/         # Born, DBIM, and CSI
│   ├── config/          # safe YAML loading + YamlSceneBuilder
│   ├── evaluation/      # metrics, benchmark, hardening suite, statistics
│   ├── reporting/       # benchmark and hardening reporters
│   └── pipeline.py      # YAML/dict → scene → data → methods → metrics → report
├── examples/
│   └── phase1_hardening.yaml
├── scripts/
│   ├── run_f1.py / run_f2.py
│   ├── run_i1.py / run_i2.py / run_i3.py
│   ├── run_phase1_benchmark.py
│   ├── run_phase1_pipeline.py
│   ├── run_phase1_hardening.py
│   └── run_p2_um_bmid.py
├── notebooks/
│   └── phase1_hardening_platform_demo.ipynb
├── tests/               # 78 tests across physics, inversion, platform, hardening, and measured data
├── docs/                # tutorials, milestones, figures, JSON, and reports
└── .github/workflows/
    └── ci.yml
```

## Install

Python 3.10 or newer is required. The normal development installation includes NumPy, SciPy, Matplotlib, PyYAML, and pytest:

```bash
python -m pip install -e ".[dev]"
```

PyYAML is the formal YAML dependency. For an offline source checkout where the numerical dependencies already exist, Phoenix also contains a deliberately restricted safe YAML-subset parser so the included example remains runnable; it does not use `eval`.

## Verify everything

```bash
python -m pytest -q -p no:cacheprovider
```

The current local baseline is `78 passed`. GitHub Actions repeats the full suite on Python 3.10 and 3.11, validates the notebook JSON, runs the YAML Pipeline smoke example, and uploads its report artifacts. The 350 MB public archive is intentionally an opt-in system test rather than a normal CI download.

## Run the Phase-2A public measured-data benchmark

Download the pinned UM-BMID Gen-One archive from Zenodo, verify its byte count and MD5, safely extract it, ingest sample 1 and reference 12, subtract the complex reference, reproduce the official 0–6 ns ICZT workflow, and write JSON/PNG evidence:

```bash
python scripts/run_p2_um_bmid.py --download --sample-id 1
```

The archive and extracted data remain under the ignored `data/external/` directory. The generated evidence is written to `docs/phase2_um_bmid/`. The benchmark validates the documented measured-data preprocessing workflow; it is not a tumor-detection or clinical-accuracy claim.

## Run the Phase-1 hardening Pipeline

Run one declarative noisy heterogeneous experiment:

```bash
python scripts/run_phase1_pipeline.py --config examples/phase1_hardening.yaml --output-dir docs/phase1_pipeline_run
```

Run the canonical off-centre, dual-target, and nested-heterogeneous suite over three deterministic seeds:

```bash
python scripts/run_phase1_hardening.py --seeds 0,1,2 --output-dir docs
```

The suite writes:

- `docs/fig_phase1_hardening_examples.png` — representative truth/DAS/Born/DBIM/CSI maps.
- `docs/fig_phase1_hardening_statistics.png` — mean ± sample-standard-deviation comparisons.
- `docs/phase1_hardening_metrics.json` — structured per-run results, aggregates, and gates.
- `docs/phase1_hardening_report.md` — human-readable experiment report.

The three-seed suite is an engineering regression smoke test, not a publication-grade uncertainty estimate. At the current 9×9 grid and threshold, all four imaging methods merge the two dual-target supports into one connected component; this is retained as an explicit resolution limitation, not converted into a passing scientific claim.

## Earlier milestone drivers

```bash
python scripts/run_f1.py
python scripts/run_f2.py
python scripts/run_i1.py
python scripts/run_i2.py
python scripts/run_i3.py
python scripts/run_phase1_benchmark.py
```

## Documentation

- [Phase-1 hardening tutorial](docs/P1H_Tutorial_Phase1-hardening-and-release-pipeline.md) — from-zero explanation of composite scenes, noise, geometry mismatch, random seeds, statistics, YAML, Pipeline, CI, and the notebook.
- [YAML schema reference](docs/P1H_YAML_Schema_Reference.md) — every supported field and a minimal valid experiment.
- [Phase-1 hardening milestone](docs/P1H_milestone.md) — acceptance evidence, design decisions, measured results, and limitations.
- [Generated hardening report](docs/phase1_hardening_report.md) — current three-seed benchmark output.
- [Codebase and algorithms from zero](docs/CODE_GUIDE_codebase-and-algorithm-from-zero.md) — Python, forward physics, inverse methods, and Krylov solvers.
- [Phase-2 measured-data tutorial](docs/P2_Tutorial_Measured-Data-from-zero-to-100.md) — from school mathematics through complex S-parameters, named axes, calibration, reference matching, ICZT, security, and the public benchmark.
- [Phase-2 measurement schema reference](docs/P2_Measurement_Schema_Reference.md) — exact `MeasurementSet`, coordinate, geometry, storage, and preprocessing contracts.
- [Phase-2A milestone](docs/P2_milestone.md) — acceptance evidence, measured results, and the scientific boundary.

## Roadmap

| Stage | Content | Status |
| --- | --- | --- |
| F1 | 2-D dense MoM forward solver + Mie validation | ✅ complete |
| F2 | Matrix-free CG-FFT acceleration | ✅ complete |
| I1–I3 | Born → DBIM → CSI quantitative inversion | ✅ complete |
| P1-A/B/C | Unified metrics + common benchmark/report + DAS | ✅ complete |
| P1-H | Composite scenes + corruption + multi-seed stats + YAML/Pipeline + CI/notebook | ✅ complete (62/62 local tests) |
| Phase 2A | Measured-data schema/ingest + gain/reference preprocessing + UM-BMID workflow benchmark | ✅ complete (78/78 local tests) |
| Phase 2B | Measured monostatic DAS/ORR + artifact-removal ablation + localization statistics | ⏳ next |
| Phase 3 | 3-D solver adapters, dispersive phantoms, and VTK export | ⏳ planned |
| Phase 4 | Learned priors and physics-guided reconstruction | ⏳ planned |
| HLS | FPGA/Zynq acceleration behind the same kernel interfaces | ⏳ planned |

## Validation boundary

The forward solver has an analytic Mie anchor. The inverse/Pipeline results use known 2-D synthetic truth with controlled model mismatch. Phase 2A adds a real-data ingestion and preprocessing anchor, but it does not establish that the synthetic plane-wave Born/DBIM/CSI operators model UM-BMID's monostatic 3-D experiment. Antenna phase-center calibration, dispersive tissue, skin/clutter artifacts, measured spatial reconstruction, sparse clinical arrays, and clinical claims remain open.

## How to cite

If you use Phoenix-MWI in academic work, please cite it. GitHub can generate a citation from `CITATION.cff`, or use:

> Chang, C. (2026). *phoenix-mwi: a pluggable platform for 2D/3D microwave imaging and tomography* (Version 0.1.0) [Software]. https://github.com/Dr-CharleyChang/phoenix-mwi

## License

Apache License 2.0; see `LICENSE` and `NOTICE`. Academic citation is requested through `CITATION.cff`.

# P1-H Tutorial: harden Phase 1 from a successful demo into a reproducible research baseline

> This tutorial starts with elementary arithmetic and physical intuition, then derives noise, geometry mismatch, Monte-Carlo statistics, multi-target metrics, YAML configuration, Pipeline orchestration, CI, and the publication notebook. Read it after the P1-A/B/C tutorial.

## 0. What P1-H adds

P1-A/B/C proved that one centered homogeneous cylinder could travel through a complete platform loop. P1-H asks a more serious question: does the same software still behave sensibly when the target is not centered, when two objects must be separated, when one object contains different materials, and when the measurements and assumed geometry are imperfect?

The completed hardening path is:

~~~mermaid
flowchart LR
    Y["YAML experiment"] --> S["YamlSceneBuilder"]
    S --> P["CompositeCirclePhantom"]
    P --> D["SyntheticDataSource"]
    D --> C1["Complex Gaussian noise"]
    D --> C2["True vs assumed receiver geometry"]
    C1 --> PL["Phase1Pipeline"]
    C2 --> PL
    PL --> DAS["DAS"]
    PL --> B["Born"]
    PL --> DB["DBIM"]
    PL --> CSI["CSI"]
    DAS --> E["Unified metrics"]
    B --> E
    DB --> E
    CSI --> E
    E --> MC["Repeat seeds"]
    MC --> ST["Mean + sample standard deviation"]
    ST --> R["PNG + JSON + Markdown"]
    R --> CI["GitHub Actions artifact"]
    R --> NB["Publication notebook"]
~~~

The main additions are:

- A CompositeCirclePhantom for off-centre, multiple, and nested heterogeneous materials.

- A SyntheticDataSource implementing the existing DataSource interface.

- Exact-SNR circular complex Gaussian noise.

- Receiver-position model error in which data use true positions but reconstruction uses nominal positions.

- A schema-versioned YAML SceneBuilder and concrete Phase1Pipeline.

- Three canonical hardening scenarios and repeated random-seed statistics.

- Connected-component counting in addition to centroid, IoU, SSIM, contrast error, and data residual.

- A hardening reporter, two command-line drivers, GitHub Actions smoke run, and publication notebook.

## 1. Why the centered noiseless cylinder was necessary but insufficient

### 1.1 A controlled first experiment

The centered cylinder was valuable because symmetry made debugging easier. If a symmetric array and symmetric target produced a strongly asymmetric result, something was probably wrong in indexing, geometry, Green functions, or view stacking.

That same symmetry can later hide weaknesses. A method that always places energy near the image center may look successful on a centered target even when it does not truly use measurement phase correctly.

### 1.2 What “hardening” means

Hardening does not mean making the code impossible to fail. It means deliberately exposing the code to controlled failure mechanisms and recording what happens.

In P1-H, each difficulty has a specific purpose:

| Difficulty | What it tests |
| --- | --- |
| off-centre target | whether phase and physical coordinates determine location |
| dual target | whether spatial resolution separates two objects |
| heterogeneous target | whether the method recovers more than one contrast level |
| additive complex noise | sensitivity to random measurement error |
| receiver geometry error | sensitivity to a wrong forward model |
| several seeds | whether one lucky noise realization controls the conclusion |
| YAML | whether an experiment is reproducible without editing Python |
| Pipeline | whether components really share interfaces and one data schema |
| CI | whether a clean machine can repeat tests and a smoke experiment |
| notebook | whether a researcher can understand and rerun the result |

## 2. Composite scenes from basic geometry

### 2.1 One off-centre circle

A circle with center $(c_x,c_y)$ and radius $R$ contains a grid cell at $(x_n,y_n)$ when

$$
(x_n-c_x)^2+(y_n-c_y)^2\le R^2.
$$

The old centered cylinder was the special case $c_x=c_y=0$. P1-H stores the center explicitly, so the same equation covers any location inside the domain.

For example, let the center be $(0.06,-0.04)$ m and let a cell be at $(0.08,-0.02)$ m. The squared distance is

$$
(0.08-0.06)^2+(-0.02+0.04)^2
=0.02^2+0.02^2
=0.0008.
$$

The distance is $\sqrt{0.0008}\approx0.0283$ m. If the radius is $0.05$ m, this cell is inside.

### 2.2 Two targets

For two non-overlapping circles, Phoenix applies the inside test twice. A cell can be background, target 1, or target 2:

$$
\chi_n=
\begin{cases}
\varepsilon_{r,1}/\varepsilon_b-1,&r_n\in\Omega_1,\\
\varepsilon_{r,2}/\varepsilon_b-1,&r_n\in\Omega_2,\\
0,&\text{otherwise}.
\end{cases}
$$

The hardening dual target deliberately gives the two circles different permittivities. Therefore the inverse method must solve both a location problem and an amplitude problem.

### 2.3 Nested heterogeneous material

A heterogeneous object contains more than one material. P1-H represents a low-contrast host circle followed by a smaller high-contrast core. Inclusions are applied in list order. With overlap_policy set to last_wins, the core overwrites the host only inside the core.

Suppose $\varepsilon_b=1$, the host has $\varepsilon_r=1.25$, and the core has $\varepsilon_r=1.70$. Their contrasts are

$$
\chi_{\mathrm{host}}=1.25-1=0.25,
$$

$$
\chi_{\mathrm{core}}=1.70-1=0.70.
$$

A cell in the host but outside the core receives $0.25$. A cell inside both geometric circles receives $0.70$ because the later core wins.

### 2.4 Why clipping is rejected

If a circle extends outside the square imaging domain, silently clipping it changes the requested target. P1-H rejects the scene when

$$
|c_x|+R>L/2
$$

or

$$
|c_y|+R>L/2.
$$

The user must enlarge the domain, reduce the radius, or move the inclusion. Failing early is scientifically safer than simulating a different phantom without warning.

### 2.5 Grid resolution still matters

A geometric circle is represented only by cell centers. A small circle can contain no cell center on a coarse grid. CompositeCirclePhantom rejects that case and asks for a smaller cell size or larger inclusion.

The hardening benchmark uses a small $9\times9$ grid for fast repeated CI and Monte-Carlo execution. It is an engineering integration resolution, not a publication convergence resolution.

## 3. Circular complex Gaussian noise from zero

### 3.1 Why frequency-domain noise is complex

One measured frequency-domain sample has an in-phase and a quadrature component:

$$
d=a+jb.
$$

Noise can disturb both components. Circular complex Gaussian noise is written

$$
n=n_R+jn_I,
$$

where $n_R$ and $n_I$ are independent zero-mean Gaussian random variables with equal variance. “Circular” means no complex-plane direction is preferred.

### 3.2 Signal and noise energy

For a stacked vector $d$ containing all views and receivers, its Euclidean norm is

$$
\|d\|_2=\sqrt{\sum_k|d_k|^2}.
$$

The squared norm is total discrete signal energy:

$$
\|d\|_2^2=\sum_k|d_k|^2.
$$

### 3.3 SNR in decibels

Power SNR is

$$
\operatorname{SNR}=\frac{\|d\|_2^2}{\|n\|_2^2}.
$$

In decibels:

$$
\operatorname{SNR}_{\mathrm{dB}}
=10\log_{10}\left(\frac{\|d\|_2^2}{\|n\|_2^2}\right).
$$

Using $\log(a^2)=2\log(a)$ gives the equivalent amplitude-norm form:

$$
\operatorname{SNR}_{\mathrm{dB}}
=20\log_{10}\left(\frac{\|d\|_2}{\|n\|_2}\right).
$$

This explains the common 10-versus-20 confusion: use 10 for a power ratio and 20 for an amplitude ratio.

### 3.4 A hand example

If $\|d\|_2=10$ and the requested SNR is 20 dB, then

$$
20=20\log_{10}\left(\frac{10}{\|n\|_2}\right).
$$

Divide by 20:

$$
1=\log_{10}\left(\frac{10}{\|n\|_2}\right).
$$

Therefore

$$
10^1=\frac{10}{\|n\|_2},
$$

so

$$
\|n\|_2=1.
$$

At 40 dB, the amplitude ratio is $10^{40/20}=100$, so the same signal would require $\|n\|_2=0.1$.

### 3.5 How the code reaches the requested SNR exactly

First generate a random complex Gaussian direction $z$. Its random norm generally does not match the requested value. Compute the target noise norm:

$$
\|n\|_{\mathrm{target}}
=\frac{\|d\|_2}{10^{\operatorname{SNR}_{\mathrm{dB}}/20}}.
$$

Then scale:

$$
n=z\frac{\|n\|_{\mathrm{target}}}{\|z\|_2}.
$$

This retains the random Gaussian direction while forcing the finite realization to have the exact requested norm ratio. The noisy data are

$$
d_{\mathrm{noisy}}=d_{\mathrm{clean}}+n.
$$

### 3.6 Why clean data, noise, and noisy data are all stored

SyntheticDataSource returns d_clean, noise, and d. This lets a test verify

$$
d=d_{\mathrm{clean}}+n
$$

and independently recompute achieved SNR. Real measured data will not provide this decomposition; it exists here because a synthetic benchmark should expose its ground truth.

## 4. Receiver geometry error as a real model mismatch

### 4.1 The wrong simulation

A common mistake is to perturb receiver positions and then use those same perturbed positions for both data generation and inversion. That creates a different but perfectly known geometry. It does not test geometry uncertainty.

### 4.2 The P1-H simulation

P1-H creates nominal positions:

$$
r_m^{\mathrm{assumed}}.
$$

It draws position error:

$$
\Delta r_m\sim\mathcal N(0,\sigma_r^2I).
$$

The true physical positions are

$$
r_m^{\mathrm{true}}=r_m^{\mathrm{assumed}}+\Delta r_m.
$$

The measured data are generated using $r_m^{\mathrm{true}}$, but DAS, Born, DBIM, CSI, and the common scoring forward model use $r_m^{\mathrm{assumed}}$. The inversion is therefore solving with an imperfect Green matrix.

### 4.3 Why a millimetre matters at microwave frequency

In a simple propagation phase $e^{-jkR}$, a small path-length error $\Delta R$ creates approximately

$$
\Delta\phi\approx k\Delta R.
$$

At 1 GHz in vacuum:

$$
\lambda_0\approx0.2998\ \mathrm{m},
$$

$$
k=\frac{2\pi}{\lambda_0}\approx20.96\ \mathrm{rad/m}.
$$

A 1 mm path error gives

$$
\Delta\phi\approx20.96\times0.001=0.02096\ \mathrm{rad}\approx1.20^\circ.
$$

One degree may look small, but coherent imaging sums many views and receivers. Systematic or random phase errors can broaden, shift, or split a focus.

### 4.4 Residual floor

Even the true contrast may not make the assumed model reproduce the data exactly:

$$
F_{\mathrm{assumed}}(\chi_{\mathrm{true}})
\ne
F_{\mathrm{true}}(\chi_{\mathrm{true}}).
$$

Noise creates another irreducible term. Consequently, demanding a residual near machine precision is wrong for corrupted data. Regularization and stopping criteria must avoid chasing an unattainable perfect fit.

## 5. Random seeds and reproducibility

### 5.1 What a seed does

A pseudo-random generator is deterministic after its initial seed. Seed 7 always gives the same random sequence for the same algorithm and library behavior. This makes a supposedly random experiment repeatable.

The seed is not a claim that the data are natural or unbiased. It is simply the identifier of one reproducible realization.

### 5.2 Independent random streams

P1-H uses NumPy SeedSequence to spawn one random stream for geometry error and another for measurement noise. This is useful because changing the geometry-error option should not accidentally shift every subsequent noise draw merely by consuming a different number of random values.

### 5.3 Why one seed is not evidence

Suppose an algorithm happens to receive a noise realization that partially cancels its model error. It may look unusually good. Another seed may reinforce the error. Reporting only the first seed creates selection bias even if nobody intended to cheat.

Therefore the hardening driver repeats each scenario for seeds 0, 1, and 2 and reports both a mean and a sample standard deviation.

## 6. Monte-Carlo mean and sample standard deviation

### 6.1 Mean

For metric values $x_1,\ldots,x_K$, the arithmetic mean is

$$
\bar x=\frac{1}{K}\sum_{k=1}^{K}x_k.
$$

If three runs produce $1$, $3$, and $5$:

$$
\bar x=\frac{1+3+5}{3}=3.
$$

### 6.2 Deviations

Subtract the mean:

$$
1-3=-2,\qquad3-3=0,\qquad5-3=2.
$$

Square them:

$$
(-2)^2=4,\qquad0^2=0,\qquad2^2=4.
$$

### 6.3 Sample standard deviation

When the observed seeds are treated as a sample from possible corruptions, use denominator $K-1$:

$$
s=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(x_k-\bar x)^2}.
$$

For the example:

$$
s=\sqrt{\frac{4+0+4}{3-1}}=\sqrt4=2.
$$

This exact example is an executable unit test.

### 6.4 Population versus sample denominator

Dividing by $K$ computes the population standard deviation of exactly those listed values. Dividing by $K-1$ is the usual unbiased variance estimate when the listed values sample a larger random process. P1-H deliberately uses NumPy std with ddof=1.

### 6.5 What three seeds mean

Three seeds are enough to prove that repeated execution, aggregation, and error bars work. They are not enough for a strong publication uncertainty claim. A paper should choose seed count, corruption distributions, and confidence intervals before seeing the result. The appropriate number depends on runtime and required statistical precision.

## 7. Metrics for dual and heterogeneous targets

### 7.1 Why centroid alone can lie

Imagine two true targets at $x=-1$ and $x=+1$. Their combined centroid is $0$. An algorithm that reconstructs one false target at $x=0$ also has centroid $0$, so localization error could be zero while the reconstruction completely misses the two-object structure.

Therefore dual-target evaluation must combine centroid with support IoU, SSIM, and connected-component count.

### 7.2 Connected components

After thresholding the support, two cells belong to the same component when they touch horizontally, vertically, or diagonally. This is 8-connectivity on a 2-D grid.

The evaluator reports:

- component_count: number of reconstructed support components;

- true_component_count: number in the true support;

- component_count_error: absolute difference.

If truth has two targets and reconstruction merges them into one:

$$
e_{\mathrm{components}}=|1-2|=1.
$$

### 7.3 Threshold choice for heterogeneous material

The host contrast is $0.25$ and the core contrast is $0.70$. A threshold of 50% of maximum is $0.35$, which excludes the host. For the nested heterogeneous scenario, P1-H uses threshold 0.20 of maximum:

$$
0.20\times0.70=0.14.
$$

Both host $0.25$ and core $0.70$ are then included in the true support.

The threshold is part of the experiment configuration because changing it changes IoU, centroid, and component count.

## 8. YAML from zero

### 8.1 Why configuration is separate from Python

If geometry and hyperparameters live inside a driver, changing an experiment also changes source code. It becomes difficult to identify which exact experiment produced a figure.

YAML stores experiment choices as data. Python implements reusable behavior. This boundary enables versioned experiments, command-line runs, notebook runs, and CI runs to share one configuration.

### 8.2 Mapping

A YAML mapping is a key followed by a value:

~~~yaml
frequency_hz: 1.0e9
n_views: 8
~~~

This becomes a Python dictionary:

~~~text
{"frequency_hz": 1.0e9, "n_views": 8}
~~~

### 8.3 Nested mapping

Indentation creates a nested dictionary:

~~~yaml
corruption:
  snr_db: 25.0
  receiver_position_std_m: 0.001
  seed: 7
~~~

The spaces are structure. Tabs are rejected.

### 8.4 List

A dash starts one list item:

~~~yaml
inclusions:
  - label: host
    center_m: [0.02, 0.00]
    radius_m: 0.08
    eps_r: 1.25
  - label: core
    center_m: [0.05, 0.02]
    radius_m: 0.035
    eps_r: 1.70
~~~

Each list item becomes one inclusion dictionary.

### 8.5 Boolean and null

YAML true and false become Python booleans. null becomes Python None:

~~~yaml
warm_start: true
output_dir: null
~~~

### 8.6 Complex material values

For future lossy contrast, the schema accepts:

~~~yaml
eps_r:
  real: 4.5
  imag: -0.3
~~~

The sign of the imaginary part must follow the repository's $e^{+j\omega t}$ convention and the chosen constitutive model. P1-H's canonical scenarios remain real-valued.

### 8.7 PyYAML and the offline fallback

The package formally depends on PyYAML and uses yaml.safe_load when available. The current offline development environment did not have PyYAML, so Phoenix also includes a small safe fallback parser for the exact schema subset used here: mappings, indentation, lists, inline numeric lists, strings, numbers, booleans, null, and comments.

The fallback uses ast.literal_eval only for literal values; it never executes Python expressions. Unsupported YAML features fail with a clear error rather than being guessed.

The exact field reference is in [P1H_YAML_Schema_Reference.md](P1H_YAML_Schema_Reference.md).

## 9. SceneBuilder and schema validation

YamlSceneBuilder implements the existing SceneBuilder ABC:

$$
\text{YAML/dict}\longrightarrow\text{CompositeCirclePhantom}.
$$

It checks schema_version, requires a scene mapping, accepts only the Phase-1 composite_circles scene type, validates required scene and inclusion fields, and delegates physical validation such as clipping and overlap to CompositeCirclePhantom.

It is registered under scene_builder/yaml, so configuration-driven code can build it by name while ordinary Python users can instantiate YamlSceneBuilder directly.

## 10. SyntheticDataSource and the unified problem dictionary

SyntheticDataSource implements DataSource.measurements. It performs:

1. Obtain centers, cell area, contrast, and background wavenumber from Phantom.

2. Build all plane-wave incident fields.

3. Build nominal receiver-ring coordinates.

4. Draw true receiver-position errors.

5. Run the full nonlinear forward solver at true receiver coordinates.

6. Draw and scale complex noise.

7. Return observed data plus truth and corruption metadata.

Important returned fields include:

| Field | Meaning |
| --- | --- |
| rx | receiver positions assumed by algorithms |
| rx_true | positions used to generate data |
| rx_error | rx_true minus rx |
| d_clean | noiseless data at true positions |
| noise | scaled complex noise |
| d | observed d_clean + noise |
| snr_db_requested | configured SNR |
| snr_db_achieved | independently measured finite-vector SNR |
| receiver_position_rmse_m | realized coordinate-error RMS |
| seed | reproducibility identifier |

## 11. The concrete Phase1Pipeline

### 11.1 Responsibility

Phase1Pipeline orchestrates components. It does not contain Green functions, MoM equations, Born operators, DBIM derivatives, or CSI updates.

Its two main methods are:

- build_problem: YAML scene and acquisition settings to one problem dictionary;

- run: problem to DAS/Born/DBIM/CSI result, metrics, and optional report artifacts.

### 11.2 Why this is a platform boundary

The Pipeline knows interface names and shared dictionary keys. Each algorithm knows only its own mathematics. A future DataSource for measured data can replace SyntheticDataSource while preserving downstream contracts. A future imager or inverter can be selected through the registry without changing Pipeline physics.

### 11.3 Pipeline pseudocode

~~~text
load and validate schema
phantom = scene_builder.build(scene)

data_source = SyntheticDataSource(
    phantom,
    acquisition,
    corruption,
    seed
)
problem = data_source.measurements()

imager = registry.build("imager", configured_name, configured_params)
born = registry.build("inverter", configured_name, configured_params)
dbim = registry.build("inverter", configured_name, configured_params)
csi = registry.build("inverter", configured_name, configured_params)

result = run_phase1_benchmark(
    same problem,
    same evaluator threshold,
    same common nonlinear scoring,
    configured warm start
)

if output directory exists:
    BenchmarkReporter.write(result)

return result
~~~

### 11.4 Output path rule

A relative output path written inside YAML is interpreted relative to that YAML file. A relative path explicitly passed from Python or the command line is interpreted relative to the current working directory. This mirrors ordinary configuration-file and CLI expectations.

## 12. The three canonical hardening scenarios

### 12.1 off_center

- One target centered at $(0.06,-0.04)$ m.

- $\varepsilon_r=1.50$.

- 30 dB SNR.

- 0.5 mm receiver-coordinate standard deviation.

This is the cleanest test that reconstruction follows measured phase away from the origin.

### 12.2 dual_target

- Target 1 at $(-0.07,0.03)$ m with $\varepsilon_r=1.40$.

- Target 2 at $(0.07,-0.03)$ m with $\varepsilon_r=1.65$.

- 25 dB SNR.

- 1 mm receiver-coordinate standard deviation.

This tests separation, unequal contrast recovery, support overlap, and connected-component count.

### 12.3 heterogeneous_nested

- Host centered at $(0.02,0)$ m, radius $0.08$ m, $\varepsilon_r=1.25$.

- Core centered at $(0.05,0.02)$ m, radius $0.035$ m, $\varepsilon_r=1.70$.

- 25 dB SNR.

- 1 mm receiver-coordinate standard deviation.

This tests whether inversion preserves a high-contrast region inside a lower-contrast object.

## 13. Multi-seed suite processing

run_hardening_suite loops over scenario first and seed second. Each pair is a complete independent Pipeline run. It then converts the result to four compact rows: DAS, Born, DBIM, and CSI.

Arrays are kept only for the first representative seed of each scenario, which is enough for example figures. Every run contributes scalar metrics to aggregation. This prevents the machine-readable JSON from becoming a large dump of repeated complex arrays.

The grouping key is:

$$
(\text{scenario},\text{method},\text{metric}).
$$

For each group, the suite stores $n$, mean, and sample standard deviation.

## 14. How to read the verified three-seed result

The completed reference run produced these $\chi$ relative-$L_2$ statistics:

| Scenario | Born | DBIM | CSI |
| --- | ---: | ---: | ---: |
| off_center | $0.536\pm0.003$ | $0.352\pm0.007$ | $0.366\pm0.004$ |
| dual_target | $0.617\pm0.005$ | $0.436\pm0.024$ | $0.364\pm0.011$ |
| heterogeneous_nested | $0.527\pm0.004$ | $0.414\pm0.028$ | $0.371\pm0.004$ |

The nonlinear methods improve image error over Born in every scenario. CSI has the lowest mean image error for the dual and heterogeneous cases; DBIM is slightly lower for the off-centre case.

The dual-target IoU is much lower than the single-object IoU. This is an honest resolution limitation at the current coarse grid, single frequency, regularization, and aperture. The purpose of hardening is to expose this weakness so the next improvement can be measured.

Do not select the “winner” from one metric alone. DBIM often obtains a much smaller full-data residual, while CSI can obtain a lower image error. This difference illustrates ill-posedness and regularization: the map that fits data most tightly is not automatically closest to truth.

## 15. Reporting artifacts

The hardening driver writes:

| File | Meaning |
| --- | --- |
| fig_phase1_hardening_examples.png | truth, DAS, Born, DBIM, CSI for each scenario at representative seed 0 |
| fig_phase1_hardening_statistics.png | mean and sample-standard-deviation bars for image/data metrics |
| phase1_hardening_metrics.json | scenarios, seeds, per-run rows, summaries, and gates |
| phase1_hardening_report.md | human-readable tables and interpretation boundary |

The one-YAML Pipeline driver separately writes the ordinary Phase-1 method and residual report under its chosen output directory.

## 16. Continuous Integration

### 16.1 What CI does

The GitHub Actions workflow runs on pushes and pull requests to main for Python 3.10 and 3.11. It installs the package with development dependencies and runs the full test suite with a noninteractive Matplotlib backend.

On Python 3.11 it also:

1. Validates that the publication notebook is legal JSON.

2. Runs the YAML Pipeline smoke example.

3. Uploads the generated PNG/JSON/Markdown files as a GitHub Actions artifact.

### 16.2 Why the full Monte-Carlo suite is not in every CI matrix job

The full three-scenario, multi-seed suite is a reproducible driver and can be scheduled or run before release. Ordinary CI uses unit tests, one shared integration fixture, and one YAML smoke run. This preserves fast feedback while still checking the complete architecture.

### 16.3 What a green badge means

A green CI badge means the declared environment reproduced the tests and smoke pipeline. It does not mean clinical validation, perfect numerical accuracy, or successful execution on every GPU/FPGA backend.

## 17. Publication notebook

The notebook [phase1_hardening_platform_demo.ipynb](../notebooks/phase1_hardening_platform_demo.ipynb) is intentionally thin. It:

1. Locates the repository.

2. Loads the YAML configuration.

3. Runs Phase1Pipeline.

4. Prints the common scorecard.

5. Displays reporter-generated figures.

6. States the scientific claim boundary.

The notebook contains no copied inversion implementation. Therefore driver, CI, and notebook cannot silently drift into three versions of the physics.

The base package does not require Jupyter. A pytest release test parses notebook JSON and compiles every Python code cell. Users who want interactive execution install Jupyter separately.

## 18. Code map

| File | Responsibility |
| --- | --- |
| mwisim/phantoms/composite.py | inclusion geometry, overlap precedence, clipping checks |
| mwisim/data/synthetic.py | full-wave data, exact SNR noise, geometry mismatch, metadata |
| mwisim/config/yaml_support.py | PyYAML loading plus safe offline subset fallback |
| mwisim/config/yaml_scene.py | schema validation and SceneBuilder |
| mwisim/pipeline.py | concrete orchestration |
| mwisim/evaluation/image_metrics.py | component counting plus existing image metrics |
| mwisim/evaluation/hardening.py | scenarios, repeated runs, row flattening, aggregation |
| mwisim/reporting/hardening.py | representative/statistical figures, JSON, Markdown |
| scripts/run_phase1_pipeline.py | one YAML experiment |
| scripts/run_phase1_hardening.py | three scenarios and several seeds |
| examples/phase1_hardening.yaml | reproducible example configuration |
| notebooks/phase1_hardening_platform_demo.ipynb | publication-facing walkthrough |
| .github/workflows/ci.yml | automated tests and smoke artifact |

## 19. Testing strategy

The engineering testing pyramid is:

### Fast unit base

- Requested off-centre centroid appears within one cell width.

- Dual scene contains both contrast levels and two true support components.

- Nested core overwrites host.

- Clipped and forbidden-overlap scenes raise errors.

- Complex noise exactly reaches requested SNR.

- Same seed reproduces the same realization.

- Mean and sample standard deviation match hand arithmetic.

- Notebook is valid JSON and all code cells compile.

### Integration middle

- SyntheticDataSource separates true and assumed geometry.

- YAML fallback parses nested mappings and list items.

- SceneBuilder works directly and through the registry.

- Pipeline runs all four imaging/inversion outputs and writes reports.

- Two-seed suite returns the expected rows and nonzero variation.

- HardeningReporter writes parseable JSON and nonempty figures/report.

### Existing physics/regression top

- F1 remains Mie-validated.

- F2 remains dense/FFT cross-validated.

- Born and distorted operators retain adjoint tests.

- DBIM and CSI retain nonlinear recovery tests.

- The complete repository suite guards all previous milestones.

## 20. Commands

Run the P1-H tests:

    python -m pytest tests/test_p1h_phantoms_and_data.py tests/test_p1h_yaml_pipeline.py tests/test_p1h_statistics.py tests/test_p1h_notebook.py -q -p no:cacheprovider

Run the whole repository:

    python -m pytest -q -p no:cacheprovider

Run one YAML experiment:

    python scripts/run_phase1_pipeline.py --config examples/phase1_hardening.yaml --output-dir docs/phase1_pipeline_run

Override only the corruption seed:

    python scripts/run_phase1_pipeline.py --config examples/phase1_hardening.yaml --output-dir docs/phase1_pipeline_seed_12 --seed 12

Run the canonical three-seed suite:

    python scripts/run_phase1_hardening.py --seeds 0,1,2 --output-dir docs

Run more seeds before publication:

    python scripts/run_phase1_hardening.py --seeds 0,1,2,3,4,5,6,7,8,9 --output-dir docs/hardening_10seed

## 21. Common misunderstandings

### “30 dB noise means every sample has exactly the same error magnitude.”

No. Individual complex noise samples are random. Only the norm of the complete noise vector is scaled to the requested SNR.

### “A receiver standard deviation of 1 mm means every receiver moves exactly 1 mm.”

No. Each $x$ and $y$ coordinate is drawn from a zero-mean Gaussian distribution with standard deviation 1 mm. The realized RMS is recorded.

### “If d was generated from chi_true, F(chi_true) must exactly equal d.”

Only when generation and scoring use the same geometry and no noise. P1-H deliberately uses true geometry for data and nominal geometry for scoring, then adds noise.

### “The same seed should produce identical results even after changing the scene.”

The same seed reproduces the random number streams, but changing signal, dimensions, or scene changes the scaled noise and forward data. Seed identity does not make different experiments numerically identical.

### “Low component-count error proves both targets are correct.”

No. Two false blobs can still give count two. Combine component count with IoU, SSIM, localization, contrast error, and data residual.

### “YAML parameters are automatically scientifically valid.”

No. Schema validation checks structure and obvious physical impossibilities. It cannot decide whether an SNR, grid spacing, regularization, or target model is appropriate for a scientific claim.

### “CI is the Monte-Carlo result.”

No. CI verifies software reproducibility. The hardening driver performs the canonical multi-seed experiment and can be expanded to more seeds for a paper.

## 22. Remaining limits after P1-H

P1-H is substantially harder than the original centered noiseless problem, but it remains:

- two-dimensional;

- single-frequency;

- based on ideal plane-wave illumination;

- based on circular receiver coverage;

- based on nondispersive piecewise-constant material;

- without antenna transfer functions and mutual coupling;

- without skin/clutter calibration and reference subtraction;

- without measured data;

- without statistical confidence suitable for clinical claims.

The correct next stage is Phase 2 data realism: dispersive complex tissue, a unified measured-data schema, calibration and artifact preprocessing, and reproduction of a public benchmark. New priors or AI should be compared against this hardened classical baseline rather than against an easier private example.

## 23. Final mental model

P1-A/B/C answered:

$$
\text{Can the platform run one complete controlled imaging experiment?}
$$

P1-H answers:

$$
\text{Does the same platform remain reproducible when scene and data assumptions are stressed?}
$$

The answer is now testable rather than rhetorical:

$$
\boxed{
\text{YAML}
\rightarrow
\text{scene}
\rightarrow
\text{corrupted full-wave data}
\rightarrow
\text{four methods}
\rightarrow
\text{common metrics}
\rightarrow
\text{multi-seed statistics}
\rightarrow
\text{CI/notebook/report}
}
$$

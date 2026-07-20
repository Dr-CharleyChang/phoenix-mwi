# P1-H YAML Schema Reference

This document is the exact configuration reference for schema version 1. For the mathematical and physical explanation, see [P1H_Tutorial_Phase1-hardening-and-release-pipeline.md](P1H_Tutorial_Phase1-hardening-and-release-pipeline.md).

## 1. Top-level fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| schema_version | integer | recommended | Must equal 1; omitted defaults to 1 |
| name | string | recommended | Experiment/scenario identifier |
| scene | mapping | yes | Phantom geometry and material |
| acquisition | mapping | required by Pipeline | Frequency, views, receivers, observation radius |
| corruption | mapping | no | Noise, receiver error, seed |
| algorithms | mapping | no | DAS/Born/DBIM/CSI selection and parameters |
| evaluation | mapping | no | Metric support threshold |
| reporting | mapping | no | Optional output directory |

## 2. scene

| Field | Type | Required | Constraint/default |
| --- | --- | --- | --- |
| type | string | no | composite_circles |
| domain_size_m | positive float | yes | Square side length |
| cell_size_m | positive float | yes | Uniform cell side |
| background_eps_r | real/complex | no | Default 1.0 |
| overlap_policy | string | no | last_wins or error; default last_wins |
| inclusions | list | yes | At least one inclusion |

Each inclusion contains:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| label | string | no | Human-readable material name |
| center_m | two-number list | yes | $[x,y]$ center in metres |
| radius_m | positive float | yes | Circle radius in metres |
| eps_r | real/complex | yes | Relative permittivity |

An inclusion must fit completely inside the square domain and must contain at least one grid-cell center.

### Complex value form

~~~yaml
eps_r:
  real: 4.5
  imag: -0.3
~~~

Allowed keys are real and imag only.

## 3. acquisition

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| frequency_hz | positive float | yes | Single Phase-1 frequency |
| n_views | positive integer | yes | Equally spaced plane-wave directions |
| n_receivers | positive integer | yes | Equally spaced nominal receiver-ring positions |
| observation_radius_m | positive float | yes | Nominal receiver-ring radius |

The observation ring should remain outside all inclusions and preferably outside the full grid.

## 4. corruption

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| snr_db | float or null | no | null | Exact finite-vector complex-noise SNR |
| receiver_position_std_m | nonnegative float | no | 0 | Per-coordinate Gaussian position standard deviation |
| seed | integer | no | 0 | Reproducible random realization |

Data are generated at perturbed rx_true coordinates. Algorithms receive nominal rx coordinates.

## 5. algorithms

### Common structure

~~~yaml
algorithms:
  warm_start: true
  imager:
    name: das
    params: {}
  inverters:
    born:
      name: born
      params: {}
    dbim:
      name: dbim
      params: {}
    csi:
      name: csi
      params: {}
~~~

The name is resolved through the Phoenix registry. params are passed to the implementation constructor.

### Built-in DAS parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| power | 2.0 | Exponent in $|b|^{power}$ |
| sensitivity_correction | true | Divide by Born-column norm |
| normalize | true | Scale peak intensity to 1 |

### Built-in Born parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| mu | 0.01 | Tikhonov weight |
| iter_lim | 200 | Maximum least-squares iterations |
| solver | lsmr | lsmr or lsqr |

### Built-in DBIM parameters

| Parameter | Typical P1-H value | Meaning |
| --- | ---: | --- |
| mu | 0.02 | Update Tikhonov weight |
| max_outer | 8 | Nonlinear outer iterations |
| inner_iter | 120 | LSMR iterations per update |
| step | 0.8 | Damped update factor |
| tol | 0.001 | Data-residual stop target |
| distorted | true | DBIM when true, BIM when false |

### Built-in CSI parameters

| Parameter | Typical P1-H value | Meaning |
| --- | ---: | --- |
| mu_chi | 0.02 | Contrast regularization |
| mu_w | 0.002 | Contrast-source regularization |
| xi | 1.0 | State-equation weight |
| max_outer | 8 | Alternating iterations |
| step | 0.8 | Contrast damping |
| tol | 0.001 | Internal source-data stop target |
| project_real | true | Project contrast to nonnegative real values |

When warm_start is true, DBIM and CSI receive the already computed Born estimate. Their total benchmark runtime includes the shared Born time.

## 6. evaluation

| Field | Type | Default | Meaning |
| --- | --- | ---: | --- |
| support_threshold | float in $(0,1]$ | 0.5 | Fraction of each map maximum used for IoU, centroid, and component count |

Use a lower threshold when a legitimate low-contrast host must remain part of heterogeneous truth support. State the chosen value in reports.

## 7. reporting

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| output_dir | string or null | null | Automatic BenchmarkReporter destination |

A relative path in YAML is resolved relative to the YAML file. A path explicitly supplied to Phase1Pipeline.run or the CLI is resolved relative to the current working directory.

## 8. Complete example

The maintained executable example is [phase1_hardening.yaml](../examples/phase1_hardening.yaml). Run it with:

    python scripts/run_phase1_pipeline.py --config examples/phase1_hardening.yaml --output-dir docs/phase1_pipeline_run

## 9. Offline parser subset

When PyYAML is unavailable, the built-in safe fallback supports the features used by this schema:

- indentation-based mappings;

- dash-based lists;

- inline literal lists such as $[0.02,0.00]$;

- strings, integers, floating-point/scientific notation;

- true, false, null, none, and tilde null;

- single- and double-quoted literals;

- comments beginning with # outside quotes;

- optional YAML document marker.

It rejects tabs, executable expressions, unsupported inline syntax, malformed indentation, and a non-mapping root. Install PyYAML for general YAML features.

## 10. Minimal valid Pipeline configuration

~~~yaml
schema_version: 1
name: minimal
scene:
  domain_size_m: 0.36
  cell_size_m: 0.04
  inclusions:
    - center_m: [0.04, 0.00]
      radius_m: 0.06
      eps_r: 1.5
acquisition:
  frequency_hz: 1.0e9
  n_views: 8
  n_receivers: 20
  observation_radius_m: 0.30
~~~

Omitted corruption gives noiseless data and exact receiver geometry. Omitted algorithms use the built-in DAS/Born/DBIM/CSI defaults. Omitted evaluation uses support threshold 0.5. Omitted reporting writes no files unless output_dir is passed to Pipeline.run.

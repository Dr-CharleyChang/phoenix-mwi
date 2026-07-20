---
title: "Phase 2 Tutorial — Measured microwave data from zero to 100"
project: phoenix-mwi
milestone: P2-A
status: implemented and verified
---

# Phase 2 Tutorial — measured microwave data from zero to 100

## 0. What changed at Phase 2

Phase 1 began with a known target contrast $\chi$, solved the electromagnetic forward problem, generated synthetic receiver data, and asked an imager or inverter to recover the target. Phase 2 begins at the other end: a real vector network analyzer has produced complex measurements, the experiment has metadata and reference scans, and Phoenix must ingest and clean those measurements without silently losing their physical meaning.

The Phase-2A path is:

```mermaid
flowchart LR
    Z["Official Zenodo archive"] --> V["Checksum verification"]
    V --> I["UMBMIDDataSource"]
    I --> M["MeasurementSet: named axes + geometry + metadata"]
    M --> G["ComplexGainCalibrator (optional)"]
    G --> R["ReferenceSubtract"]
    R --> T["Inverse Fourier / ICZT"]
    T --> B["Measured sinogram + JSON benchmark"]
```

This milestone is deliberately about the measured-data foundation. It proves that Phoenix can preserve the real data, match references correctly, reproduce the public UM-BMID preprocessing/ICZT example, and leave a machine-readable provenance trail. It does not yet prove tumor detection, dielectric-property recovery, or clinical performance.

## 1. Start from school mathematics: what is a measurement table?

Imagine measuring temperature at three times. The three numbers alone are incomplete unless we also know that the axis means time and that its coordinates are 08:00, 12:00, and 16:00. Microwave data have the same issue, but with more axes.

UM-BMID Gen-One records one complex reflection coefficient $S_{11}$ at many frequencies and antenna angles. After all scans are consolidated, the array has shape

$$
(N_s,N_f,N_a)=(323,1001,72),
$$

where $N_s$ is the number of scans, $N_f$ is the number of frequencies, and $N_a$ is the number of clockwise antenna positions. One entry is

$$
Y[s,f,a]\in\mathbb C.
$$

The value is complex because a VNA measures both magnitude and phase. We can write

$$
Y=Y_{\mathrm{re}}+jY_{\mathrm{im}}=|Y|e^{j\phi}.
$$

For example, if $Y=3+4j$, then $|Y|=5$ and $\phi=\tan^{-1}(4/3)\approx53.1^\circ$. Replacing this value by $|Y|=5$ destroys its phase, and coherent imaging depends on that phase. Phoenix therefore keeps complex data complex through ingestion, calibration, subtraction, and time transformation.

## 2. Why a plain NumPy array is not enough

Suppose an array has shape `(323, 1001, 72)`. A program cannot infer whether 1001 means frequencies, transmitters, time samples, or pixels. Worse, an array of length 72 used as a gain correction could accidentally be broadcast along the scan axis of another dataset that also happens to have 72 scans. NumPy would calculate a result without reporting the conceptual mistake.

`MeasurementSet` prevents this ambiguity by carrying six connected pieces:

| Part | Question it answers | UM-BMID example |
| --- | --- | --- |
| `values` | What numerical tensor was measured? | complex $S_{11}$, shape `(323,1001,72)` |
| `dims` | What does each axis mean and in what order? | `("scan","frequency","angle")` |
| `coords` | Which physical/sample coordinate belongs to each axis entry? | scan IDs, frequency in Hz, angle in rad |
| `geometry` | Where were antennas or other auxiliary objects? | antenna position `(scan,angle,xyz)` in m |
| `metadata` | What does each scan describe and which reference belongs to it? | phantom, tumor, empty-reference ID, antenna radius |
| `attrs` and `history` | Where did the record come from and what happened to it? | DOI, license, ingestion, selection, subtraction |

The central invariant is simple: every array axis has a name, every named dimension has a one-dimensional coordinate, and any geometry array declares its own dimensions. If `values.shape[1] == 1001`, then `dims[1] == "frequency"` and `coords["frequency"].size == 1001` must all agree.

### 2.1 A tiny complete example

```python
import numpy as np
from mwisim.data import AuxiliaryArray, MeasurementSet

record = MeasurementSet(
    values=np.zeros((2, 3, 4), dtype=complex),
    dims=("scan", "frequency", "angle"),
    coords={
        "scan": np.array([101, 102]),
        "frequency": np.array([1e9, 1.5e9, 2e9]),
        "angle": np.deg2rad([0, 90, 180, 270]),
        "xyz": np.array(["x", "y", "z"]),
    },
    geometry={
        "antenna_position": AuxiliaryArray(
            np.zeros((2, 4, 3)),
            dims=("scan", "angle", "xyz"),
            unit="m",
        )
    },
    metadata=[
        {"sample_id": 101, "empty_reference_id": 102},
        {"sample_id": 102, "empty_reference_id": None},
    ],
    attrs={"dataset": "tiny demonstration"},
)
```

The arrays are copied and marked read-only when the record is constructed. A processing step returns a new record through `record.evolve(...)`; it does not alter the old record in place. This is important in research because a result should not change merely because another function later reused the same input object.

### 2.2 Selection must keep everything aligned

```python
chosen = record.select("scan", [1, 0])
```

This reorders `values`, `coords["scan"]`, scan-dependent geometry, and per-scan metadata together. Hand-slicing only `values[[1,0]]` would leave the original metadata order and silently attach the wrong phantom/reference information to each row.

### 2.3 Native storage is pickle-free

```python
from mwisim.data import save_measurement_npz, load_measurement_npz

save_measurement_npz(record, "measurement.npz")
rebuilt = load_measurement_npz("measurement.npz")
```

Phoenix stores numeric arrays plus a JSON manifest in compressed NPZ and always loads it with `allow_pickle=False`. This preserves dimensions, coordinates, geometry, metadata, attributes, and history without allowing arbitrary Python objects to execute during loading.

## 3. From an official archive to `MeasurementSet`

### 3.1 Why UM-BMID Gen-One was selected

The [University of Manitoba Breast Microwave Imaging Dataset](https://github.com/UManitoba-BMS/UM-BMID) is public, breast-imaging-specific, experimentally measured, documented, and accompanied by official preprocessing code. Its Gen-One system measured monostatic $S_{11}$ from 1 to 8 GHz at 1001 linearly spaced frequencies and 72 clockwise rotational positions. The first antenna position is documented as $-102.5^\circ$ in the dataset coordinate system.

The public Gen-One Zenodo archive used here is pinned to DOI [10.5281/zenodo.5120981](https://zenodo.org/records/5120981), size 350,526,155 bytes, and MD5 `4ac179a5b9fb2ec072adc6d2a7ac8ad3`. The dataset is CC-BY-4.0; the official companion software uses Apache-2.0.

### 3.2 Why pickle receives special treatment

A NumPy or JSON parser reads data. Python `pickle.load`, by design, may reconstruct arbitrary Python objects and can execute code. Therefore a file extension such as `.pickle` is a security boundary, not merely a format choice.

The official Gen-One archive contains `fd_data_gen_one_s11.pickle` and `metadata_gen_one.pickle`. Phoenix follows this sequence:

1. Download only from the pinned Zenodo record.
2. Verify both byte count and MD5 before extraction.
3. Reject ZIP absolute paths, `..` traversal, and symbolic links.
4. Refuse pickle loading unless the caller explicitly passes `trusted_pickle=True`.
5. Normalize the result into `MeasurementSet` and optionally save a pickle-free NPZ for future runs.

Passing `trusted_pickle=True` is not a magic safety filter. It records that the caller has made the trust decision; the checksum and source provenance are what support that decision for this benchmark.

### 3.3 Metadata normalization

Public datasets evolve, so semantically identical fields sometimes have different names. Phoenix preserves every source field and also creates stable canonical fields:

| Source field | Phoenix field | Meaning |
| --- | --- | --- |
| `id` | `sample_id` | unique scan ID |
| `emp_ref_id` or `empty_ref_id` | `empty_reference_id` | linked empty-chamber scan |
| `adi_ref_id` | `adipose_reference_id` | linked adipose-only scan |
| `fib_ref_id` | `healthy_reference_id` | linked adipose/fibroglandular scan |
| `tum_rad` in cm | `tumor_radius_m` | tumor radius in SI metres |
| `ant_rad` in cm | `antenna_radius_m` | metadata antenna radius in SI metres |
| `ant_z` or `ant_height` in cm | `antenna_z_m` | antenna height in SI metres |

This “preserve raw + add canonical” policy avoids two opposite failures: deleting evidence from the source and forcing all downstream code to know every historical spelling.

### 3.4 Geometry construction and its limitation

For scan $s$ and acquisition angle $\theta_a$, the geometry helper constructs

$$
x_{s,a}=r_s\cos\theta_a,\qquad y_{s,a}=r_s\sin\theta_a,\qquad z_{s,a}=z_s.
$$

The official metadata describe `ant_rad` as the radial distance to the antenna's SMA connection point, not a rigorously calibrated electromagnetic phase center. Phoenix therefore labels the geometry as a proxy and does not pretend it is a phase-center calibration. That distinction matters before quantitative measured-data inversion.

## 4. Calibration from first principles

### 4.1 The simplest instrument model

Let $x$ be the complex signal that an ideal system would measure. A simple real instrument model is

$$
y=g\,x+b+n,
$$

where $g$ is a multiplicative complex gain, $b$ is an additive background/direct-coupling term, and $n$ is noise. Because $g$ is complex, it changes both magnitude and phase.

No single operation can estimate every part of this model without calibration measurements. Phase 2 therefore provides two explicit, composable operations rather than one vague function called `clean`.

### 4.2 Complex-gain calibration

Suppose a known standard should produce $x_{\mathrm{std}}$, while the instrument measures $y_{\mathrm{std}}$. Ignoring additive error for this calibration step,

$$
g=\frac{y_{\mathrm{std}}}{x_{\mathrm{std}}},\qquad \hat x=\frac{y}{g}.
$$

Worked complex-number example: let the true signal be $x=1+0.5j$ and the system gain be $g=0.8-0.2j$. The measured value is

$$
y=(0.8-0.2j)(1+0.5j)=0.9+0.2j.
$$

Dividing by the known gain recovers the original value:

$$
\frac{0.9+0.2j}{0.8-0.2j}=1+0.5j.
$$

In code:

```python
from mwisim.preprocessing import ComplexGainCalibrator

calibrator = ComplexGainCalibrator(
    gain=frequency_gain,
    gain_dims=("frequency",),
)
gain_corrected = calibrator.apply(record)
```

The named `gain_dims` is crucial. A gain of shape `(1001,)` is declared to vary with frequency, so Phoenix reshapes it to `(1,1001,1)` before division. This is the explicit version of NumPy broadcasting.

The code rejects zero, near-zero, or non-finite gain values because division would create meaningless infinities or amplify noise without bound.

### 4.3 Reference subtraction

Suppose target and reference scans share the same stable system response:

$$
y_{\mathrm{target}}=g(x_{\mathrm{object}}+x_{\mathrm{background}})+b+n_t,
$$

$$
y_{\mathrm{reference}}=g\,x_{\mathrm{background}}+b+n_r.
$$

Subtracting gives

$$
y_{\mathrm{target}}-y_{\mathrm{reference}}=g\,x_{\mathrm{object}}+(n_t-n_r).
$$

The common additive background $b$ and common background response cancel, but the multiplicative gain $g$ remains. Subtraction also combines the noise from two measurements. This explains both why reference subtraction is useful and why it is not a complete calibration theory.

### 4.4 A three-row arithmetic example

Assume the table contains these scalar stand-ins for full frequency-angle arrays:

| Row | Scan ID | Value | Empty-reference ID |
| ---: | ---: | ---: | ---: |
| 0 | 10 | 5 | 20 |
| 1 | 20 | 2 | missing |
| 2 | 30 | 7 | 20 |

The calibrated target values are

$$
y_{10}^{\mathrm{cal}}=5-2=3,\qquad y_{30}^{\mathrm{cal}}=7-2=5.
$$

The important point is that scan 10's reference is found by matching ID 20, not by subtracting the previous or next row. Real metadata can be reordered, filtered, or combined; row-neighbor assumptions are unsafe.

```python
from mwisim.preprocessing import ReferenceSubtract

subtract = ReferenceSubtract(
    reference_key="empty_reference_id",
    missing="drop",
)
calibrated = subtract.apply(record)
```

The `missing` policy is explicit:

- `"drop"` keeps only scans with a usable linked reference; this matches the usual target-only benchmark flow.
- `"keep"` keeps a scan unchanged and records `applied: false`; useful for inspecting reference scans.
- `"raise"` stops immediately; useful when a complete calibrated cohort is required.

Duplicate scan IDs are rejected because a reference ID would otherwise point to two possible rows.

### 4.5 Processing order

```python
from mwisim.preprocessing import (
    ComplexGainCalibrator,
    PreprocessingPipeline,
    ReferenceSubtract,
)

pipeline = PreprocessingPipeline([
    ComplexGainCalibrator(gain, gain_dims=("frequency",)),
    ReferenceSubtract(reference_key="empty_reference_id", missing="drop"),
])
clean = pipeline.apply(record)
```

Pipeline order is a scientific choice, not decoration. If target and reference share exactly the same gain, divide-then-subtract and subtract-then-divide are algebraically equivalent. If gain changes by scan, they are not equivalent. Phoenix records the actual order in `history` rather than hiding it.

## 5. From frequency to time: ICZT from zero

### 5.1 One frequency is a rotating arrow

A complex frequency sample $S(f_k)$ represents the magnitude and phase of a sinusoidal component at frequency $f_k$. At time $t$, that component contributes

$$
S(f_k)e^{+j2\pi f_k t}.
$$

To reconstruct the response at that time, add all frequency components and normalize by their count:

$$
s(t)=\frac{1}{N_f}\sum_{k=0}^{N_f-1}S(f_k)e^{+j2\pi f_k t}.
$$

This is an inverse Fourier sum evaluated at user-chosen time points. Calling it an inverse chirp-Z transform emphasizes that the output time grid need not be the fixed grid returned by a normal IFFT.

### 5.2 Two-frequency school-math example

Take frequencies $f_0=1$ Hz and $f_1=2$ Hz with samples $S(f_0)=S(f_1)=1$. At $t=0$,

$$
s(0)=\frac{1}{2}(1+1)=1.
$$

At $t=0.5$ s,

$$
s(0.5)=\frac{1}{2}\left(e^{j\pi}+e^{j2\pi}\right)=\frac{1}{2}(-1+1)=0.
$$

The two rotating arrows cancel. This constructive/destructive addition is the basis of coherent time-domain and image focusing.

### 5.3 Why the official algorithm has a separate phase-compensation factor

UM-BMID frequencies are uniformly spaced:

$$
f_k=f_0+k\Delta f.
$$

Substitute this into the inverse sum:

$$
s(t)=\frac{1}{N_f}\sum_k S(f_k)e^{j2\pi(f_0+k\Delta f)t}
=e^{j2\pi f_0t}\frac{1}{N_f}\sum_k S(f_k)e^{j2\pi k\Delta f t}.
$$

The official code computes the second factor using zero-based frequency bins and then multiplies by $e^{j2\pi f_0t}$, which it calls phase compensation. Phoenix computes the direct first expression in `frequency_to_time` and independently computes the factored expression in `um_bmid_iczt_reference`. Their agreement checks the sign, normalization, starting-frequency phase, and array axis.

### 5.4 Pseudocode

```text
input: calibrated FD data S[f, angle]
input: physical frequencies f[f]
input: requested times t[time]

for every requested time m:
    for every antenna angle a:
        total = 0
        for every frequency k:
            total += S[k, a] * exp(+j * 2*pi * f[k] * t[m])
        time_data[m, a] = total / number_of_frequencies

return time_data
```

The implementation replaces the three Python loops with one matrix multiplication, but the mathematics is exactly this pseudocode.

## 6. The public benchmark reproduced

The driver `scripts/run_p2_um_bmid.py` reproduces the official UM-BMID simple-data workflow on the checksum-pinned Gen-One archive:

1. Verify `gen-one.zip` against the Zenodo byte count and MD5.
2. Load all 323 scans, each with 1001 frequencies and 72 angles.
3. Select public sample ID 1.
4. Read its metadata-linked empty-reference ID 12.
5. Compute sample 1 minus scan 12.
6. Evaluate the inverse Fourier/ICZT response from 0 to 6 ns at 1024 points.
7. Compare reference subtraction against the literal target-minus-reference equation.
8. Compare the direct inverse Fourier implementation against the independent phase-compensated UM-BMID formulation.
9. Save a sinogram and JSON evidence.

Verified results on the current archive:

| Gate | Result | Threshold | Status |
| --- | ---: | ---: | --- |
| Archive size | 350,526,155 bytes | exact | pass |
| Archive MD5 | `4ac179a5b9fb2ec072adc6d2a7ac8ad3` | exact | pass |
| Reference-subtraction relative $L_2$ | 0 | $\le10^{-14}$ | pass |
| Reference-subtraction maximum absolute error | 0 | $\le10^{-14}$ | pass |
| ICZT-reference relative $L_2$ | $1.6253\times10^{-15}$ | $\le10^{-11}$ | pass |
| ICZT-reference maximum absolute error | $8.4897\times10^{-17}$ | $\le10^{-11}$ | pass |

The strongest legitimate statement is: **Phoenix reproduces the documented UM-BMID reference-subtraction and ICZT measured-data workflow to floating-point precision.** This benchmark does not contain a reconstructed $\chi$ map, and it must not be described as a tumor-localization or clinical-validation result.

![UM-BMID Gen-One sample 1 sinogram](phase2_um_bmid/sinogram.png)

## 7. Run it yourself

Install the project:

```powershell
cd C:\Projects\Project_Pheonix\mwi
python -m pip install -e ".[dev]"
```

Download, verify, safely extract, ingest, subtract, transform, and report in one command:

```powershell
python scripts\run_p2_um_bmid.py --download --sample-id 1
```

If the verified archive is already in `data/external/um_bmid/gen-one.zip`, omit `--download`:

```powershell
python scripts\run_p2_um_bmid.py --sample-id 1
```

Outputs are:

- `docs/phase2_um_bmid/benchmark.json` — dimensions, IDs, checksums, numerical gates, and record fingerprint.
- `docs/phase2_um_bmid/sinogram.png` — magnitude of the 0–6 ns coherent response versus clockwise antenna position.

The large archive and extracted pickle files are intentionally excluded by `.gitignore`. The small figure and JSON evidence are versioned.

## 8. Code map: where every operation lives

| File | Responsibility |
| --- | --- |
| `mwisim/data/schema.py` | `MeasurementSet`, `AuxiliaryArray`, validation, selection, fingerprint, safe NPZ |
| `mwisim/data/um_bmid.py` | checksum/download, safe ZIP, pickle trust gate, MAT/raw/pickle ingestion, metadata/SI normalization, geometry |
| `mwisim/preprocessing/calibration.py` | named complex-gain calibration and metadata-linked reference subtraction |
| `mwisim/preprocessing/pipeline.py` | ordered composition of `Preprocessor` stages |
| `mwisim/evaluation/measured.py` | inverse Fourier/ICZT, independent reference formulation, benchmark metrics |
| `scripts/run_p2_um_bmid.py` | one-command real-data benchmark and artifact generation |
| `tests/test_p2_*.py` | 16 fast tests covering the new contracts and failure modes |

## 9. What the tests prove

The Phase-2 tests follow a pyramid:

- Arithmetic unit tests check complex gain, target-minus-reference values, relative $L_2$, and the inverse Fourier sum against explicit scalar loops.
- Schema tests reject missing/misaligned axes, non-increasing frequencies, invalid geometry dimensions, and mutation of numerical arrays.
- Security tests prove that pickle is rejected by default and ZIP path traversal is rejected.
- Format tests ingest synthetic MAT and raw real/imaginary text without needing the large public archive in CI.
- Integration tests pass a complete `MeasurementSet` through gain calibration, reference subtraction, provenance, and benchmark reporting.
- The opt-in driver performs the large-data system test against the checksum-pinned public archive.

The public archive is not downloaded in normal CI because a 350 MB network dependency would make every pull request slow and fragile. CI tests the exact contracts with small fixtures; the committed benchmark JSON and figure record the opt-in real-data run.

## 10. Common mistakes and how Phoenix prevents them

### Mistake 1: “The third dimension is probably angle.”

Never guess from shape. Read `dims`, then obtain the integer axis through `record.axis("angle")`.

### Mistake 2: subtracting row $i-1$ as the reference

References are linked by metadata ID. `ReferenceSubtract` builds an ID-to-row map and rejects duplicates.

### Mistake 3: taking magnitude before subtraction

In general, $|a-b|\ne|a|-|b|$. Subtract the complex measurements first; take magnitude only for display or a specifically defined metric.

### Mistake 4: using `np.load(..., allow_pickle=True)` casually

That silently reopens code execution. Native Phoenix NPZ always uses `allow_pickle=False`; public pickle ingestion has a separate explicit trust gate.

### Mistake 5: calling reference subtraction “full calibration”

Subtraction removes components that are stable and common between target and reference. It does not automatically correct time drift, antenna phase-center error, scan-dependent gain, coupling changes caused by the phantom, or model mismatch.

### Mistake 6: treating the sinogram as a $\chi$ image

A time-angle sinogram is preprocessed measurement evidence. Reconstructing spatial dielectric contrast requires an imaging or inversion model plus a justified propagation/calibration model.

## 11. What comes next

Phase 2A has built the trustworthy measured-data entrance. The next scientifically meaningful slice is Phase 2B: measured qualitative imaging and artifact suppression. It should add a UM-BMID-compatible monostatic DAS/ORR baseline, quantify localization against phantom metadata, and introduce one artifact-removal method with ablations against plain empty-reference subtraction. Only after that anchor should Phoenix attempt measured quantitative DBIM/CSI, because those methods need antenna/background/3-D and dispersive-model assumptions that the present 2-D synthetic operator does not yet satisfy.

The immediate research question is no longer “Can Python open the file?” It is “Which physically justified preprocessing and propagation model makes a measured target response focus at the documented target location, and how robust is that conclusion across scans?”

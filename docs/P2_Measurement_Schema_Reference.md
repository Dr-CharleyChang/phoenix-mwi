# Phase 2 measurement schema reference

## 1. Canonical object

`mwisim.data.MeasurementSet` is the canonical measured frequency-domain record. It is a frozen dataclass with read-only NumPy arrays and append-only processing history through `evolve`.

| Field | Type | Required rule |
| --- | --- | --- |
| `values` | numeric `np.ndarray` | rank equals `len(dims)`; object dtype forbidden |
| `dims` | tuple of names | unique valid identifiers; must include `frequency` |
| `coords` | name → 1-D array | one coordinate for every data dimension; lengths match axes |
| `geometry` | name → `AuxiliaryArray` | every auxiliary dimension exists in `coords`; lengths match |
| `metadata` | sequence of mappings | if non-empty and `scan` exists, exactly one record per scan |
| `attrs` | mapping | JSON-safe dataset-level information |
| `history` | sequence of mappings | ordered JSON-safe transformations |
| `schema_version` | integer | currently `1` |

Frequency coordinates must be finite, positive, and strictly increasing.

## 2. Recommended dimension names

| Name | Meaning | Recommended coordinate unit |
| --- | --- | --- |
| `scan` | unique experiment/sample record | stable dataset ID, not row number |
| `frequency` | frequency-domain sample | Hz |
| `angle` | acquisition angle in mathematical CCW-positive coordinates | rad |
| `transmitter` | transmitter element/view | stable ID or rad when itself angular |
| `receiver` | receiver element | stable ID |
| `time` | time-domain sample | s |
| `x`, `y`, `z` | grid axes | m |
| `xyz` | Cartesian component label | strings `x,y,z` |

Do not encode units into numerical values implicitly. Record coordinate units in `attrs` and geometry units in `AuxiliaryArray.unit`.

## 3. UM-BMID Gen-One profile

```text
values.shape = (scan=323, frequency=1001, angle=72)
dims         = ("scan", "frequency", "angle")
frequency    = linspace(1e9, 8e9, 1001) Hz
angle        = deg2rad(-102.5 - arange(72)*5) rad
geometry["antenna_position"].dims = ("scan", "angle", "xyz")
```

Required normalized metadata are `sample_id`, `empty_reference_id`, `adipose_reference_id`, and `healthy_reference_id`; a missing reference is represented by `None`. SI aliases such as `tumor_radius_m` and `antenna_radius_m` are added while the original dataset fields remain present.

## 4. Core operations

```python
frequency_axis = record.axis("frequency")
subset = record.select("scan", [0, 4, 9])
changed = record.evolve(values=new_values, operation="my_step", parameters={"alpha": 2})
summary = record.summary()
fingerprint = record.fingerprint()
```

`select` preserves the selected dimension even for one item, so selecting one scan returns shape `(1,Nf,Na)`, not `(Nf,Na)`. This avoids implicit rank changes in a pipeline.

## 5. Native archive

```python
from mwisim.data import save_measurement_npz, load_measurement_npz

save_measurement_npz(record, "record.npz")
record = load_measurement_npz("record.npz")
```

The archive contains numeric arrays and `__manifest_json__`. Loading always uses `allow_pickle=False`. Files with a schema version other than the currently supported version are rejected rather than guessed.

## 6. Preprocessor contracts

Every measured-data `Preprocessor.apply` accepts and returns a `MeasurementSet`. It must keep dimensions, coordinates, geometry, and metadata aligned and append a history entry that names the operation and parameters.

`ComplexGainCalibrator(gain, gain_dims=...)` divides by a scalar or named-axis complex gain. Every named gain dimension must occur in `record.dims`, and its length must match exactly.

`ReferenceSubtract(reference_key="empty_reference_id", missing="drop")` matches scan IDs by value and calculates target minus reference. `missing` is one of `drop`, `keep`, or `raise`.

`PreprocessingPipeline([stage_1, stage_2, ...])` executes stages in the listed order and type-checks every returned record.

## 7. Compatibility boundary

The Phase-1 `SyntheticDataSource` still returns the inversion-ready legacy problem dictionary used by Born/DBIM/CSI. Measured data return `MeasurementSet`. This is intentional transitional compatibility; forcing real S-parameters into the synthetic plane-wave problem dict would hide incompatible geometry and physics. A future acquisition-model adapter will bridge a measured `MeasurementSet` to a specific imager/inverter only after that model's required metadata are explicit.

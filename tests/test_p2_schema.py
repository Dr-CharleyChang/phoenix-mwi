"""P2 schema tests: named axes, validation, provenance, and safe round-trip."""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.data.schema import (
    AuxiliaryArray,
    MeasurementSchemaError,
    MeasurementSet,
    load_measurement_npz,
    save_measurement_npz,
)


def _record() -> MeasurementSet:
    values = np.arange(24).reshape(3, 4, 2).astype(float) * (1 + 0.5j)
    positions = np.zeros((3, 2, 3))
    return MeasurementSet(
        values=values,
        dims=("scan", "frequency", "angle"),
        coords={
            "scan": np.array([101, 102, 103]),
            "frequency": np.linspace(1e9, 2e9, 4),
            "angle": np.array([0.0, np.pi]),
            "xyz": np.array(["x", "y", "z"]),
        },
        geometry={
            "antenna_position": AuxiliaryArray(
                positions, ("scan", "angle", "xyz"), "m"
            )
        },
        metadata=[{"sample_id": item} for item in (101, 102, 103)],
        attrs={"dataset": "tiny"},
    )


def test_P2S_1_named_axes_are_validated_and_arrays_are_read_only():
    record = _record()
    assert record.axis("frequency") == 1
    assert record.values.shape == (3, 4, 2)
    assert not record.values.flags.writeable
    assert not record.coords["frequency"].flags.writeable
    with pytest.raises(ValueError):
        record.values[0, 0, 0] = 99
    with pytest.raises(TypeError):
        record.attrs["changed"] = True
    with pytest.raises(TypeError):
        record.metadata[0]["sample_id"] = 99
    with pytest.raises(MeasurementSchemaError, match="strictly increasing"):
        MeasurementSet(
            values=np.zeros((2, 2)),
            dims=("frequency", "angle"),
            coords={"frequency": [2e9, 1e9], "angle": [0.0, 1.0]},
        )


def test_P2S_2_selection_keeps_data_metadata_and_geometry_aligned():
    record = _record().select("scan", [2, 0])
    assert record.coords["scan"].tolist() == [103, 101]
    assert [item["sample_id"] for item in record.metadata] == [103, 101]
    assert record.values.shape == (2, 4, 2)
    assert record.geometry["antenna_position"].values.shape == (2, 2, 3)
    assert record.history[-1]["operation"] == "select"


def test_P2S_3_native_npz_round_trip_is_exact_and_pickle_free(tmp_path):
    original = _record().evolve(
        values=_record().values * 2,
        operation="test_scale",
        parameters={"factor": 2},
    )
    path = save_measurement_npz(original, tmp_path / "measurement.npz")
    rebuilt = load_measurement_npz(path)
    assert rebuilt.dims == original.dims
    assert np.array_equal(rebuilt.values, original.values)
    assert np.array_equal(rebuilt.coords["scan"], original.coords["scan"])
    assert np.array_equal(
        rebuilt.geometry["antenna_position"].values,
        original.geometry["antenna_position"].values,
    )
    assert rebuilt.metadata == original.metadata
    assert rebuilt.history == original.history
    assert rebuilt.fingerprint() == original.fingerprint()


def test_P2S_4_auxiliary_geometry_must_name_known_matching_axes():
    with pytest.raises(MeasurementSchemaError, match="unknown dimension"):
        MeasurementSet(
            values=np.zeros((2, 2)),
            dims=("frequency", "angle"),
            coords={"frequency": [1e9, 2e9], "angle": [0.0, 1.0]},
            geometry={
                "bad": AuxiliaryArray(np.zeros((2, 3)), ("angle", "xyz"))
            },
        )

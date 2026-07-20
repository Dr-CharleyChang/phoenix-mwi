"""P2 preprocessing tests: labeled gain removal and matched references."""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.core.registry import available, build
from mwisim.data.schema import MeasurementSchemaError, MeasurementSet
from mwisim.preprocessing import (
    ComplexGainCalibrator,
    PreprocessingPipeline,
    ReferenceSubtract,
    estimate_complex_gain,
)


def _record() -> MeasurementSet:
    rows = np.array([5.0, 2.0, 7.0])[:, None, None]
    values = np.broadcast_to(rows, (3, 4, 2)).astype(complex)
    return MeasurementSet(
        values=values,
        dims=("scan", "frequency", "angle"),
        coords={
            "scan": [10, 20, 30],
            "frequency": np.linspace(1e9, 2e9, 4),
            "angle": [0.0, 1.0],
        },
        metadata=[
            {"sample_id": 10, "empty_reference_id": 20},
            {"sample_id": 20, "empty_reference_id": None},
            {"sample_id": 30, "empty_reference_id": 20},
        ],
    )


def test_P2P_1_complex_gain_calibration_uses_named_broadcasting():
    truth = _record()
    gain = np.array([1 + 1j, 2 - 0.5j, 0.5 + 0.2j, 3 + 0j])
    measured = truth.evolve(values=truth.values * gain[None, :, None])
    calibrated = ComplexGainCalibrator(gain, gain_dims=("frequency",)).apply(
        measured
    )
    assert np.allclose(calibrated.values, truth.values)
    assert calibrated.history[-1]["operation"] == "complex_gain_calibration"
    with pytest.raises(MeasurementSchemaError, match="expected"):
        ComplexGainCalibrator(np.ones(2), gain_dims=("frequency",)).apply(measured)

    two_axis_gain = np.arange(1, 9).reshape(2, 4) + 0.25j
    measured_two_axis = truth.evolve(
        values=truth.values * two_axis_gain.T[None, :, :]
    )
    reordered = ComplexGainCalibrator(
        two_axis_gain, gain_dims=("angle", "frequency")
    ).apply(measured_two_axis)
    assert np.allclose(reordered.values, truth.values)


def test_P2P_2_gain_estimator_recovers_the_known_complex_response():
    expected = np.array([1 + 2j, 2 - 1j])
    gain = np.array([0.8 + 0.1j, 1.2 - 0.3j])
    assert np.allclose(estimate_complex_gain(gain * expected, expected), gain)
    with pytest.raises(ValueError, match="zero"):
        estimate_complex_gain(np.ones(2), np.array([1.0, 0.0]))


def test_P2P_3_reference_ids_are_matched_by_value_not_row_position():
    calibrated = ReferenceSubtract(missing="drop").apply(_record())
    assert calibrated.coords["scan"].tolist() == [10, 30]
    assert np.allclose(calibrated.values[0], 3.0)
    assert np.allclose(calibrated.values[1], 5.0)
    assert calibrated.metadata[0]["reference_subtraction"]["reference_id"] == 20
    assert calibrated.history[-1]["operation"] == "reference_subtraction"


def test_P2P_4_missing_reference_policy_is_explicit():
    kept = ReferenceSubtract(missing="keep").apply(_record())
    assert kept.coords["scan"].tolist() == [10, 20, 30]
    assert np.allclose(kept.values[1], 2.0)
    assert kept.metadata[1]["reference_subtraction"]["applied"] is False
    with pytest.raises(MeasurementSchemaError, match="no usable"):
        ReferenceSubtract(missing="raise").apply(_record())


def test_P2P_5_pipeline_and_registry_compose_stages_in_order():
    gain = 2 - 0.5j
    measured = _record().evolve(values=_record().values * gain)
    pipeline = PreprocessingPipeline(
        [ComplexGainCalibrator(gain), ReferenceSubtract(missing="drop")]
    )
    result = pipeline.apply(measured)
    assert np.allclose(result.values[0], 3.0)
    assert {"pipeline", "complex_gain", "reference_subtract"}.issubset(
        set(available("preprocessor"))
    )
    rebuilt = build("preprocessor", "reference_subtract", missing="drop")
    assert isinstance(rebuilt, ReferenceSubtract)

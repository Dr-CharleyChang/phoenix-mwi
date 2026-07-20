"""P2-B artifact-removal arithmetic and provenance tests."""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.core.registry import available
from mwisim.data.schema import MeasurementSet
from mwisim.preprocessing import AngularMeanSubtract, LowRankClutterFilter


def _record(values: np.ndarray) -> MeasurementSet:
    values = np.asarray(values, dtype=complex)
    return MeasurementSet(
        values=values,
        dims=("scan", "frequency", "angle"),
        coords={
            "scan": np.arange(values.shape[0]),
            "frequency": np.linspace(1e9, 2e9, values.shape[1]),
            "angle": np.arange(values.shape[2]),
        },
        metadata=[{"sample_id": index} for index in range(values.shape[0])],
    )


def test_P2BA_1_angular_mean_subtraction_is_complex_and_per_frequency():
    base = np.array(
        [
            [[1 + 2j, 3 + 4j, 5 + 6j], [2 - 1j, 4 - 3j, 9 + 2j]],
            [[7 + 1j, 8 + 2j, 12 - 1j], [0 + 2j, 3 + 5j, 6 + 8j]],
        ]
    )
    result = AngularMeanSubtract().apply(_record(base))
    assert np.allclose(np.mean(result.values, axis=2), 0.0, atol=1e-15)
    assert np.allclose(
        result.values, base - np.mean(base, axis=2, keepdims=True)
    )
    assert result.history[-1]["operation"] == "angular_mean_subtraction"


def test_P2BA_2_rank_one_svd_filter_removes_a_known_outer_product():
    frequency_mode = np.array([1 + 1j, 2 - 0.5j, -1 + 2j, 0.2 - 0.1j])
    angle_mode = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
    matrix = frequency_mode[:, None] * angle_mode[None, :]
    result = LowRankClutterFilter(rank=1).apply(_record(matrix[None, :, :]))
    assert np.linalg.norm(result.values) < 1e-12 * np.linalg.norm(matrix)
    parameters = result.history[-1]["parameters"]
    assert parameters["rank"] == 1
    assert parameters["removed_energy_fraction_per_scan"][0] == pytest.approx(1.0)


def test_P2BA_3_rank_zero_is_identity_and_invalid_rank_is_rejected():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(2, 4, 3)) + 1j * rng.normal(size=(2, 4, 3))
    record = _record(values)
    result = LowRankClutterFilter(rank=0).apply(record)
    assert np.array_equal(result.values, record.values)
    with pytest.raises(ValueError, match="exceeds"):
        LowRankClutterFilter(rank=4).apply(record)
    assert {"angular_mean", "low_rank"}.issubset(available("preprocessor"))

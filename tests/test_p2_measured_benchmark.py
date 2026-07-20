"""P2 benchmark-transform tests against a deliberately literal scalar oracle."""
from __future__ import annotations

import numpy as np

from mwisim.data.um_bmid import measurement_from_um_bmid_arrays
from mwisim.evaluation.measured import (
    frequency_to_time,
    reproduce_um_bmid_reference_example,
    um_bmid_iczt_reference,
)


def test_P2B_1_frequency_to_time_matches_the_inverse_fourier_definition():
    frequencies = np.linspace(1e9, 1.3e9, 4)
    times = np.linspace(0.0, 3e-9, 7)
    data = np.array(
        [[1 + 2j, 2 - 1j], [3 + 0.5j, 0.2j], [1 - 1j, 4], [2, -2j]]
    )
    actual = frequency_to_time(data, frequencies, times, axis=0)
    expected = np.empty((times.size, data.shape[1]), dtype=complex)
    for time_index, time_s in enumerate(times):
        for angle_index in range(data.shape[1]):
            total = 0j
            for frequency_index, frequency_hz in enumerate(frequencies):
                total += data[frequency_index, angle_index] * np.exp(
                    2j * np.pi * frequency_hz * time_s
                )
            expected[time_index, angle_index] = total / frequencies.size
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    decomposed = um_bmid_iczt_reference(data, frequencies, times, axis=0)
    assert np.allclose(decomposed, expected, rtol=1e-13, atol=1e-13)


def test_P2B_2_public_workflow_contract_uses_metadata_reference_and_reports_scope():
    frequencies = np.linspace(1e9, 8e9, 5)
    reference = np.ones((5, 3), dtype=complex) * (2 + 0.5j)
    scatter = np.arange(15).reshape(5, 3) * (0.1 + 0.2j)
    values = np.stack([reference + scatter, reference])
    metadata = [
        {"id": 101, "emp_ref_id": 202, "ant_rad": 20.0},
        {"id": 202, "emp_ref_id": np.nan, "ant_rad": 20.0},
    ]
    record = measurement_from_um_bmid_arrays(
        values, metadata, frequency_hz=frequencies
    )
    report, time_data, times = reproduce_um_bmid_reference_example(
        record, sample_id=101, n_time_points=11
    )
    assert report["sample_id"] == 101
    assert report["reference_id"] == 202
    assert report["reference_subtraction_relative_l2"] < 1e-15
    assert report["iczt_reference_relative_l2"] < 1e-12
    assert report["pass"] is True
    assert "not a tumor-detection" in report["benchmark_scope"]
    assert time_data.shape == (11, 3)
    assert times.shape == (11,)

"""P2-B tests for monostatic geometry, DAS, ORR, and image coordinates."""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.core.registry import available, build
from mwisim.data.schema import AuxiliaryArray, MeasurementSet
from mwisim.evaluation.measured_imaging import (
    aggregate_measured_metrics,
    localization_error_m,
    measured_image_metrics,
    peak_location_m,
    signal_to_clutter_ratio_db,
)
from mwisim.imaging.measured import (
    MonostaticImagingOperator,
    ORRImager,
    extract_monostatic_scan,
    measured_das,
    normalized_intensity,
    orr_reconstruct,
    select_frequency_band,
    square_image_grid,
)


def _ring(n_angles: int, radius_m: float = 0.20) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    return np.column_stack(
        (radius_m * np.cos(angles), radius_m * np.sin(angles))
    )


def _record(
    data: np.ndarray,
    frequencies_hz: np.ndarray,
    antennas_m: np.ndarray,
    *,
    tumor_xy_m: tuple[float, float] = (0.0, 0.0),
) -> MeasurementSet:
    data = np.asarray(data)
    antennas_3d = np.column_stack((antennas_m, np.zeros(antennas_m.shape[0])))
    return MeasurementSet(
        values=data[None, :, :],
        dims=("scan", "frequency", "angle"),
        coords={
            "scan": [101],
            "frequency": frequencies_hz,
            "angle": np.linspace(0.0, 2.0 * np.pi, antennas_m.shape[0], endpoint=False),
            "xyz": ["x", "y", "z"],
        },
        geometry={
            "antenna_position": AuxiliaryArray(
                antennas_3d[None, :, :], ("scan", "angle", "xyz"), unit="m"
            )
        },
        metadata=[
            {
                "sample_id": 101,
                "tumor_x_m": tumor_xy_m[0],
                "tumor_y_m": tumor_xy_m[1],
                "tumor_radius_m": 0.008,
            }
        ],
    )


def test_P2BI_1_grid_is_cell_centred_and_schema_extraction_uses_named_axes():
    grid = square_image_grid(4, radius_m=0.04)
    assert grid.shape == (4, 4)
    assert grid.x_m.tolist() == pytest.approx([-0.03, -0.01, 0.01, 0.03])
    assert grid.extent_m == pytest.approx((-0.04, 0.04, -0.04, 0.04))
    antennas = _ring(6)
    frequencies = np.linspace(1e9, 2e9, 5)
    record = _record(np.ones((5, 6), dtype=complex), frequencies, antennas)
    scan = extract_monostatic_scan(record, radial_offset_m=0.01)
    assert scan.data.shape == (5, 6)
    assert scan.scan_id == 101
    assert np.linalg.norm(scan.antenna_positions_m, axis=1) == pytest.approx(0.21)


def test_P2BI_2_forward_phase_and_adjoint_identity_are_exact():
    frequencies = np.array([1.0e9, 1.3e9, 1.7e9])
    antennas = _ring(5)
    pixels = np.array([[-0.02, 0.01], [0.0, 0.0], [0.025, -0.015]])
    speed = 2.4e8
    operator = MonostaticImagingOperator(
        frequencies,
        antennas,
        pixels,
        propagation_speed_m_s=speed,
        precompute=False,
        frequency_chunk=2,
    )
    point = np.array([1.0, 0.0, 0.0])
    distance = np.linalg.norm(antennas[0] - pixels[0])
    expected = np.exp(-4j * np.pi * frequencies[0] * distance / speed)
    assert operator.matvec(point)[0, 0] == pytest.approx(expected)

    rng = np.random.default_rng(20260719)
    model = rng.normal(size=3) + 1j * rng.normal(size=3)
    data = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    left = np.vdot(operator.matvec(model), data)
    right = np.vdot(model, operator.rmatvec(data))
    assert left == pytest.approx(right, rel=1e-12, abs=1e-12)


def test_P2BI_3_das_is_the_adjoint_and_localizes_a_synthetic_point():
    grid = square_image_grid(12, radius_m=0.06)
    frequencies = np.linspace(1.5e9, 4.5e9, 17)
    antennas = _ring(20)
    speed = 2.5e8
    operator = MonostaticImagingOperator(
        frequencies,
        antennas,
        grid.positions_m,
        propagation_speed_m_s=speed,
        precompute=True,
    )
    target_index = np.ravel_multi_index((8, 3), grid.shape)
    truth = np.zeros(operator.n_pixels)
    truth[target_index] = 1.0
    data = operator.matvec(truth)
    record = _record(
        data,
        frequencies,
        antennas,
        tumor_xy_m=(grid.x_m[3], grid.y_m[8]),
    )
    image, info = measured_das(
        record, grid, propagation_speed_m_s=speed, precompute=False
    )
    assert info["phase_model"].startswith("exp(-1j*4*pi")
    assert peak_location_m(image, grid) == pytest.approx(
        (grid.x_m[3], grid.y_m[8])
    )
    assert localization_error_m(
        image, grid, (grid.x_m[3], grid.y_m[8])
    ) < 1e-15


def test_P2BI_4_orr_decreases_data_misfit_and_localizes_the_point():
    grid = square_image_grid(8, radius_m=0.05)
    frequencies = np.linspace(1.5e9, 5.0e9, 12)
    antennas = _ring(16)
    speed = 2.45e8
    operator = MonostaticImagingOperator(
        frequencies,
        antennas,
        grid.positions_m,
        propagation_speed_m_s=speed,
        precompute=True,
    )
    row, column = 5, 2
    truth = np.zeros(operator.n_pixels)
    truth[np.ravel_multi_index((row, column), grid.shape)] = 1.0
    data = operator.matvec(truth)
    record = _record(
        data,
        frequencies,
        antennas,
        tumor_xy_m=(grid.x_m[column], grid.y_m[row]),
    )
    model, info = orr_reconstruct(
        record,
        grid,
        propagation_speed_m_s=speed,
        max_iterations=45,
        tolerance=1e-7,
        regularization=1e-6,
        precompute=True,
    )
    history = np.asarray(info["objective_history"])
    assert np.all(np.diff(history) <= 1e-12)
    assert history[-1] < 0.05 * history[0]
    image = normalized_intensity(model)
    assert peak_location_m(image, grid) == pytest.approx(
        (grid.x_m[column], grid.y_m[row])
    )


def test_P2BI_5_frequency_selection_registry_and_image_metrics_are_coordinate_checked():
    grid = square_image_grid(5, radius_m=0.05)
    image = np.zeros(grid.shape)
    image[3, 1] = 1.0
    image[0, 4] = 0.1
    truth = (grid.x_m[1], grid.y_m[3])
    metrics = measured_image_metrics(
        image,
        grid,
        tumor_xy_m=truth,
        tumor_radius_m=0.004,
        roi_radius_m=0.07,
        position_tolerance_m=0.005,
    )
    assert metrics["localization_error_m"] == pytest.approx(0.0)
    assert metrics["localized_within_declared_gate"] is True
    assert metrics["signal_to_clutter_db"] == pytest.approx(10.0)
    assert signal_to_clutter_ratio_db(
        np.sqrt(image),
        grid,
        target_xy_m=truth,
        target_radius_m=0.004,
        roi_radius_m=0.07,
        image_is_intensity=False,
    ) == pytest.approx(10.0)
    aggregate = aggregate_measured_metrics([metrics, metrics])
    assert aggregate["localized_fraction"] == 1.0
    assert aggregate["localization_error_sample_std_m"] == 0.0

    frequencies = np.linspace(1e9, 8e9, 15)
    antennas = _ring(4)
    record = _record(np.ones((15, 4)), frequencies, antennas)
    selected = select_frequency_band(
        record, minimum_hz=2e9, maximum_hz=7e9, max_points=5
    )
    assert selected.coords["frequency"].size == 5
    assert selected.coords["frequency"][0] >= 2e9
    assert selected.coords["frequency"][-1] <= 7e9
    assert {"measured_das", "orr"}.issubset(available("imager"))
    assert isinstance(
        build("imager", "orr", propagation_speed_m_s=2.4e8, n_pixels=8),
        ORRImager,
    )

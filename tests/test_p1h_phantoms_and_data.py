"""P1-H unit tests: composite scenes, noise, and geometry mismatch."""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.core.registry import available, build
from mwisim.data.synthetic import (
    SyntheticDataSource,
    achieved_snr_db,
    add_complex_gaussian_noise,
)
from mwisim.evaluation.image_metrics import support_component_count
from mwisim.phantoms.composite import CircularInclusion, CompositeCirclePhantom


def test_P1H_1_off_center_circle_has_the_requested_physical_centroid():
    phantom = CompositeCirclePhantom(
        domain_size=0.36,
        d=0.02,
        inclusions=[CircularInclusion((0.06, -0.04), 0.05, 1.5, "offset")],
    )
    centers, _ = phantom.grid()
    mask = np.abs(phantom.contrast()) > 0
    centroid = np.mean(centers[mask], axis=0)
    assert np.linalg.norm(centroid - np.array([0.06, -0.04])) <= phantom.d


def test_P1H_2_dual_and_nested_heterogeneous_materials_are_represented():
    dual = CompositeCirclePhantom(
        domain_size=0.36,
        d=0.02,
        inclusions=[
            CircularInclusion((-0.06, 0.0), 0.04, 1.4, "left"),
            CircularInclusion((0.06, 0.0), 0.04, 1.7, "right"),
        ],
    )
    values = np.unique(np.round(dual.contrast().real, 12))
    assert {0.0, 0.4, 0.7}.issubset(set(values))
    assert support_component_count(dual.contrast(), threshold_fraction=0.2) == 2

    nested = CompositeCirclePhantom(
        domain_size=0.36,
        d=0.02,
        inclusions=[
            CircularInclusion((0.0, 0.0), 0.08, 1.2, "host"),
            CircularInclusion((0.02, 0.0), 0.03, 1.8, "core"),
        ],
        overlap_policy="last_wins",
    )
    centers, _ = nested.grid()
    core_cell = np.argmin(np.linalg.norm(centers - np.array([0.02, 0.0]), axis=1))
    assert nested.contrast()[core_cell].real == pytest.approx(0.8)


def test_P1H_3_clipped_or_forbidden_overlapping_inclusions_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        CompositeCirclePhantom(
            domain_size=0.2,
            d=0.02,
            inclusions=[CircularInclusion((0.09, 0.0), 0.03, 1.5)],
        )
    with pytest.raises(ValueError, match="overlaps"):
        CompositeCirclePhantom(
            domain_size=0.3,
            d=0.02,
            inclusions=[
                CircularInclusion((0.0, 0.0), 0.06, 1.3),
                CircularInclusion((0.02, 0.0), 0.04, 1.6),
            ],
            overlap_policy="error",
        )


def test_P1H_4_complex_noise_hits_the_requested_snr_and_is_seed_reproducible():
    signal = np.arange(1, 17, dtype=float).astype(complex)
    noisy_a, noise_a, snr_a = add_complex_gaussian_noise(
        signal, 25.0, np.random.default_rng(7)
    )
    noisy_b, noise_b, snr_b = add_complex_gaussian_noise(
        signal, 25.0, np.random.default_rng(7)
    )
    assert np.allclose(noisy_a, noisy_b)
    assert np.allclose(noise_a, noise_b)
    assert snr_a == pytest.approx(25.0, abs=1e-12)
    assert snr_b == pytest.approx(achieved_snr_db(signal, noisy_b), abs=1e-12)


def test_P1H_5_synthetic_source_separates_true_and_assumed_receiver_geometry():
    phantom = CompositeCirclePhantom(
        domain_size=0.32,
        d=0.04,
        inclusions=[CircularInclusion((0.04, -0.02), 0.05, 1.4)],
    )
    source = SyntheticDataSource(
        phantom,
        frequency_hz=1e9,
        n_views=4,
        n_receivers=12,
        observation_radius_m=0.28,
        snr_db=30.0,
        receiver_position_std_m=1e-3,
        seed=11,
    )
    first = source.measurements()
    second = source.measurements()
    assert first is second
    assert first["d"].shape == (4 * 12,)
    assert not np.allclose(first["rx"], first["rx_true"])
    assert first["snr_db_achieved"] == pytest.approx(30.0, abs=1e-10)
    assert np.linalg.norm(first["d"] - first["d_clean"]) > 0
    assert "synthetic" in available("data_source")
    rebuilt = build(
        "data_source",
        "synthetic",
        phantom=phantom,
        frequency_hz=1e9,
        n_views=2,
        n_receivers=8,
        observation_radius_m=0.28,
    )
    assert isinstance(rebuilt, SyntheticDataSource)

"""Reproducible P2-B benchmark orchestration for UM-BMID Gen-One."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..data.schema import MeasurementSet
from ..imaging.measured import (
    ImageGrid2D,
    MonostaticImagingOperator,
    extract_monostatic_scan,
    normalized_intensity,
    orr_reconstruct,
    select_frequency_band,
    square_image_grid,
)
from ..preprocessing import (
    AngularMeanSubtract,
    LowRankClutterFilter,
    ReferenceSubtract,
)
from .measured_imaging import aggregate_measured_metrics, measured_image_metrics


DEFAULT_P2B_CALIBRATION_IDS = (1, 104, 135)
DEFAULT_P2B_EVALUATION_IDS = (13, 25, 36, 37, 117, 136, 147, 269)
DEFAULT_P2B_SPEEDS_M_S = (1.8e8, 2.0e8, 2.2e8, 2.4e8, 2.6e8, 2.8e8)


def scan_id_to_index(record: MeasurementSet) -> dict[Any, int]:
    """Map canonical scan IDs to row indices and reject duplicates."""
    if "scan" not in record.dims:
        raise ValueError("the benchmark requires a scan dimension")
    mapping: dict[Any, int] = {}
    for index, sample_id in enumerate(record.coords["scan"]):
        if isinstance(sample_id, np.generic):
            sample_id = sample_id.item()
        if sample_id in mapping:
            raise ValueError(f"duplicate scan ID {sample_id!r}")
        mapping[sample_id] = index
    return mapping


def select_scan_ids(record: MeasurementSet, sample_ids: Sequence[Any]) -> MeasurementSet:
    """Select scan rows in the requested ID order."""
    mapping = scan_id_to_index(record)
    missing = [sample_id for sample_id in sample_ids if sample_id not in mapping]
    if missing:
        raise KeyError(f"scan IDs are absent from the record: {missing}")
    return record.select("scan", [mapping[sample_id] for sample_id in sample_ids])


def reference_subtracted_targets(
    record: MeasurementSet,
    target_ids: Sequence[Any],
    *,
    reference_key: str,
) -> MeasurementSet:
    """Select targets plus their referenced rows, subtract, and return targets only."""
    mapping = scan_id_to_index(record)
    reference_ids: list[Any] = []
    for target_id in target_ids:
        metadata = record.metadata[mapping[target_id]]
        reference_id = metadata.get(reference_key)
        if reference_id is None:
            raise ValueError(f"scan {target_id!r} has no {reference_key}")
        reference_ids.append(reference_id)
    ordered_ids = list(target_ids)
    for reference_id in reference_ids:
        if reference_id not in ordered_ids:
            ordered_ids.append(reference_id)
    subset = select_scan_ids(record, ordered_ids)
    result = ReferenceSubtract(
        reference_key=reference_key, missing="drop"
    ).apply(subset)
    actual_ids = list(result.coords["scan"])
    if actual_ids != list(target_ids):
        raise RuntimeError(
            f"reference subtraction returned IDs {actual_ids}; expected {list(target_ids)}"
        )
    return result


def artifact_ablation_records(
    record: MeasurementSet,
    target_ids: Sequence[Any],
    *,
    low_rank: int = 1,
) -> dict[str, MeasurementSet]:
    """Build matched reference/artifact-removal variants for the same target scans."""
    empty = reference_subtracted_targets(
        record, target_ids, reference_key="empty_reference_id"
    )
    return {
        "empty_reference": empty,
        "empty_plus_angular_mean": AngularMeanSubtract().apply(empty),
        f"empty_plus_low_rank_{int(low_rank)}": LowRankClutterFilter(
            int(low_rank)
        ).apply(empty),
        "adipose_reference": reference_subtracted_targets(
            record, target_ids, reference_key="adipose_reference_id"
        ),
        "healthy_reference": reference_subtracted_targets(
            record, target_ids, reference_key="healthy_reference_id"
        ),
    }


def approximate_adipose_radius_m(metadata: dict | Any) -> float:
    """Return the published Gen-3-style A1/A2/A3 radius proxy used only for the ROI."""
    phantom_id = str(metadata.get("phant_id", "")).upper()
    sizes = {"A1": 0.05, "A2": 0.06, "A3": 0.07}
    for prefix, radius in sizes.items():
        if phantom_id.startswith(prefix):
            return radius
    return 0.07


def _truth_from_metadata(metadata: dict | Any) -> tuple[tuple[float, float], float]:
    required = ("tumor_x_m", "tumor_y_m", "tumor_radius_m")
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise ValueError(f"tumor metadata is missing fields: {missing}")
    return (
        (float(metadata["tumor_x_m"]), float(metadata["tumor_y_m"])),
        float(metadata["tumor_radius_m"]),
    )


def _das_images_for_records(
    records: dict[str, MeasurementSet],
    grid: ImageGrid2D,
    *,
    propagation_speed_m_s: float,
    radial_offset_m: float,
    frequency_chunk: int,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, str], np.ndarray]]:
    first_record = next(iter(records.values()))
    n_scans = first_record.coords["scan"].size
    rows: list[dict[str, Any]] = []
    images: dict[tuple[Any, str], np.ndarray] = {}
    for scan_index in range(n_scans):
        base_scan = extract_monostatic_scan(
            first_record, scan=scan_index, radial_offset_m=radial_offset_m
        )
        operator = MonostaticImagingOperator(
            base_scan.frequencies_hz,
            base_scan.antenna_positions_m,
            grid.positions_m,
            propagation_speed_m_s=propagation_speed_m_s,
            frequency_chunk=frequency_chunk,
            precompute="auto",
        )
        truth_xy_m, tumor_radius_m = _truth_from_metadata(base_scan.metadata)
        roi_radius_m = approximate_adipose_radius_m(base_scan.metadata) + 0.01
        for method_name, method_record in records.items():
            scan = extract_monostatic_scan(
                method_record, scan=scan_index, radial_offset_m=radial_offset_m
            )
            image = normalized_intensity(
                operator.rmatvec(scan.data).reshape(grid.shape), power=2.0
            )
            metrics = measured_image_metrics(
                image,
                grid,
                tumor_xy_m=truth_xy_m,
                tumor_radius_m=tumor_radius_m,
                roi_radius_m=roi_radius_m,
                position_tolerance_m=0.01,
            )
            row = {
                "sample_id": scan.scan_id,
                "phantom_id": str(scan.metadata.get("phant_id", "")),
                "method": f"das_{method_name}",
                "propagation_speed_m_s": float(propagation_speed_m_s),
                "radial_offset_m": float(radial_offset_m),
                **metrics,
            }
            rows.append(row)
            images[(scan.scan_id, row["method"])] = image
    return rows, images


def speed_sensitivity(
    record: MeasurementSet,
    calibration_ids: Sequence[Any],
    candidate_speeds_m_s: Iterable[float],
    *,
    n_pixels: int = 32,
    image_radius_m: float = 0.09,
    radial_offset_m: float = 0.0,
    frequency_chunk: int = 8,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose one global speed by median calibration localization, with all rows exposed."""
    healthy = reference_subtracted_targets(
        record, calibration_ids, reference_key="healthy_reference_id"
    )
    grid = square_image_grid(n_pixels, radius_m=image_radius_m)
    rows: list[dict[str, Any]] = []
    for speed in candidate_speeds_m_s:
        speed = float(speed)
        method_rows, _ = _das_images_for_records(
            {"healthy_reference": healthy},
            grid,
            propagation_speed_m_s=speed,
            radial_offset_m=radial_offset_m,
            frequency_chunk=frequency_chunk,
        )
        aggregate = aggregate_measured_metrics(method_rows)
        rows.append({"propagation_speed_m_s": speed, **aggregate})
    selected = min(
        rows,
        key=lambda row: (
            row["localization_error_median_m"],
            row["localization_error_mean_m"],
            row["propagation_speed_m_s"],
        ),
    )
    return float(selected["propagation_speed_m_s"]), rows


def run_p2b_benchmark(
    record: MeasurementSet,
    *,
    calibration_ids: Sequence[Any] = DEFAULT_P2B_CALIBRATION_IDS,
    evaluation_ids: Sequence[Any] = DEFAULT_P2B_EVALUATION_IDS,
    candidate_speeds_m_s: Iterable[float] = DEFAULT_P2B_SPEEDS_M_S,
    minimum_frequency_hz: float = 2e9,
    maximum_frequency_hz: float = 8e9,
    max_frequency_points: int = 51,
    n_pixels: int = 36,
    image_radius_m: float = 0.09,
    radial_offset_m: float = 0.0,
    orr_iterations: int = 25,
    orr_regularization: float = 1e-4,
    frequency_chunk: int = 8,
) -> tuple[dict[str, Any], dict[tuple[Any, str], np.ndarray], ImageGrid2D]:
    """Run label-transparent calibration, held-out DAS ablations, and ORR baseline."""
    calibration_ids = tuple(calibration_ids)
    evaluation_ids = tuple(evaluation_ids)
    candidate_speeds_m_s = tuple(float(value) for value in candidate_speeds_m_s)
    overlap = set(calibration_ids).intersection(evaluation_ids)
    if overlap:
        raise ValueError(f"calibration and evaluation IDs overlap: {sorted(overlap)}")
    band = select_frequency_band(
        record,
        minimum_hz=minimum_frequency_hz,
        maximum_hz=maximum_frequency_hz,
        max_points=max_frequency_points,
    )
    selected_speed, speed_rows = speed_sensitivity(
        band,
        calibration_ids,
        candidate_speeds_m_s,
        n_pixels=min(32, n_pixels),
        image_radius_m=image_radius_m,
        radial_offset_m=radial_offset_m,
        frequency_chunk=frequency_chunk,
    )
    records = artifact_ablation_records(band, evaluation_ids)
    grid = square_image_grid(n_pixels, radius_m=image_radius_m)
    rows, images = _das_images_for_records(
        records,
        grid,
        propagation_speed_m_s=selected_speed,
        radial_offset_m=radial_offset_m,
        frequency_chunk=frequency_chunk,
    )

    healthy = records["healthy_reference"]
    for scan_index in range(healthy.coords["scan"].size):
        one_scan = healthy.select("scan", scan_index)
        model, info = orr_reconstruct(
            one_scan,
            grid,
            propagation_speed_m_s=selected_speed,
            radial_offset_m=radial_offset_m,
            regularization=orr_regularization,
            max_iterations=orr_iterations,
            tolerance=1e-3,
            frequency_chunk=frequency_chunk,
            precompute="auto",
        )
        scan = extract_monostatic_scan(one_scan)
        truth_xy_m, tumor_radius_m = _truth_from_metadata(scan.metadata)
        roi_radius_m = approximate_adipose_radius_m(scan.metadata) + 0.01
        image = normalized_intensity(model, power=2.0)
        metrics = measured_image_metrics(
            image,
            grid,
            tumor_xy_m=truth_xy_m,
            tumor_radius_m=tumor_radius_m,
            roi_radius_m=roi_radius_m,
            position_tolerance_m=0.01,
        )
        method = "orr_healthy_reference"
        rows.append(
            {
                "sample_id": scan.scan_id,
                "phantom_id": str(scan.metadata.get("phant_id", "")),
                "method": method,
                "propagation_speed_m_s": selected_speed,
                "radial_offset_m": float(radial_offset_m),
                "orr_iterations": info["iterations"],
                "orr_converged": info["converged"],
                "orr_initial_objective": info["objective_history"][0],
                "orr_final_objective": info["objective_history"][-1],
                "orr_normalized_data_residual": info["normalized_data_residual"],
                **metrics,
            }
        )
        images[(scan.scan_id, method)] = image

    methods = sorted({row["method"] for row in rows})
    aggregate = {
        method: aggregate_measured_metrics(
            [row for row in rows if row["method"] == method]
        )
        for method in methods
    }
    benchmark_method = "das_empty_plus_angular_mean"
    gate = {
        "method": benchmark_method,
        "median_localization_limit_m": 0.03,
        "localized_fraction_minimum": 0.5,
        "median_localization_pass": bool(
            aggregate[benchmark_method]["localization_error_median_m"] <= 0.03
        ),
        "localized_fraction_pass": bool(
            aggregate[benchmark_method]["localized_fraction"] >= 0.5
        ),
    }
    gate["pass"] = bool(
        gate["median_localization_pass"] and gate["localized_fraction_pass"]
    )
    report = {
        "milestone": "P2-B measured radar imaging",
        "scope": (
            "UM-BMID Gen-One breast-phantom radar imaging benchmark; qualitative "
            "reflectivity only, not a clinical or quantitative-permittivity claim"
        ),
        "calibration_protocol": (
            "one global homogeneous speed selected on disjoint calibration scans by "
            "median healthy-reference DAS localization; no per-scan tuning"
        ),
        "calibration_ids": list(calibration_ids),
        "evaluation_ids": list(evaluation_ids),
        "candidate_speeds_m_s": [float(value) for value in candidate_speeds_m_s],
        "selected_propagation_speed_m_s": selected_speed,
        "speed_sensitivity": speed_rows,
        "frequency_selection": {
            "minimum_hz": float(minimum_frequency_hz),
            "maximum_hz": float(maximum_frequency_hz),
            "n_points": int(band.coords["frequency"].size),
            "original_n_points": int(record.coords["frequency"].size),
        },
        "grid": {
            "shape": list(grid.shape),
            "image_radius_m": float(image_radius_m),
            "pixel_area_m2": grid.pixel_area_m2,
        },
        "radial_phase_center_offset_m": float(radial_offset_m),
        "phase_model": "exp(-1j*4*pi*f*distance/speed)",
        "artifact_ablation": [
            "empty_reference",
            "empty_plus_angular_mean",
            "empty_plus_low_rank_1",
            "adipose_reference",
            "healthy_reference",
        ],
        "orr": {
            "reference": "healthy_reference",
            "iterations": int(orr_iterations),
            "regularization": float(orr_regularization),
            "optimizer": "bounded monotone gradient descent with backtracking",
        },
        "per_scan": rows,
        "aggregate": aggregate,
        "acceptance_gate": gate,
        "pass": gate["pass"],
    }
    return report, images, grid


__all__ = [
    "DEFAULT_P2B_CALIBRATION_IDS",
    "DEFAULT_P2B_EVALUATION_IDS",
    "DEFAULT_P2B_SPEEDS_M_S",
    "approximate_adipose_radius_m",
    "artifact_ablation_records",
    "reference_subtracted_targets",
    "run_p2b_benchmark",
    "scan_id_to_index",
    "select_scan_ids",
    "speed_sensitivity",
]

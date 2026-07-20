"""Reproducibility metrics and transforms for measured frequency-domain data."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..data.schema import MeasurementSchemaError, MeasurementSet, json_safe
from ..preprocessing.calibration import ReferenceSubtract


def relative_l2(actual: Any, expected: Any) -> float:
    """Return ||actual - expected||_2 / ||expected||_2 with a zero-safe denominator."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} versus {expected.shape}")
    denominator = np.linalg.norm(expected.ravel())
    numerator = np.linalg.norm((actual - expected).ravel())
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


def frequency_to_time(
    frequency_data: Any,
    frequencies_hz: Any,
    times_s: Any,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Evaluate the inverse Fourier sum on an arbitrary time grid.

    For uniformly spaced UM-BMID frequencies this is the same inverse chirp-Z transform
    used by the dataset's official example: s(t) = mean_k S(f_k) exp(+j 2 pi f_k t).
    The explicit sign is recorded because changing it mirrors/conjugates phase behavior.
    """
    data = np.asarray(frequency_data)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if frequencies.ndim != 1 or times.ndim != 1:
        raise ValueError("frequencies_hz and times_s must be 1-D")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(times)):
        raise ValueError("frequencies_hz and times_s must be finite")
    axis = int(axis) % data.ndim
    if data.shape[axis] != frequencies.size:
        raise ValueError(
            f"frequency axis has {data.shape[axis]} points; "
            f"frequencies_hz has {frequencies.size}"
        )
    if frequencies.size < 2 or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies_hz must contain at least two increasing points")
    moved = np.moveaxis(data, axis, 0)
    flattened = moved.reshape(frequencies.size, -1)
    inverse_kernel = np.exp(2j * np.pi * np.outer(times, frequencies))
    transformed = (inverse_kernel @ flattened) / frequencies.size
    transformed = transformed.reshape((times.size,) + moved.shape[1:])
    return np.moveaxis(transformed, 0, axis)


def um_bmid_iczt_reference(
    frequency_data: Any,
    frequencies_hz: Any,
    times_s: Any,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Independent decomposition of the ICZT used in the UM-BMID example.

    This form first transforms the zero-based frequency bins k*delta_f, then applies
    exp(+j 2 pi f_start t) as a separate phase compensation. It intentionally follows a
    different calculation path from frequency_to_time so the measured benchmark can
    catch a wrong sign, start-frequency phase, normalization, or axis.
    """
    data = np.asarray(frequency_data)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if frequencies.ndim != 1 or times.ndim != 1 or frequencies.size < 2:
        raise ValueError("frequencies_hz and times_s must be 1-D; two frequencies required")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(times)):
        raise ValueError("frequencies_hz and times_s must be finite")
    frequency_step = frequencies[1] - frequencies[0]
    expected = frequencies[0] + frequency_step * np.arange(frequencies.size)
    if frequency_step <= 0 or not np.allclose(
        frequencies, expected, rtol=1e-12, atol=max(1e-9, abs(frequency_step) * 1e-12)
    ):
        raise ValueError("the UM-BMID ICZT reference requires uniform increasing frequencies")
    axis = int(axis) % data.ndim
    if data.shape[axis] != frequencies.size:
        raise ValueError("frequency data axis and frequency coordinate disagree")
    moved = np.moveaxis(data, axis, 0)
    flattened = moved.reshape(frequencies.size, -1)
    bin_phase = np.exp(
        2j
        * np.pi
        * frequency_step
        * np.outer(times, np.arange(frequencies.size))
    )
    unshifted = (bin_phase @ flattened) / frequencies.size
    start_phase = np.exp(2j * np.pi * frequencies[0] * times)
    transformed = start_phase[:, None] * unshifted
    transformed = transformed.reshape((times.size,) + moved.shape[1:])
    return np.moveaxis(transformed, 0, axis)


def _id(value: Any):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return int(value)
    return value


def reproduce_um_bmid_reference_example(
    record: MeasurementSet,
    *,
    sample_id: Any | None = None,
    start_time_s: float = 0.0,
    stop_time_s: float = 6e-9,
    n_time_points: int = 1024,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Reproduce the official Gen-One empty-reference subtraction and ICZT workflow.

    The official example chooses a target, follows its empty-reference metadata ID,
    subtracts that scan, and transforms 1--8 GHz data onto 0--6 ns. This function makes
    the same choices deterministically and reports numerical self-checks.
    """
    if not isinstance(record, MeasurementSet):
        raise TypeError("record must be a MeasurementSet")
    required = {"scan", "frequency", "angle"}
    if not required.issubset(record.dims):
        raise MeasurementSchemaError(
            f"UM-BMID benchmark requires dimensions {sorted(required)}"
        )
    scan_ids = [_id(value) for value in record.coords["scan"]]
    index_by_id = {value: index for index, value in enumerate(scan_ids)}
    if len(index_by_id) != len(scan_ids):
        raise MeasurementSchemaError("benchmark scan IDs must be unique")

    target_index = None
    if sample_id is not None:
        normalized = _id(sample_id)
        if normalized not in index_by_id:
            raise KeyError(f"sample ID {sample_id!r} is absent from the record")
        target_index = index_by_id[normalized]
    else:
        for index, metadata in enumerate(record.metadata):
            ref_id = _id(metadata.get("empty_reference_id"))
            if ref_id is not None and ref_id in index_by_id and ref_id != scan_ids[index]:
                target_index = index
                break
    if target_index is None:
        raise MeasurementSchemaError("no sample has a usable empty reference")
    target_id = scan_ids[target_index]
    reference_id = _id(record.metadata[target_index].get("empty_reference_id"))
    if reference_id not in index_by_id:
        raise MeasurementSchemaError(
            f"sample {target_id!r} points to missing reference {reference_id!r}"
        )
    reference_index = index_by_id[reference_id]

    pair = record.select("scan", [target_index, reference_index])
    calibrated_record = ReferenceSubtract(
        reference_key="empty_reference_id", missing="drop"
    ).apply(pair)
    calibrated = np.take(
        calibrated_record.values,
        0,
        axis=calibrated_record.axis("scan"),
    )
    target = np.take(record.values, target_index, axis=record.axis("scan"))
    reference = np.take(record.values, reference_index, axis=record.axis("scan"))
    direct = target - reference
    fd_relative_l2 = relative_l2(calibrated, direct)
    fd_max_abs = float(np.max(np.abs(calibrated - direct)))

    frequency_axis_after_scan_take = record.axis("frequency") - (
        record.axis("scan") < record.axis("frequency")
    )
    times = np.linspace(float(start_time_s), float(stop_time_s), int(n_time_points))
    time_data = frequency_to_time(
        calibrated,
        record.coords["frequency"],
        times,
        axis=frequency_axis_after_scan_take,
    )
    reference_time_data = um_bmid_iczt_reference(
        calibrated,
        record.coords["frequency"],
        times,
        axis=frequency_axis_after_scan_take,
    )
    iczt_relative_l2 = relative_l2(time_data, reference_time_data)
    iczt_max_abs = float(np.max(np.abs(time_data - reference_time_data)))
    magnitude = np.abs(time_data)
    peak_flat_index = int(np.argmax(magnitude))
    peak_index = np.unravel_index(peak_flat_index, magnitude.shape)
    time_axis_after_scan_take = frequency_axis_after_scan_take
    peak_time_index = peak_index[time_axis_after_scan_take]

    report = {
        "benchmark": "UM-BMID Gen-One official reference-subtraction and ICZT example",
        "benchmark_scope": "workflow reproduction; not a tumor-detection accuracy claim",
        "dataset": record.attrs.get("dataset", "UM-BMID"),
        "doi": record.attrs.get("doi"),
        "generation": record.attrs.get("generation"),
        "s_parameter": record.attrs.get("s_parameter"),
        "sample_id": target_id,
        "reference_id": reference_id,
        "sample_metadata": json_safe(record.metadata[target_index]),
        "reference_metadata": json_safe(record.metadata[reference_index]),
        "frequency_points": int(record.coords["frequency"].size),
        "angle_points": int(record.coords["angle"].size),
        "frequency_start_hz": float(record.coords["frequency"][0]),
        "frequency_stop_hz": float(record.coords["frequency"][-1]),
        "time_start_s": float(times[0]),
        "time_stop_s": float(times[-1]),
        "time_points": int(times.size),
        "reference_subtraction_relative_l2": fd_relative_l2,
        "reference_subtraction_max_abs": fd_max_abs,
        "iczt_reference_relative_l2": iczt_relative_l2,
        "iczt_reference_max_abs": iczt_max_abs,
        "calibrated_frequency_l2_norm": float(np.linalg.norm(calibrated)),
        "time_domain_peak_magnitude": float(magnitude[peak_index]),
        "time_domain_peak_time_s": float(times[peak_time_index]),
        "pass": bool(
            fd_relative_l2 <= 1e-14
            and fd_max_abs <= 1e-14
            and iczt_relative_l2 <= 1e-11
            and iczt_max_abs <= 1e-11
        ),
    }
    return report, time_data, times


__all__ = [
    "frequency_to_time",
    "relative_l2",
    "reproduce_um_bmid_reference_example",
    "um_bmid_iczt_reference",
]

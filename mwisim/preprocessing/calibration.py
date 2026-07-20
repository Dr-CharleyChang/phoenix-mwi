"""Complex-gain calibration and metadata-matched reference subtraction."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..core.interfaces import Preprocessor
from ..core.registry import register
from ..data.schema import MeasurementSchemaError, MeasurementSet, json_safe
from .pipeline import require_measurement_set


def _named_broadcast(
    values: Any,
    dims: Sequence[str],
    record: MeasurementSet,
    *,
    name: str,
) -> np.ndarray:
    """Reshape an array with named axes so NumPy broadcasts it onto record.values."""
    dims = tuple(str(dim) for dim in dims)
    if len(dims) != len(set(dims)):
        raise MeasurementSchemaError(f"{name} dimensions are not unique: {dims}")
    unknown = [dim for dim in dims if dim not in record.dims]
    if unknown:
        raise MeasurementSchemaError(
            f"{name} uses dimensions absent from the data: {unknown}"
        )
    array = np.asarray(values)
    if array.ndim != len(dims):
        raise MeasurementSchemaError(
            f"{name} has ndim={array.ndim}, but {len(dims)} named dimensions were given"
        )
    expected = tuple(record.values.shape[record.axis(dim)] for dim in dims)
    if array.shape != expected:
        raise MeasurementSchemaError(
            f"{name} has shape {array.shape}; expected {expected} for dimensions {dims}"
        )
    ordered_dims = tuple(dim for dim in record.dims if dim in dims)
    if dims:
        array = np.transpose(array, axes=[dims.index(dim) for dim in ordered_dims])
    shape = [1] * record.values.ndim
    for size, dim in zip(array.shape, ordered_dims):
        shape[record.axis(dim)] = size
    return array.reshape(shape)


def estimate_complex_gain(
    measured_standard: Any,
    expected_standard: Any,
    *,
    floor: float = 1e-15,
) -> np.ndarray:
    """Estimate the multiplicative system gain g = measured / expected.

    If a known calibration standard should produce s_true but the instrument reports
    s_measured = g * s_true, dividing later measurements by g removes that response.
    """
    measured = np.asarray(measured_standard)
    expected = np.asarray(expected_standard)
    if measured.shape != expected.shape:
        raise ValueError(
            "measured_standard and expected_standard must have identical shapes"
        )
    floor = float(floor)
    if floor < 0 or np.any(np.abs(expected) <= floor):
        raise ValueError("expected_standard is zero or below the requested floor")
    return measured / expected


@register("preprocessor", "complex_gain")
class ComplexGainCalibrator(Preprocessor):
    """Remove a known multiplicative complex system response.

    gain may be scalar or may vary along named dimensions such as frequency or scan.
    Labeled broadcasting prevents a length-72 angle vector from accidentally being
    interpreted as a length-72 scan vector.
    """

    def __init__(
        self,
        gain: Any,
        *,
        gain_dims: Sequence[str] = (),
        floor: float = 1e-15,
    ):
        self.gain = np.asarray(gain)
        self.gain_dims = tuple(gain_dims)
        self.floor = float(floor)
        if self.floor < 0:
            raise ValueError("floor must be non-negative")

    def apply(self, data, **kwargs) -> MeasurementSet:
        record = require_measurement_set(data)
        gain = _named_broadcast(
            self.gain, self.gain_dims, record, name="complex gain"
        )
        if not np.all(np.isfinite(gain)):
            raise ValueError("complex gain contains non-finite values")
        if np.any(np.abs(gain) <= self.floor):
            raise ValueError("complex gain is zero or below the requested floor")
        return record.evolve(
            values=record.values / gain,
            operation="complex_gain_calibration",
            parameters={
                "gain_dims": list(self.gain_dims),
                "gain_shape": list(self.gain.shape),
                "floor": self.floor,
            },
        )


def _reference_id(value: Any):
    """Normalize IDs while treating None, blank strings, and NaN as missing."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


@register("preprocessor", "reference_subtract")
class ReferenceSubtract(Preprocessor):
    """Subtract the metadata-linked reference scan from each target scan.

    For target scan i with metadata reference ID r(i), the operation is
    y_cal[i, ...] = y[i, ...] - y[r(i), ...]. Reference IDs are matched by value,
    never by assuming that the reference occupies a neighboring array row.
    """

    def __init__(
        self,
        *,
        reference_key: str = "empty_reference_id",
        scan_dim: str = "scan",
        missing: str = "drop",
    ):
        self.reference_key = str(reference_key)
        self.scan_dim = str(scan_dim)
        self.missing = str(missing)
        if self.missing not in {"drop", "keep", "raise"}:
            raise ValueError("missing must be 'drop', 'keep', or 'raise'")

    def apply(self, data, **kwargs) -> MeasurementSet:
        record = require_measurement_set(data)
        if self.scan_dim not in record.dims:
            raise MeasurementSchemaError(
                f"reference subtraction requires data dimension {self.scan_dim!r}"
            )
        if not record.metadata:
            raise MeasurementSchemaError(
                "reference subtraction requires one metadata record per scan"
            )
        scan_ids = [_reference_id(item) for item in record.coords[self.scan_dim]]
        if any(item is None for item in scan_ids):
            raise MeasurementSchemaError("scan coordinates must contain non-missing IDs")
        id_to_index: dict[Any, int] = {}
        for index, sample_id in enumerate(scan_ids):
            if sample_id in id_to_index:
                raise MeasurementSchemaError(f"duplicate scan ID {sample_id!r}")
            id_to_index[sample_id] = index

        scan_axis = record.axis(self.scan_dim)
        output_rows: list[np.ndarray] = []
        output_indices: list[int] = []
        output_metadata: list[Mapping[str, Any]] = []
        missing_ids: list[Any] = []
        for target_index, metadata in enumerate(record.metadata):
            ref_id = _reference_id(metadata.get(self.reference_key))
            ref_index = id_to_index.get(ref_id) if ref_id is not None else None
            if ref_index is None:
                missing_ids.append(scan_ids[target_index])
                if self.missing == "raise":
                    raise MeasurementSchemaError(
                        f"scan {scan_ids[target_index]!r} has no usable "
                        f"{self.reference_key!r} reference"
                    )
                if self.missing == "drop":
                    continue
                calibrated = np.take(record.values, target_index, axis=scan_axis)
                applied = False
            else:
                target = np.take(record.values, target_index, axis=scan_axis)
                reference = np.take(record.values, ref_index, axis=scan_axis)
                calibrated = target - reference
                applied = True
            output_rows.append(calibrated)
            output_indices.append(target_index)
            updated_metadata = dict(metadata)
            updated_metadata["reference_subtraction"] = {
                "reference_key": self.reference_key,
                "reference_id": json_safe(ref_id),
                "applied": applied,
            }
            output_metadata.append(updated_metadata)

        if not output_rows:
            raise MeasurementSchemaError(
                "reference subtraction produced no scans; check metadata and missing policy"
            )
        values = np.stack(output_rows, axis=scan_axis)
        selected = record.select(self.scan_dim, output_indices)
        return selected.evolve(
            values=values,
            metadata=output_metadata,
            operation="reference_subtraction",
            parameters={
                "reference_key": self.reference_key,
                "missing_policy": self.missing,
                "input_scans": len(scan_ids),
                "output_scans": len(output_indices),
                "missing_scan_ids": json_safe(missing_ids),
            },
        )


__all__ = [
    "ComplexGainCalibrator",
    "ReferenceSubtract",
    "estimate_complex_gain",
]

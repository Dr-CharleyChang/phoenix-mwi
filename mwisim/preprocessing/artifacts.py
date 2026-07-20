"""Measured-data artifact-removal stages with explicit named-axis semantics."""
from __future__ import annotations

import numpy as np

from ..core.interfaces import Preprocessor
from ..core.registry import register
from ..data.schema import MeasurementSchemaError, MeasurementSet
from .pipeline import require_measurement_set


def _frequency_angle_stack(record: MeasurementSet) -> tuple[np.ndarray, list[str], bool]:
    required = {"frequency", "angle"}
    missing = required.difference(record.dims)
    if missing:
        raise MeasurementSchemaError(
            f"artifact removal is missing axes: {sorted(missing)}"
        )
    allowed = required | {"scan"}
    extra = set(record.dims).difference(allowed)
    if extra:
        raise MeasurementSchemaError(
            f"artifact removal does not know how to collapse axes: {sorted(extra)}"
        )
    original_dims = list(record.dims)
    has_scan = "scan" in original_dims
    target_dims = (["scan"] if has_scan else []) + ["frequency", "angle"]
    axes = [original_dims.index(dim) for dim in target_dims]
    values = np.transpose(record.values, axes)
    if not has_scan:
        values = values[None, ...]
    return np.asarray(values), original_dims, has_scan


def _restore_stack(
    values: np.ndarray, original_dims: list[str], has_scan: bool
) -> np.ndarray:
    target_dims = (["scan"] if has_scan else []) + ["frequency", "angle"]
    if not has_scan:
        values = values[0]
    axes = [target_dims.index(dim) for dim in original_dims]
    return np.transpose(values, axes)


@register("preprocessor", "angular_mean")
class AngularMeanSubtract(Preprocessor):
    """Remove the complex angular mean independently at each scan and frequency.

    A response that is identical at all antenna positions lies in the rank-one angular
    vector ``[1, 1, ..., 1]``.  Subtracting the mean removes exactly that component.
    This is a transparent baseline, not a claim that all skin/system clutter is constant.
    """

    def apply(self, data, **kwargs) -> MeasurementSet:
        record = require_measurement_set(data)
        angle_axis = record.axis("angle")
        mean = np.mean(record.values, axis=angle_axis, keepdims=True)
        return record.evolve(
            values=record.values - mean,
            operation="angular_mean_subtraction",
            parameters={"dimension": "angle"},
        )


@register("preprocessor", "low_rank")
class LowRankClutterFilter(Preprocessor):
    """Remove leading SVD components from each frequency-by-angle scan matrix."""

    def __init__(self, rank: int = 1):
        self.rank = int(rank)
        if self.rank < 0:
            raise ValueError("rank must be non-negative")

    def apply(self, data, **kwargs) -> MeasurementSet:
        record = require_measurement_set(data)
        values, original_dims, has_scan = _frequency_angle_stack(record)
        maximum_rank = min(values.shape[-2:])
        if self.rank > maximum_rank:
            raise ValueError(
                f"rank={self.rank} exceeds matrix rank bound {maximum_rank}"
            )
        filtered = np.empty_like(values)
        removed_energy = []
        for scan_index, matrix in enumerate(values):
            if self.rank == 0:
                filtered[scan_index] = matrix
                removed_energy.append(0.0)
                continue
            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            clutter = (
                u[:, : self.rank] * singular_values[None, : self.rank]
            ) @ vh[: self.rank]
            filtered[scan_index] = matrix - clutter
            denominator = float(np.linalg.norm(matrix) ** 2)
            removed = float(np.linalg.norm(clutter) ** 2)
            removed_energy.append(removed / denominator if denominator > 0 else 0.0)
        restored = _restore_stack(filtered, original_dims, has_scan)
        return record.evolve(
            values=restored,
            operation="low_rank_clutter_filter",
            parameters={
                "rank": self.rank,
                "matrix_axes": ["frequency", "angle"],
                "removed_energy_fraction_per_scan": removed_energy,
            },
        )


__all__ = ["AngularMeanSubtract", "LowRankClutterFilter"]

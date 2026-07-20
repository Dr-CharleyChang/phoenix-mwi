"""Coordinate-aware metrics for qualitative measured microwave images."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..imaging.measured import ImageGrid2D


def _validated_image(image: np.ndarray, grid: ImageGrid2D) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.shape != grid.shape:
        raise ValueError(f"image has shape {image.shape}; expected {grid.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains non-finite values")
    if np.any(image < 0):
        raise ValueError("image metrics require a nonnegative amplitude or intensity map")
    return image


def circular_mask(
    grid: ImageGrid2D,
    center_m: tuple[float, float],
    radius_m: float,
) -> np.ndarray:
    """Return pixels whose centres lie inside or on a specified circle."""
    center = np.asarray(center_m, dtype=float)
    radius_m = float(radius_m)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_m must contain two finite coordinates")
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("radius_m must be finite and non-negative")
    xx, yy = np.meshgrid(grid.x_m, grid.y_m, indexing="xy")
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius_m**2


def peak_location_m(
    image: np.ndarray,
    grid: ImageGrid2D,
    *,
    roi_radius_m: float | None = None,
    roi_center_m: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Return the coordinate of the highest image value inside an optional ROI."""
    image = _validated_image(image, grid)
    if roi_radius_m is None:
        mask = np.ones(grid.shape, dtype=bool)
    else:
        mask = circular_mask(grid, roi_center_m, roi_radius_m)
    if not np.any(mask):
        raise ValueError("the requested ROI contains no pixel centres")
    candidate = np.where(mask, image, -np.inf)
    row, column = np.unravel_index(int(np.argmax(candidate)), grid.shape)
    return float(grid.x_m[column]), float(grid.y_m[row])


def localization_error_m(
    image: np.ndarray,
    grid: ImageGrid2D,
    truth_xy_m: tuple[float, float],
    *,
    roi_radius_m: float | None = None,
) -> float:
    """Return Euclidean distance from the image maximum to the known target centre."""
    truth = np.asarray(truth_xy_m, dtype=float)
    if truth.shape != (2,) or not np.all(np.isfinite(truth)):
        raise ValueError("truth_xy_m must contain two finite coordinates")
    peak = np.asarray(
        peak_location_m(image, grid, roi_radius_m=roi_radius_m), dtype=float
    )
    return float(np.linalg.norm(peak - truth))


def signal_to_clutter_ratio_db(
    image: np.ndarray,
    grid: ImageGrid2D,
    *,
    target_xy_m: tuple[float, float],
    target_radius_m: float,
    roi_radius_m: float,
    target_margin_m: float = 0.005,
    image_is_intensity: bool = True,
    floor: float = 1e-15,
) -> float:
    """Return max-target to max-clutter ratio in dB within a circular ROI.

    For an intensity map the formula is ``10*log10(I_target/I_clutter)``.  For an
    amplitude map it is ``20*log10(A_target/A_clutter)``; the two are equivalent when
    intensity equals amplitude squared.
    """
    image = _validated_image(image, grid)
    target_radius = float(target_radius_m) + float(target_margin_m)
    target = circular_mask(grid, target_xy_m, target_radius)
    roi = circular_mask(grid, (0.0, 0.0), float(roi_radius_m))
    clutter = roi & ~target
    if not np.any(target & roi):
        raise ValueError("the target region does not intersect the imaging ROI")
    if not np.any(clutter):
        raise ValueError("the clutter region contains no pixels")
    target_peak = max(float(np.max(image[target & roi])), float(floor))
    clutter_peak = max(float(np.max(image[clutter])), float(floor))
    multiplier = 10.0 if image_is_intensity else 20.0
    return float(multiplier * np.log10(target_peak / clutter_peak))


def measured_image_metrics(
    image: np.ndarray,
    grid: ImageGrid2D,
    *,
    tumor_xy_m: tuple[float, float],
    tumor_radius_m: float,
    roi_radius_m: float,
    position_tolerance_m: float = 0.01,
) -> dict[str, Any]:
    """Compute localization and contrast metrics with a declared coordinate gate."""
    peak_x, peak_y = peak_location_m(image, grid, roi_radius_m=roi_radius_m)
    error = localization_error_m(
        image, grid, tumor_xy_m, roi_radius_m=roi_radius_m
    )
    gate = float(tumor_radius_m) + float(position_tolerance_m)
    return {
        "peak_x_m": peak_x,
        "peak_y_m": peak_y,
        "tumor_x_m": float(tumor_xy_m[0]),
        "tumor_y_m": float(tumor_xy_m[1]),
        "tumor_radius_m": float(tumor_radius_m),
        "localization_error_m": error,
        "localization_gate_m": gate,
        "localized_within_declared_gate": bool(error <= gate),
        "signal_to_clutter_db": signal_to_clutter_ratio_db(
            image,
            grid,
            target_xy_m=tumor_xy_m,
            target_radius_m=tumor_radius_m,
            roi_radius_m=roi_radius_m,
            image_is_intensity=True,
        ),
    }


def aggregate_measured_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-scan rows without hiding individual failures."""
    rows = list(rows)
    if not rows:
        raise ValueError("at least one metric row is required")
    localization = np.asarray([row["localization_error_m"] for row in rows], dtype=float)
    scr = np.asarray([row["signal_to_clutter_db"] for row in rows], dtype=float)
    passes = np.asarray(
        [row["localized_within_declared_gate"] for row in rows], dtype=bool
    )
    return {
        "n_scans": len(rows),
        "localization_error_mean_m": float(np.mean(localization)),
        "localization_error_median_m": float(np.median(localization)),
        "localization_error_sample_std_m": float(np.std(localization, ddof=1))
        if len(rows) > 1
        else 0.0,
        "localization_error_min_m": float(np.min(localization)),
        "localization_error_max_m": float(np.max(localization)),
        "signal_to_clutter_mean_db": float(np.mean(scr)),
        "signal_to_clutter_median_db": float(np.median(scr)),
        "localized_fraction": float(np.mean(passes)),
        "localized_count": int(np.sum(passes)),
    }


__all__ = [
    "aggregate_measured_metrics",
    "circular_mask",
    "localization_error_m",
    "measured_image_metrics",
    "peak_location_m",
    "signal_to_clutter_ratio_db",
]

"""Monostatic measured-data DAS and optimization-based radar imaging.

The model in this module is deliberately smaller than the full Maxwell forward model.
For a pixel p, antenna a, and frequency f it assumes one homogeneous propagation
speed and the round-trip phase ``exp(-1j * 4*pi*f*R[a,p]/v)``.  Its adjoint applies
the conjugate phase and is frequency-domain delay-and-sum (DAS).

The optimization-based radar reconstruction (ORR) implementation follows the linear
primary-scatter model described by Reimer and Pistorius (Sensors 2021).  Phoenix adds
bounded iterations, an exact adjoint, deterministic step-size estimation, optional
Tikhonov regularization, and backtracking.  These are numerical safety features; they
do not turn the qualitative radar model into a quantitative permittivity inversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..core.interfaces import Imager
from ..core.registry import register
from ..data.schema import MeasurementSchemaError, MeasurementSet


@dataclass(frozen=True)
class ImageGrid2D:
    """A regular cell-centred Cartesian image grid in metres."""

    x_m: np.ndarray
    y_m: np.ndarray

    def __post_init__(self) -> None:
        x_m = np.array(self.x_m, dtype=float, copy=True)
        y_m = np.array(self.y_m, dtype=float, copy=True)
        for name, values in (("x_m", x_m), ("y_m", y_m)):
            if values.ndim != 1 or values.size < 2:
                raise ValueError(f"{name} must be a 1-D array with at least two cells")
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
                raise ValueError(f"{name} must be finite and strictly increasing")
            steps = np.diff(values)
            if not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-15):
                raise ValueError(f"{name} must be regularly spaced")
            values.setflags(write=False)
        object.__setattr__(self, "x_m", x_m)
        object.__setattr__(self, "y_m", y_m)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the image-array shape ``(n_y, n_x)``."""
        return (self.y_m.size, self.x_m.size)

    @property
    def positions_m(self) -> np.ndarray:
        """Return flattened pixel positions ``(n_pixels, 2)`` in C image order."""
        xx, yy = np.meshgrid(self.x_m, self.y_m, indexing="xy")
        return np.column_stack((xx.ravel(), yy.ravel()))

    @property
    def pixel_area_m2(self) -> float:
        """Return the area represented by one cell."""
        return float((self.x_m[1] - self.x_m[0]) * (self.y_m[1] - self.y_m[0]))

    @property
    def extent_m(self) -> tuple[float, float, float, float]:
        """Return outer cell-edge limits for ``imshow``."""
        dx = float(self.x_m[1] - self.x_m[0])
        dy = float(self.y_m[1] - self.y_m[0])
        return (
            float(self.x_m[0] - dx / 2),
            float(self.x_m[-1] + dx / 2),
            float(self.y_m[0] - dy / 2),
            float(self.y_m[-1] + dy / 2),
        )


def square_image_grid(
    n_pixels: int = 64,
    *,
    radius_m: float = 0.09,
    center_m: tuple[float, float] = (0.0, 0.0),
) -> ImageGrid2D:
    """Build an ``n_pixels`` by ``n_pixels`` cell-centred square grid."""
    n_pixels = int(n_pixels)
    radius_m = float(radius_m)
    center = np.asarray(center_m, dtype=float)
    if n_pixels < 2:
        raise ValueError("n_pixels must be at least two")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("center_m must contain two finite coordinates")
    spacing = 2.0 * radius_m / n_pixels
    offsets = -radius_m + spacing * (0.5 + np.arange(n_pixels))
    return ImageGrid2D(offsets + center[0], offsets + center[1])


@dataclass(frozen=True)
class MonostaticScan:
    """One canonical ``(frequency, angle)`` scan plus its 2-D antenna geometry."""

    data: np.ndarray
    frequencies_hz: np.ndarray
    antenna_positions_m: np.ndarray
    scan_id: Any
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        data = np.array(self.data, dtype=complex, copy=True)
        frequencies = np.array(self.frequencies_hz, dtype=float, copy=True)
        antennas = np.array(self.antenna_positions_m, dtype=float, copy=True)
        if data.ndim != 2:
            raise ValueError("scan data must have shape (frequency, angle)")
        if frequencies.shape != (data.shape[0],):
            raise ValueError("frequency coordinates do not match scan data")
        if antennas.shape != (data.shape[1], 2):
            raise ValueError("antenna positions must have shape (angle, 2)")
        if not np.all(np.isfinite(data)):
            raise ValueError("scan data contains non-finite values")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("frequencies must be finite and positive")
        if not np.all(np.isfinite(antennas)):
            raise ValueError("antenna positions contain non-finite values")
        data.setflags(write=False)
        frequencies.setflags(write=False)
        antennas.setflags(write=False)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "antenna_positions_m", antennas)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _scan_index(record: MeasurementSet, scan: int | None) -> int:
    if "scan" not in record.dims:
        if scan not in (None, 0):
            raise IndexError("scan index was supplied, but the record has no scan axis")
        return 0
    n_scans = record.coords["scan"].size
    index = 0 if scan is None else int(scan)
    if index < 0:
        index += n_scans
    if index < 0 or index >= n_scans:
        raise IndexError(f"scan index {scan} is outside [0, {n_scans})")
    return index


def extract_monostatic_scan(
    record: MeasurementSet,
    *,
    scan: int | None = 0,
    radial_offset_m: float = 0.0,
) -> MonostaticScan:
    """Extract one scan using named axes and apply an explicit radial phase-centre offset."""
    if not isinstance(record, MeasurementSet):
        raise TypeError("record must be a MeasurementSet")
    required = {"frequency", "angle"}
    missing = required.difference(record.dims)
    if missing:
        raise MeasurementSchemaError(f"monostatic imaging is missing axes: {sorted(missing)}")
    extra = set(record.dims).difference(required | {"scan"})
    if extra:
        raise MeasurementSchemaError(
            f"monostatic imaging does not know how to collapse axes: {sorted(extra)}"
        )
    scan_index = _scan_index(record, scan)
    values = record.values
    dims = list(record.dims)
    if "scan" in dims:
        values = np.take(values, scan_index, axis=dims.index("scan"))
        dims.remove("scan")
    values = np.transpose(values, (dims.index("frequency"), dims.index("angle")))

    try:
        geometry = record.geometry["antenna_position"]
    except KeyError:
        raise MeasurementSchemaError(
            "monostatic imaging requires geometry['antenna_position']"
        ) from None
    positions = geometry.values
    geometry_dims = list(geometry.dims)
    if "scan" in geometry_dims:
        positions = np.take(positions, scan_index, axis=geometry_dims.index("scan"))
        geometry_dims.remove("scan")
    if "angle" not in geometry_dims or "xyz" not in geometry_dims:
        raise MeasurementSchemaError(
            "antenna_position must use named dimensions 'angle' and 'xyz'"
        )
    positions = np.transpose(
        positions,
        (geometry_dims.index("angle"), geometry_dims.index("xyz")),
    )
    xyz_labels = [str(item) for item in record.coords["xyz"]]
    try:
        xy_indices = [xyz_labels.index("x"), xyz_labels.index("y")]
    except ValueError:
        raise MeasurementSchemaError("xyz coordinate must contain x and y") from None
    positions = positions[:, xy_indices]

    radial_offset_m = float(radial_offset_m)
    if not np.isfinite(radial_offset_m):
        raise ValueError("radial_offset_m must be finite")
    if radial_offset_m:
        radii = np.linalg.norm(positions, axis=1)
        if np.any(radii <= 0):
            raise ValueError("cannot apply a radial offset to a zero-radius antenna")
        positions = positions + radial_offset_m * positions / radii[:, None]

    scan_id = record.coords["scan"][scan_index] if "scan" in record.dims else None
    metadata = record.metadata[scan_index] if record.metadata else {}
    return MonostaticScan(
        data=values,
        frequencies_hz=record.coords["frequency"],
        antenna_positions_m=positions,
        scan_id=scan_id,
        metadata=metadata,
    )


def select_frequency_band(
    record: MeasurementSet,
    *,
    minimum_hz: float | None = None,
    maximum_hz: float | None = None,
    max_points: int | None = None,
) -> MeasurementSet:
    """Select a frequency interval and, if requested, evenly decimate it."""
    frequencies = np.asarray(record.coords["frequency"], dtype=float)
    keep = np.ones(frequencies.size, dtype=bool)
    if minimum_hz is not None:
        keep &= frequencies >= float(minimum_hz)
    if maximum_hz is not None:
        keep &= frequencies <= float(maximum_hz)
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError("the requested frequency interval contains no samples")
    if max_points is not None:
        max_points = int(max_points)
        if max_points < 2:
            raise ValueError("max_points must be at least two")
        if indices.size > max_points:
            positions = np.linspace(0, indices.size - 1, max_points)
            indices = indices[np.round(positions).astype(int)]
            indices = np.unique(indices)
    return record.select("frequency", indices)


class MonostaticImagingOperator:
    """Matrix-free primary-scatter map and its exact Hermitian adjoint.

    ``matvec(x)`` predicts a complex array with shape ``(frequency, angle)``.
    ``rmatvec(y)`` back-projects that array to a flat pixel vector.  The optional
    ``cell_weight`` represents the quadrature area in the published integral model;
    it defaults to one because P2-B reconstructs a normalized qualitative map.
    """

    def __init__(
        self,
        frequencies_hz: np.ndarray,
        antenna_positions_m: np.ndarray,
        pixel_positions_m: np.ndarray,
        *,
        propagation_speed_m_s: float,
        cell_weight: float = 1.0,
        frequency_chunk: int = 8,
        precompute: bool | str = "auto",
        max_cache_bytes: int = 160_000_000,
    ):
        frequencies = np.asarray(frequencies_hz, dtype=float)
        antennas = np.asarray(antenna_positions_m, dtype=float)
        pixels = np.asarray(pixel_positions_m, dtype=float)
        if frequencies.ndim != 1 or frequencies.size < 1:
            raise ValueError("frequencies_hz must be a non-empty 1-D array")
        if antennas.ndim != 2 or antennas.shape[1] != 2:
            raise ValueError("antenna_positions_m must have shape (n_angles, 2)")
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixel_positions_m must have shape (n_pixels, 2)")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("frequencies_hz must be finite and positive")
        if not np.all(np.isfinite(antennas)) or not np.all(np.isfinite(pixels)):
            raise ValueError("antenna and pixel positions must be finite")
        speed = float(propagation_speed_m_s)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("propagation_speed_m_s must be finite and positive")
        cell_weight = float(cell_weight)
        if not np.isfinite(cell_weight) or cell_weight <= 0:
            raise ValueError("cell_weight must be finite and positive")
        frequency_chunk = int(frequency_chunk)
        if frequency_chunk < 1:
            raise ValueError("frequency_chunk must be positive")
        if precompute not in (True, False, "auto"):
            raise ValueError("precompute must be True, False, or 'auto'")

        self.frequencies_hz = np.array(frequencies, copy=True)
        self.antenna_positions_m = np.array(antennas, copy=True)
        self.pixel_positions_m = np.array(pixels, copy=True)
        self.propagation_speed_m_s = speed
        self.cell_weight = cell_weight
        self.frequency_chunk = frequency_chunk
        self.distances_m = np.linalg.norm(
            antennas[:, None, :] - pixels[None, :, :], axis=2
        )
        self.shape = (frequencies.size * antennas.shape[0], pixels.shape[0])
        estimated_bytes = int(np.prod(self.shape) * np.dtype(np.complex128).itemsize)
        should_cache = precompute is True or (
            precompute == "auto" and estimated_bytes <= int(max_cache_bytes)
        )
        self._matrix = self._build_matrix() if should_cache else None

    @property
    def n_frequencies(self) -> int:
        return int(self.frequencies_hz.size)

    @property
    def n_angles(self) -> int:
        return int(self.antenna_positions_m.shape[0])

    @property
    def n_pixels(self) -> int:
        return int(self.pixel_positions_m.shape[0])

    @property
    def cached(self) -> bool:
        return self._matrix is not None

    def _phase(self, frequencies: np.ndarray) -> np.ndarray:
        return np.exp(
            -4j
            * np.pi
            * frequencies[:, None, None]
            * self.distances_m[None, :, :]
            / self.propagation_speed_m_s
        ) * self.cell_weight

    def _build_matrix(self) -> np.ndarray:
        blocks = []
        for start in range(0, self.n_frequencies, self.frequency_chunk):
            stop = min(start + self.frequency_chunk, self.n_frequencies)
            blocks.append(self._phase(self.frequencies_hz[start:stop]).reshape(-1, self.n_pixels))
        return np.concatenate(blocks, axis=0)

    def matvec(self, model: np.ndarray) -> np.ndarray:
        """Apply the radar forward model and return ``(frequency, angle)`` data."""
        model = np.asarray(model, dtype=complex).reshape(-1)
        if model.shape != (self.n_pixels,):
            raise ValueError(f"model must contain {self.n_pixels} pixels")
        if self._matrix is not None:
            return (self._matrix @ model).reshape(self.n_frequencies, self.n_angles)
        result = np.empty((self.n_frequencies, self.n_angles), dtype=complex)
        for start in range(0, self.n_frequencies, self.frequency_chunk):
            stop = min(start + self.frequency_chunk, self.n_frequencies)
            result[start:stop] = np.einsum(
                "fap,p->fa",
                self._phase(self.frequencies_hz[start:stop]),
                model,
                optimize=True,
            )
        return result

    def rmatvec(self, data: np.ndarray) -> np.ndarray:
        """Apply the exact Hermitian adjoint and return a flat complex image."""
        data = np.asarray(data, dtype=complex)
        expected = (self.n_frequencies, self.n_angles)
        if data.shape != expected:
            raise ValueError(f"data has shape {data.shape}; expected {expected}")
        if self._matrix is not None:
            return self._matrix.conj().T @ data.ravel()
        result = np.zeros(self.n_pixels, dtype=complex)
        for start in range(0, self.n_frequencies, self.frequency_chunk):
            stop = min(start + self.frequency_chunk, self.n_frequencies)
            result += np.einsum(
                "fap,fa->p",
                self._phase(self.frequencies_hz[start:stop]).conj(),
                data[start:stop],
                optimize=True,
            )
        return result


def operator_from_measurement(
    record: MeasurementSet,
    grid: ImageGrid2D,
    *,
    propagation_speed_m_s: float,
    scan: int | None = 0,
    radial_offset_m: float = 0.0,
    cell_weight: float = 1.0,
    frequency_chunk: int = 8,
    precompute: bool | str = "auto",
    max_cache_bytes: int = 160_000_000,
) -> tuple[MonostaticScan, MonostaticImagingOperator]:
    """Create a matched scan/operator pair from the named measured-data schema."""
    monostatic_scan = extract_monostatic_scan(
        record, scan=scan, radial_offset_m=radial_offset_m
    )
    operator = MonostaticImagingOperator(
        monostatic_scan.frequencies_hz,
        monostatic_scan.antenna_positions_m,
        grid.positions_m,
        propagation_speed_m_s=propagation_speed_m_s,
        cell_weight=cell_weight,
        frequency_chunk=frequency_chunk,
        precompute=precompute,
        max_cache_bytes=max_cache_bytes,
    )
    return monostatic_scan, operator


def normalized_intensity(values: np.ndarray, *, power: float = 2.0) -> np.ndarray:
    """Convert a real/complex map to finite nonnegative unit-peak intensity."""
    power = float(power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    intensity = np.abs(np.asarray(values)) ** power
    peak = float(np.max(intensity)) if intensity.size else 0.0
    if peak > 0:
        intensity = intensity / peak
    return intensity.astype(float, copy=False)


def measured_das(
    record: MeasurementSet,
    grid: ImageGrid2D,
    *,
    propagation_speed_m_s: float,
    scan: int | None = 0,
    radial_offset_m: float = 0.0,
    power: float = 2.0,
    frequency_chunk: int = 8,
    precompute: bool | str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a normalized monostatic DAS intensity image and reproducibility info."""
    monostatic_scan, operator = operator_from_measurement(
        record,
        grid,
        propagation_speed_m_s=propagation_speed_m_s,
        scan=scan,
        radial_offset_m=radial_offset_m,
        frequency_chunk=frequency_chunk,
        precompute=precompute,
    )
    coherent = operator.rmatvec(monostatic_scan.data).reshape(grid.shape)
    image = normalized_intensity(coherent, power=power)
    info = {
        "method": "measured_das",
        "scan_id": monostatic_scan.scan_id,
        "n_frequencies": operator.n_frequencies,
        "n_angles": operator.n_angles,
        "grid_shape": list(grid.shape),
        "propagation_speed_m_s": float(propagation_speed_m_s),
        "radial_offset_m": float(radial_offset_m),
        "operator_cached": operator.cached,
        "phase_model": "exp(-1j*4*pi*f*distance/speed)",
    }
    return image, info


def _orr_objective(
    operator: MonostaticImagingOperator,
    model: np.ndarray,
    normalized_data: np.ndarray,
    measurement_scale: float,
    regularization: float,
) -> tuple[float, np.ndarray]:
    residual = operator.matvec(model) / measurement_scale - normalized_data
    value = 0.5 * float(np.vdot(residual, residual).real)
    value += 0.5 * regularization * float(np.dot(model, model))
    return value, residual


def _orr_lipschitz(
    operator: MonostaticImagingOperator,
    measurement_scale: float,
    regularization: float,
    iterations: int,
) -> float:
    model = np.ones(operator.n_pixels, dtype=float)
    model /= np.linalg.norm(model)
    for _ in range(max(1, int(iterations))):
        projected = operator.matvec(model) / measurement_scale
        normal = np.real(operator.rmatvec(projected)) / measurement_scale
        normal += regularization * model
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= np.finfo(float).eps:
            return max(regularization, 1.0)
        model = normal / norm
    projected = operator.matvec(model) / measurement_scale
    normal = np.real(operator.rmatvec(projected)) / measurement_scale
    normal += regularization * model
    estimate = float(np.dot(model, normal))
    return max(1.05 * estimate, np.finfo(float).eps)


def orr_reconstruct(
    record: MeasurementSet,
    grid: ImageGrid2D,
    *,
    propagation_speed_m_s: float,
    scan: int | None = 0,
    radial_offset_m: float = 0.0,
    regularization: float = 0.0,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    minimum_iterations: int = 2,
    step_size: float | None = None,
    power_iterations: int = 8,
    nonnegative: bool = False,
    frequency_chunk: int = 8,
    precompute: bool | str = "auto",
    max_cache_bytes: int = 160_000_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize the normalized ORR least-squares objective by safe gradient descent.

    The returned image is the real reflectivity estimate, not permittivity contrast.
    Its absolute magnitude may be converted to a qualitative intensity map with
    :func:`normalized_intensity`.
    """
    regularization = float(regularization)
    tolerance = float(tolerance)
    max_iterations = int(max_iterations)
    minimum_iterations = int(minimum_iterations)
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if minimum_iterations < 0 or minimum_iterations > max_iterations:
        raise ValueError("minimum_iterations must lie within [0, max_iterations]")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    monostatic_scan, operator = operator_from_measurement(
        record,
        grid,
        propagation_speed_m_s=propagation_speed_m_s,
        scan=scan,
        radial_offset_m=radial_offset_m,
        frequency_chunk=frequency_chunk,
        precompute=precompute,
        max_cache_bytes=max_cache_bytes,
    )
    data_norm = float(np.linalg.norm(monostatic_scan.data))
    if not np.isfinite(data_norm) or data_norm <= np.finfo(float).eps:
        raise ValueError("ORR requires a non-zero finite measured-data norm")
    normalized_data = monostatic_scan.data / data_norm
    measurement_scale = float(np.sqrt(operator.n_frequencies * operator.n_angles))
    if step_size is None:
        lipschitz = _orr_lipschitz(
            operator,
            measurement_scale,
            regularization,
            power_iterations,
        )
        step = 1.0 / lipschitz
    else:
        step = float(step_size)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step_size must be finite and positive")
        lipschitz = 1.0 / step

    model = np.zeros(operator.n_pixels, dtype=float)
    objective, residual = _orr_objective(
        operator, model, normalized_data, measurement_scale, regularization
    )
    history = [objective]
    converged = False
    relative_change = np.inf
    accepted_step = step
    for iteration in range(1, max_iterations + 1):
        gradient = np.real(operator.rmatvec(residual)) / measurement_scale
        gradient += regularization * model
        trial_step = accepted_step
        for _ in range(25):
            candidate = model - trial_step * gradient
            if nonnegative:
                candidate = np.maximum(candidate, 0.0)
            candidate_objective, candidate_residual = _orr_objective(
                operator,
                candidate,
                normalized_data,
                measurement_scale,
                regularization,
            )
            if candidate_objective <= objective + 1e-14:
                break
            trial_step *= 0.5
        else:
            raise RuntimeError("ORR backtracking could not find a decreasing step")
        relative_change = (objective - candidate_objective) / max(
            objective, np.finfo(float).eps
        )
        model = candidate
        objective = candidate_objective
        residual = candidate_residual
        history.append(objective)
        accepted_step = trial_step
        if iteration >= minimum_iterations and 0 <= relative_change < tolerance:
            converged = True
            break

    info = {
        "method": "orr_style_gradient_descent",
        "scope": "qualitative primary-scatter reflectivity; not quantitative chi",
        "scan_id": monostatic_scan.scan_id,
        "iterations": iteration,
        "converged": converged,
        "stopping_reason": "relative_objective_change" if converged else "max_iterations",
        "objective_history": [float(value) for value in history],
        "relative_objective_change": float(relative_change),
        "normalized_data_residual": float(np.linalg.norm(residual)),
        "regularization": regularization,
        "nonnegative": bool(nonnegative),
        "step_size": float(accepted_step),
        "lipschitz_estimate": float(lipschitz),
        "n_frequencies": operator.n_frequencies,
        "n_angles": operator.n_angles,
        "grid_shape": list(grid.shape),
        "propagation_speed_m_s": float(propagation_speed_m_s),
        "radial_offset_m": float(radial_offset_m),
        "operator_cached": operator.cached,
        "phase_model": "exp(-1j*4*pi*f*distance/speed)",
    }
    return model.reshape(grid.shape), info


@register("imager", "measured_das")
class MeasuredDAS(Imager):
    """Registry adapter for monostatic measured-data DAS."""

    def __init__(
        self,
        *,
        propagation_speed_m_s: float,
        n_pixels: int = 64,
        radius_m: float = 0.09,
        radial_offset_m: float = 0.0,
        power: float = 2.0,
    ):
        self.propagation_speed_m_s = float(propagation_speed_m_s)
        self.grid = square_image_grid(n_pixels, radius_m=radius_m)
        self.radial_offset_m = float(radial_offset_m)
        self.power = float(power)

    def image(self, data, geom=None, **kwargs):
        grid = self.grid if geom is None else geom
        if not isinstance(grid, ImageGrid2D):
            raise TypeError("geom must be an ImageGrid2D")
        image, info = measured_das(
            data,
            grid,
            propagation_speed_m_s=kwargs.get(
                "propagation_speed_m_s", self.propagation_speed_m_s
            ),
            scan=kwargs.get("scan", 0),
            radial_offset_m=kwargs.get("radial_offset_m", self.radial_offset_m),
            power=kwargs.get("power", self.power),
            frequency_chunk=kwargs.get("frequency_chunk", 8),
            precompute=kwargs.get("precompute", "auto"),
        )
        return (image, info) if kwargs.get("return_info", False) else image


@register("imager", "orr")
class ORRImager(Imager):
    """Registry adapter for bounded ORR-style gradient-descent imaging."""

    def __init__(
        self,
        *,
        propagation_speed_m_s: float,
        n_pixels: int = 48,
        radius_m: float = 0.09,
        radial_offset_m: float = 0.0,
        regularization: float = 0.0,
        max_iterations: int = 50,
        tolerance: float = 1e-3,
        nonnegative: bool = False,
        power: float = 2.0,
    ):
        self.propagation_speed_m_s = float(propagation_speed_m_s)
        self.grid = square_image_grid(n_pixels, radius_m=radius_m)
        self.radial_offset_m = float(radial_offset_m)
        self.regularization = float(regularization)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.nonnegative = bool(nonnegative)
        self.power = float(power)

    def image(self, data, geom=None, **kwargs):
        grid = self.grid if geom is None else geom
        if not isinstance(grid, ImageGrid2D):
            raise TypeError("geom must be an ImageGrid2D")
        model, info = orr_reconstruct(
            data,
            grid,
            propagation_speed_m_s=kwargs.get(
                "propagation_speed_m_s", self.propagation_speed_m_s
            ),
            scan=kwargs.get("scan", 0),
            radial_offset_m=kwargs.get("radial_offset_m", self.radial_offset_m),
            regularization=kwargs.get("regularization", self.regularization),
            max_iterations=kwargs.get("max_iterations", self.max_iterations),
            tolerance=kwargs.get("tolerance", self.tolerance),
            nonnegative=kwargs.get("nonnegative", self.nonnegative),
            frequency_chunk=kwargs.get("frequency_chunk", 8),
            precompute=kwargs.get("precompute", "auto"),
            max_cache_bytes=kwargs.get("max_cache_bytes", 160_000_000),
        )
        image = normalized_intensity(model, power=kwargs.get("power", self.power))
        return (image, info) if kwargs.get("return_info", False) else image


__all__ = [
    "ImageGrid2D",
    "MonostaticScan",
    "MonostaticImagingOperator",
    "MeasuredDAS",
    "ORRImager",
    "extract_monostatic_scan",
    "measured_das",
    "normalized_intensity",
    "operator_from_measurement",
    "orr_reconstruct",
    "select_frequency_band",
    "square_image_grid",
]

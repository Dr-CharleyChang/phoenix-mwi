"""Axis-aware, provenance-carrying schema for measured microwave data.

The schema is intentionally a small, dependency-free subset of what xarray provides:
the numerical tensor has named dimensions, every dimension has a coordinate, geometry
arrays declare their own dimensions, and every preprocessing step appends immutable
history. Native NPZ serialization never enables pickle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MeasurementSchemaError(ValueError):
    """Raised when measurement axes, coordinates, geometry, or metadata disagree."""


def json_safe(value: Any) -> Any:
    """Convert NumPy-rich metadata into strict JSON-compatible Python values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a value that has already been made JSON-compatible."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validated_name(name: str, *, kind: str) -> str:
    name = str(name)
    if not _NAME_RE.fullmatch(name):
        raise MeasurementSchemaError(
            f"{kind} name {name!r} must match {_NAME_RE.pattern}"
        )
    return name


def _readonly_array(value: Any, *, name: str, allow_object: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object and not allow_object:
        raise MeasurementSchemaError(f"{name} must not use object dtype")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AuxiliaryArray:
    """A geometry/calibration array with explicit named dimensions and a unit."""

    values: np.ndarray
    dims: tuple[str, ...]
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        dims = tuple(_validated_name(dim, kind="dimension") for dim in self.dims)
        if len(dims) != len(set(dims)):
            raise MeasurementSchemaError(f"auxiliary dimensions are not unique: {dims}")
        values = _readonly_array(self.values, name="auxiliary values")
        if values.ndim != len(dims):
            raise MeasurementSchemaError(
                f"auxiliary array has ndim={values.ndim}, but dims={dims}"
            )
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "unit", str(self.unit))
        object.__setattr__(self, "description", str(self.description))


@dataclass(frozen=True)
class MeasurementSet:
    """Canonical frequency-domain measurement tensor plus geometry and provenance.

    A UM-BMID Gen-One record uses dims (scan, frequency, angle). The coords mapping
    holds one 1-D coordinate for each named dimension. Auxiliary dimensions such as
    xyz may also appear in coords and be used by geometry.
    """

    values: np.ndarray
    dims: tuple[str, ...]
    coords: Mapping[str, np.ndarray]
    geometry: Mapping[str, AuxiliaryArray] = field(default_factory=dict)
    metadata: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise MeasurementSchemaError(
                f"unsupported schema_version={self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        dims = tuple(_validated_name(dim, kind="dimension") for dim in self.dims)
        if len(dims) != len(set(dims)):
            raise MeasurementSchemaError(f"data dimensions are not unique: {dims}")
        values = _readonly_array(self.values, name="measurement values")
        if values.ndim != len(dims):
            raise MeasurementSchemaError(
                f"values has ndim={values.ndim}, but dims={dims}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise MeasurementSchemaError("measurement values must be numeric")

        coords: dict[str, np.ndarray] = {}
        for raw_name, raw_values in self.coords.items():
            name = _validated_name(raw_name, kind="coordinate")
            coordinate = _readonly_array(raw_values, name=f"coordinate {name}")
            if coordinate.ndim != 1:
                raise MeasurementSchemaError(f"coordinate {name!r} must be 1-D")
            if coordinate.size == 0:
                raise MeasurementSchemaError(f"coordinate {name!r} must not be empty")
            coords[name] = coordinate
        missing = [dim for dim in dims if dim not in coords]
        if missing:
            raise MeasurementSchemaError(f"missing coordinates for data dimensions: {missing}")
        for axis, dim in enumerate(dims):
            if coords[dim].size != values.shape[axis]:
                raise MeasurementSchemaError(
                    f"coordinate {dim!r} has {coords[dim].size} entries, "
                    f"but data axis {axis} has {values.shape[axis]}"
                )
        if "frequency" not in dims:
            raise MeasurementSchemaError("frequency-domain records require a frequency axis")
        frequencies = np.asarray(coords["frequency"], dtype=float)
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise MeasurementSchemaError("frequency coordinates must be finite and positive")
        if frequencies.size > 1 and np.any(np.diff(frequencies) <= 0):
            raise MeasurementSchemaError("frequency coordinates must be strictly increasing")

        geometry: dict[str, AuxiliaryArray] = {}
        for raw_name, raw_array in self.geometry.items():
            name = _validated_name(raw_name, kind="geometry")
            if not isinstance(raw_array, AuxiliaryArray):
                raise MeasurementSchemaError(
                    f"geometry {name!r} must be an AuxiliaryArray"
                )
            for axis, dim in enumerate(raw_array.dims):
                if dim not in coords:
                    raise MeasurementSchemaError(
                        f"geometry {name!r} uses unknown dimension {dim!r}"
                    )
                if raw_array.values.shape[axis] != coords[dim].size:
                    raise MeasurementSchemaError(
                        f"geometry {name!r} axis {dim!r} has "
                        f"{raw_array.values.shape[axis]} entries; "
                        f"expected {coords[dim].size}"
                    )
            geometry[name] = raw_array

        metadata = tuple(_freeze_json(json_safe(item)) for item in self.metadata)
        if "scan" in dims and metadata and len(metadata) != coords["scan"].size:
            raise MeasurementSchemaError(
                f"metadata has {len(metadata)} records; scan axis has "
                f"{coords['scan'].size}"
            )
        attrs = _freeze_json(json_safe(self.attrs))
        history = tuple(_freeze_json(json_safe(item)) for item in self.history)

        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "attrs", attrs)
        object.__setattr__(self, "history", history)

    def axis(self, dim: str) -> int:
        """Return the integer data-axis index for dim."""
        try:
            return self.dims.index(dim)
        except ValueError:
            raise KeyError(f"data has no dimension {dim!r}; available: {self.dims}") from None

    def select(self, dim: str, indexer: int | slice | Sequence[int] | np.ndarray):
        """Select coordinate entries while preserving the named dimension."""
        if dim not in self.coords:
            raise KeyError(f"unknown coordinate dimension {dim!r}")
        n_items = self.coords[dim].size
        if isinstance(indexer, slice):
            indices = np.arange(n_items)[indexer]
        elif np.isscalar(indexer):
            indices = np.asarray([int(indexer)])
        else:
            raw = np.asarray(indexer)
            if raw.dtype == bool:
                if raw.shape != (n_items,):
                    raise MeasurementSchemaError(
                        f"boolean selector for {dim!r} must have shape {(n_items,)}"
                    )
                indices = np.flatnonzero(raw)
            else:
                indices = raw.astype(int, copy=False).ravel()
        if indices.size == 0:
            raise MeasurementSchemaError(f"selection on {dim!r} is empty")
        if np.any(indices < 0) or np.any(indices >= n_items):
            raise IndexError(f"selection on {dim!r} is outside [0, {n_items})")

        values = self.values
        if dim in self.dims:
            values = np.take(values, indices, axis=self.axis(dim))
        coords = dict(self.coords)
        coords[dim] = np.take(coords[dim], indices)
        geometry: dict[str, AuxiliaryArray] = {}
        for name, array in self.geometry.items():
            array_values = array.values
            if dim in array.dims:
                array_values = np.take(array_values, indices, axis=array.dims.index(dim))
            geometry[name] = AuxiliaryArray(
                array_values, array.dims, array.unit, array.description
            )
        metadata = self.metadata
        if dim == "scan" and metadata:
            metadata = tuple(metadata[int(index)] for index in indices)
        return self.evolve(
            values=values,
            coords=coords,
            geometry=geometry,
            metadata=metadata,
            operation="select",
            parameters={"dimension": dim, "indices": indices.tolist()},
        )

    def evolve(
        self,
        *,
        values: np.ndarray | None = None,
        coords: Mapping[str, np.ndarray] | None = None,
        geometry: Mapping[str, AuxiliaryArray] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        attrs: Mapping[str, Any] | None = None,
        operation: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ):
        """Return a new record, optionally appending one provenance step."""
        history = list(self.history)
        if operation is not None:
            history.append(
                {
                    "operation": str(operation),
                    "parameters": json_safe(parameters or {}),
                }
            )
        return MeasurementSet(
            values=self.values if values is None else values,
            dims=self.dims,
            coords=self.coords if coords is None else coords,
            geometry=self.geometry if geometry is None else geometry,
            metadata=self.metadata if metadata is None else metadata,
            attrs=self.attrs if attrs is None else attrs,
            history=history,
        )

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of data and metadata."""
        digest = sha256()
        digest.update(np.ascontiguousarray(self.values).view(np.uint8))
        manifest = {
            "schema_version": self.schema_version,
            "dims": self.dims,
            "coords": {
                name: json_safe(values) for name, values in sorted(self.coords.items())
            },
            "geometry": {
                name: {
                    "dims": array.dims,
                    "unit": array.unit,
                    "description": array.description,
                    "values": json_safe(array.values),
                }
                for name, array in sorted(self.geometry.items())
            },
            "metadata": json_safe(self.metadata),
            "attrs": json_safe(self.attrs),
            "history": json_safe(self.history),
        }
        digest.update(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return digest.hexdigest()

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe structural summary without copying the data tensor."""
        return {
            "schema_version": self.schema_version,
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "dims": list(self.dims),
            "coordinates": {name: int(value.size) for name, value in self.coords.items()},
            "geometry": {
                name: {
                    "shape": list(value.values.shape),
                    "dims": list(value.dims),
                    "unit": value.unit,
                }
                for name, value in self.geometry.items()
            },
            "metadata_records": len(self.metadata),
            "history_steps": len(self.history),
            "fingerprint": self.fingerprint(),
        }


def save_measurement_npz(record: MeasurementSet, path: str | Path) -> Path:
    """Save a record atomically as a pickle-free compressed NPZ archive."""
    if not isinstance(record, MeasurementSet):
        raise TypeError("record must be a MeasurementSet")
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("native measurement archives must use the .npz suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": record.schema_version,
        "dims": list(record.dims),
        "coordinate_names": sorted(record.coords),
        "geometry": {
            name: {
                "dims": list(array.dims),
                "unit": array.unit,
                "description": array.description,
            }
            for name, array in sorted(record.geometry.items())
        },
        "metadata": json_safe(record.metadata),
        "attrs": json_safe(record.attrs),
        "history": json_safe(record.history),
    }
    payload: dict[str, np.ndarray] = {
        "values": record.values,
        "__manifest_json__": np.asarray(
            json.dumps(manifest, sort_keys=True, allow_nan=False)
        ),
    }
    payload.update({f"coord__{name}": value for name, value in record.coords.items()})
    payload.update(
        {f"geometry__{name}": value.values for name, value in record.geometry.items()}
    )
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            np.savez_compressed(handle, **payload)
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def load_measurement_npz(path: str | Path) -> MeasurementSet:
    """Load a native archive with pickle disabled."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["__manifest_json__"].item()))
        coords = {
            name: archive[f"coord__{name}"]
            for name in manifest["coordinate_names"]
        }
        geometry = {
            name: AuxiliaryArray(
                archive[f"geometry__{name}"],
                tuple(spec["dims"]),
                spec.get("unit", ""),
                spec.get("description", ""),
            )
            for name, spec in manifest["geometry"].items()
        }
        return MeasurementSet(
            values=archive["values"],
            dims=tuple(manifest["dims"]),
            coords=coords,
            geometry=geometry,
            metadata=manifest.get("metadata", ()),
            attrs=manifest.get("attrs", {}),
            history=manifest.get("history", ()),
            schema_version=int(manifest["schema_version"]),
        )


__all__ = [
    "SCHEMA_VERSION",
    "MeasurementSchemaError",
    "AuxiliaryArray",
    "MeasurementSet",
    "json_safe",
    "save_measurement_npz",
    "load_measurement_npz",
]

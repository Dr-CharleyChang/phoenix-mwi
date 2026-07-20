"""UM-BMID Gen-One ingestion, normalization, and provenance helpers.

The public consolidated archive is distributed as Python pickle files. Pickle can run
code while loading, so Phoenix refuses it unless the caller explicitly marks the file as
trusted. The official Zenodo archive can be pinned by its published byte count and MD5.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import md5
import json
from pathlib import Path
import pickle
import shutil
from typing import Any
from urllib.request import urlopen
import warnings
import zlib
from zipfile import ZipFile, ZipInfo

import numpy as np
from scipy.io import loadmat

from ..core.interfaces import DataSource
from ..core.registry import register
from .schema import AuxiliaryArray, MeasurementSchemaError, MeasurementSet, load_measurement_npz

UM_BMID_DOI = "10.5281/zenodo.5120981"
UM_BMID_RECORD_URL = "https://zenodo.org/records/5120981"
UM_BMID_GEN_ONE_URL = (
    "https://zenodo.org/records/5120981/files/gen-one.zip?download=1"
)
UM_BMID_GEN_ONE_BYTES = 350_526_155
UM_BMID_GEN_ONE_MD5 = "4ac179a5b9fb2ec072adc6d2a7ac8ad3"
UM_BMID_GEN_ONE_DATA_FILE = "fd_data_gen_one_s11.pickle"
UM_BMID_GEN_ONE_METADATA_FILE = "metadata_gen_one.pickle"


class UntrustedPickleError(PermissionError):
    """Raised when a pickle load was not explicitly authorized."""


def file_md5(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase MD5 digest for a downloaded public archive."""
    digest = md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_um_bmid_gen_one_archive(path: str | Path) -> dict[str, Any]:
    """Verify the official Gen-One archive against Zenodo's size and checksum."""
    path = Path(path)
    actual_bytes = path.stat().st_size
    actual_md5 = file_md5(path)
    if actual_bytes != UM_BMID_GEN_ONE_BYTES:
        raise ValueError(
            f"Gen-One archive has {actual_bytes} bytes; expected {UM_BMID_GEN_ONE_BYTES}"
        )
    if actual_md5.lower() != UM_BMID_GEN_ONE_MD5:
        raise ValueError(
            f"Gen-One archive MD5 is {actual_md5}; expected {UM_BMID_GEN_ONE_MD5}"
        )
    return {
        "file": path.name,
        "bytes": actual_bytes,
        "md5": actual_md5,
        "doi": UM_BMID_DOI,
        "verified": True,
    }


def download_um_bmid_gen_one(
    archive_path: str | Path,
    *,
    timeout: float = 60.0,
) -> Path:
    """Download the pinned Gen-One archive and publish it only after verification."""
    archive_path = Path(archive_path)
    if archive_path.exists():
        verify_um_bmid_gen_one_archive(archive_path)
        return archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_suffix(archive_path.suffix + ".partial")
    try:
        with urlopen(UM_BMID_GEN_ONE_URL, timeout=timeout) as response:
            with partial.open("wb") as destination:
                shutil.copyfileobj(response, destination, length=1024 * 1024)
        verify_um_bmid_gen_one_archive(partial)
        partial.replace(archive_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return archive_path


def _is_zip_symlink(info: ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _validated_zip_targets(
    archive: ZipFile,
    destination: Path,
    *,
    overwrite: bool,
    max_total_bytes: int,
) -> list[tuple[ZipInfo, Path]]:
    entries: list[tuple[ZipInfo, Path]] = []
    total_bytes = 0
    seen: set[str] = set()
    for info in archive.infolist():
        if _is_zip_symlink(info):
            raise ValueError(f"ZIP symbolic link is not allowed: {info.filename!r}")
        relative = Path(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
        target = (destination / relative).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"ZIP member escapes destination: {info.filename!r}")
        target_key = str(target).casefold()
        if target_key in seen:
            raise ValueError(f"duplicate ZIP destination: {info.filename!r}")
        seen.add(target_key)
        total_bytes += int(info.file_size)
        if total_bytes > int(max_total_bytes):
            raise ValueError(
                f"ZIP expands to more than the allowed {int(max_total_bytes)} bytes"
            )
        if target.exists() and not info.is_dir() and not overwrite:
            raise FileExistsError(f"refusing to overwrite extracted file: {target}")
        entries.append((info, target))
    return entries


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
    max_total_bytes: int = 5 * 1024**3,
) -> tuple[Path, ...]:
    """Extract a ZIP after rejecting absolute paths, traversal, and symbolic links."""
    archive_path = Path(archive_path)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with ZipFile(archive_path) as archive:
        entries = _validated_zip_targets(
            archive,
            destination,
            overwrite=overwrite,
            max_total_bytes=max_total_bytes,
        )
        for info, target in entries:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)
    return tuple(extracted)


def verify_extracted_zip_members(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_total_bytes: int = 5 * 1024**3,
) -> tuple[Path, ...]:
    """Check extracted file sizes and CRC-32 values against a verified ZIP."""
    destination = Path(destination).resolve()
    verified: list[Path] = []
    with ZipFile(archive_path) as archive:
        entries = _validated_zip_targets(
            archive,
            destination,
            overwrite=True,
            max_total_bytes=max_total_bytes,
        )
        for info, target in entries:
            if info.is_dir():
                continue
            if not target.is_file():
                raise FileNotFoundError(f"extracted ZIP member is missing: {target}")
            if target.stat().st_size != info.file_size:
                raise ValueError(f"extracted ZIP member has the wrong size: {target}")
            checksum = 0
            with target.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    checksum = zlib.crc32(chunk, checksum)
            if (checksum & 0xFFFFFFFF) != info.CRC:
                raise ValueError(f"extracted ZIP member has the wrong CRC: {target}")
            verified.append(target)
    return tuple(verified)


def load_trusted_pickle(path: str | Path, *, trusted_pickle: bool = False):
    """Load pickle only after a deliberate trust decision at the call site."""
    if not trusted_pickle:
        raise UntrustedPickleError(
            "pickle loading is disabled by default because pickle can execute code; "
            "use trusted_pickle=True only for a provenance-verified source"
        )
    with Path(path).open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed.*",
            )
            return pickle.load(handle, encoding="latin1")


def _mat_payload(path: Path, *, preferred_keys: Sequence[str]):
    payload = loadmat(path, simplify_cells=True)
    for key in preferred_keys:
        if key in payload:
            return payload[key]
    candidates = [value for key, value in payload.items() if not key.startswith("__")]
    if len(candidates) != 1:
        visible = sorted(key for key in payload if not key.startswith("__"))
        raise ValueError(f"cannot choose a variable from {path}; found {visible}")
    return candidates[0]


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, np.void) and value.dtype.names:
        return {name: value[name] for name in value.dtype.names}
    field_names = getattr(value, "_fieldnames", None)
    if field_names:
        return {name: getattr(value, name) for name in field_names}
    raise TypeError(f"metadata item is not mapping-like: {type(value).__name__}")


def coerce_metadata_records(value: Any) -> list[dict[str, Any]]:
    """Convert common SciPy-MAT and pickle metadata layouts to list-of-dicts."""
    if isinstance(value, Mapping):
        mapping = _plain_mapping(value)
        lengths = []
        for item in mapping.values():
            array = np.asarray(item)
            if array.ndim > 0 and array.size > 1:
                lengths.append(array.size)
        if lengths and len(set(lengths)) == 1:
            n_records = lengths[0]
            return [
                {
                    key: np.asarray(item).ravel()[index]
                    if np.asarray(item).size == n_records
                    else item
                    for key, item in mapping.items()
                }
                for index in range(n_records)
            ]
        return [mapping]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return coerce_metadata_records(value.item())
        items = value.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    return [_plain_mapping(item) for item in items]


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.size == 1:
        value = array.reshape(-1)[0]
        return value.item() if isinstance(value, np.generic) else value
    return value


def _identifier(value: Any) -> Any:
    value = _scalar(value)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _metres_from_cm(value: Any) -> float | None:
    value = _scalar(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number * 1e-2 if np.isfinite(number) else None


def normalize_um_bmid_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep source fields and add stable IDs plus explicit SI-unit aliases."""
    source = {str(key): _scalar(value) for key, value in record.items()}
    normalized = dict(source)
    normalized["sample_id"] = _identifier(
        source.get("id", source.get("sample_id"))
    )
    aliases = {
        "empty_reference_id": ("empty_reference_id", "empty_ref_id", "emp_ref_id"),
        "adipose_reference_id": ("adipose_reference_id", "adi_ref_id"),
        "healthy_reference_id": ("healthy_reference_id", "fib_ref_id"),
    }
    for canonical, candidates in aliases.items():
        value = None
        for candidate in candidates:
            if candidate in source:
                value = _identifier(source[candidate])
                break
        normalized[canonical] = value
    cm_fields = {
        "tumor_radius_m": ("tum_rad",),
        "tumor_x_m": ("tum_x",),
        "tumor_y_m": ("tum_y",),
        "tumor_z_m": ("tum_z",),
        "antenna_radius_m": ("ant_rad",),
        "antenna_z_m": ("ant_z", "ant_height"),
        "adipose_x_m": ("adi_x",),
        "adipose_y_m": ("adi_y",),
        "fibroglandular_x_m": ("fib_x",),
        "fibroglandular_y_m": ("fib_y",),
    }
    for canonical, candidates in cm_fields.items():
        value = None
        for candidate in candidates:
            if candidate in source:
                value = _metres_from_cm(source[candidate])
                break
        normalized[canonical] = value
    return normalized


def gen_one_angles_rad(n_angles: int = 72) -> np.ndarray:
    """Return acquisition-order antenna angles in CCW-positive mathematical radians."""
    n_angles = int(n_angles)
    if n_angles < 1:
        raise ValueError("n_angles must be positive")
    return np.deg2rad(-102.5 - np.arange(n_angles) * (360.0 / n_angles))


def measurement_from_um_bmid_arrays(
    values: Any,
    metadata: Sequence[Mapping[str, Any]],
    *,
    generation: str = "one",
    s_parameter: str = "s11",
    frequency_hz: Any | None = None,
    source_path: str | Path | None = None,
) -> MeasurementSet:
    """Normalize consolidated UM-BMID arrays into the Phoenix schema."""
    values = np.asarray(values)
    if values.ndim != 3:
        raise MeasurementSchemaError(
            "consolidated UM-BMID data must have shape (scan, frequency, angle)"
        )
    n_scans, n_frequencies, n_angles = values.shape
    records = [normalize_um_bmid_metadata(item) for item in metadata]
    if len(records) != n_scans:
        raise MeasurementSchemaError(
            f"metadata has {len(records)} records, but data has {n_scans} scans"
        )
    scan_ids = [item.get("sample_id") for item in records]
    if any(item is None for item in scan_ids):
        raise MeasurementSchemaError("every UM-BMID scan requires a unique metadata ID")
    if len(set(scan_ids)) != len(scan_ids):
        raise MeasurementSchemaError("UM-BMID metadata contains duplicate scan IDs")
    frequencies = (
        np.linspace(1e9, 8e9, n_frequencies)
        if frequency_hz is None
        else np.asarray(frequency_hz, dtype=float)
    )
    if frequencies.shape != (n_frequencies,):
        raise MeasurementSchemaError(
            f"frequency_hz has shape {frequencies.shape}; expected {(n_frequencies,)}"
        )
    if str(generation).lower() != "one":
        raise NotImplementedError("Phase 2 currently freezes the Gen-One geometry only")
    angles = gen_one_angles_rad(n_angles)
    xyz = np.asarray(["x", "y", "z"])
    antenna_positions = np.full((n_scans, n_angles, 3), np.nan, dtype=float)
    for scan_index, item in enumerate(records):
        radius = item.get("antenna_radius_m")
        z_position = item.get("antenna_z_m")
        if radius is not None:
            antenna_positions[scan_index, :, 0] = radius * np.cos(angles)
            antenna_positions[scan_index, :, 1] = radius * np.sin(angles)
        if z_position is not None:
            antenna_positions[scan_index, :, 2] = z_position
    return MeasurementSet(
        values=values,
        dims=("scan", "frequency", "angle"),
        coords={
            "scan": np.asarray(scan_ids),
            "frequency": frequencies,
            "angle": angles,
            "xyz": xyz,
        },
        geometry={
            "antenna_position": AuxiliaryArray(
                antenna_positions,
                ("scan", "angle", "xyz"),
                unit="m",
                description="Gen-One monostatic antenna phase-reference proxy from metadata",
            )
        },
        metadata=records,
        attrs={
            "dataset": "UM-BMID",
            "generation": "one",
            "s_parameter": str(s_parameter).lower(),
            "data_domain": "frequency",
            "frequency_unit": "Hz",
            "angle_unit": "rad",
            "angle_direction": "clockwise acquisition represented in CCW-positive coordinates",
            "doi": UM_BMID_DOI,
            "record_url": UM_BMID_RECORD_URL,
            "license": "CC-BY-4.0",
            "source_file": Path(source_path).name if source_path else None,
        },
        history=(
            {
                "operation": "um_bmid_ingestion",
                "parameters": {
                    "generation": "one",
                    "s_parameter": str(s_parameter).lower(),
                    "source_file": Path(source_path).name if source_path else None,
                },
            },
        ),
    )


def load_um_bmid_raw_txt(path: str | Path) -> np.ndarray:
    """Load one official raw text scan by pairing real/imaginary columns."""
    raw = np.genfromtxt(path, dtype=float, delimiter=None)
    if raw.ndim != 2 or raw.shape[1] % 2:
        raise ValueError(
            "raw UM-BMID text must be 2-D with real/imaginary column pairs"
        )
    return raw[:, 0::2] + 1j * raw[:, 1::2]


def _load_data_file(path: Path, *, trusted_pickle: bool):
    suffix = path.suffix.lower()
    if suffix in {".pickle", ".pkl"}:
        return load_trusted_pickle(path, trusted_pickle=trusted_pickle)
    if suffix == ".mat":
        return _mat_payload(path, preferred_keys=("fd_data", "fd_data_s11"))
    raise ValueError(f"unsupported UM-BMID data file: {path}")


def _load_metadata_file(path: Path, *, trusted_pickle: bool):
    suffix = path.suffix.lower()
    if suffix in {".pickle", ".pkl"}:
        value = load_trusted_pickle(path, trusted_pickle=trusted_pickle)
    elif suffix == ".mat":
        value = _mat_payload(path, preferred_keys=("metadata", "md_s11"))
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported UM-BMID metadata file: {path}")
    return coerce_metadata_records(value)


@register("data_source", "um_bmid")
class UMBMIDDataSource(DataSource):
    """Load a consolidated UM-BMID data file and its matching metadata."""

    def __init__(
        self,
        data_path: str | Path,
        metadata_path: str | Path | None = None,
        *,
        generation: str = "one",
        s_parameter: str = "s11",
        trusted_pickle: bool = False,
        frequency_hz: Any | None = None,
    ):
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.generation = str(generation)
        self.s_parameter = str(s_parameter)
        self.trusted_pickle = bool(trusted_pickle)
        self.frequency_hz = frequency_hz
        self._record: MeasurementSet | None = None

    def _resolve_paths(self) -> tuple[Path, Path]:
        if self.data_path.is_dir():
            candidates = [
                self.data_path / UM_BMID_GEN_ONE_DATA_FILE,
                self.data_path / "fd_data_gen_one_s11.mat",
            ]
            data_path = next((path for path in candidates if path.exists()), None)
            if data_path is None:
                raise FileNotFoundError(
                    f"no consolidated Gen-One S11 file found in {self.data_path}"
                )
        else:
            data_path = self.data_path
        metadata_path = self.metadata_path
        if metadata_path is None:
            candidates = [
                data_path.parent / UM_BMID_GEN_ONE_METADATA_FILE,
                data_path.parent / "metadata_gen_one.mat",
                data_path.parent / "metadata_gen_one.json",
            ]
            metadata_path = next((path for path in candidates if path.exists()), None)
            if metadata_path is None:
                raise FileNotFoundError(
                    f"no matching Gen-One metadata file found beside {data_path}"
                )
        return data_path, metadata_path

    def measurements(self, **kwargs) -> MeasurementSet:
        if self._record is not None:
            return self._record
        if self.data_path.suffix.lower() == ".npz" and self.metadata_path is None:
            self._record = load_measurement_npz(self.data_path)
            return self._record
        data_path, metadata_path = self._resolve_paths()
        values = _load_data_file(data_path, trusted_pickle=self.trusted_pickle)
        metadata = _load_metadata_file(
            metadata_path, trusted_pickle=self.trusted_pickle
        )
        self._record = measurement_from_um_bmid_arrays(
            values,
            metadata,
            generation=self.generation,
            s_parameter=self.s_parameter,
            frequency_hz=self.frequency_hz,
            source_path=data_path,
        )
        return self._record


__all__ = [
    "UM_BMID_DOI",
    "UM_BMID_RECORD_URL",
    "UM_BMID_GEN_ONE_URL",
    "UM_BMID_GEN_ONE_BYTES",
    "UM_BMID_GEN_ONE_MD5",
    "UM_BMID_GEN_ONE_DATA_FILE",
    "UM_BMID_GEN_ONE_METADATA_FILE",
    "UntrustedPickleError",
    "UMBMIDDataSource",
    "coerce_metadata_records",
    "download_um_bmid_gen_one",
    "file_md5",
    "gen_one_angles_rad",
    "load_trusted_pickle",
    "load_um_bmid_raw_txt",
    "measurement_from_um_bmid_arrays",
    "normalize_um_bmid_metadata",
    "safe_extract_zip",
    "verify_um_bmid_gen_one_archive",
    "verify_extracted_zip_members",
]

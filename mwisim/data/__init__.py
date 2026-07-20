"""Synthetic and measured-data sources (Layer 3)."""
from __future__ import annotations

from .schema import (
    SCHEMA_VERSION,
    AuxiliaryArray,
    MeasurementSchemaError,
    MeasurementSet,
    load_measurement_npz,
    save_measurement_npz,
)
from .synthetic import (
    SyntheticDataSource,
    add_complex_gaussian_noise,
    achieved_snr_db,
    receiver_ring,
)
from .um_bmid import (
    UM_BMID_DOI,
    UM_BMID_GEN_ONE_MD5,
    UMBMIDDataSource,
    UntrustedPickleError,
    download_um_bmid_gen_one,
    load_um_bmid_raw_txt,
    measurement_from_um_bmid_arrays,
    safe_extract_zip,
    verify_extracted_zip_members,
    verify_um_bmid_gen_one_archive,
)

__all__ = [
    "SCHEMA_VERSION",
    "AuxiliaryArray",
    "MeasurementSchemaError",
    "MeasurementSet",
    "load_measurement_npz",
    "save_measurement_npz",
    "SyntheticDataSource",
    "add_complex_gaussian_noise",
    "achieved_snr_db",
    "receiver_ring",
    "UM_BMID_DOI",
    "UM_BMID_GEN_ONE_MD5",
    "UMBMIDDataSource",
    "UntrustedPickleError",
    "download_um_bmid_gen_one",
    "load_um_bmid_raw_txt",
    "measurement_from_um_bmid_arrays",
    "safe_extract_zip",
    "verify_extracted_zip_members",
    "verify_um_bmid_gen_one_archive",
]

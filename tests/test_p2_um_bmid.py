"""P2 UM-BMID tests: normalization, trust boundary, MAT/raw ingest, safe ZIP."""
from __future__ import annotations

import pickle
from zipfile import ZipFile

import numpy as np
import pytest
from scipy.io import savemat

from mwisim.core.registry import available
from mwisim.data.um_bmid import (
    UMBMIDDataSource,
    UntrustedPickleError,
    gen_one_angles_rad,
    load_trusted_pickle,
    load_um_bmid_raw_txt,
    measurement_from_um_bmid_arrays,
    safe_extract_zip,
    verify_extracted_zip_members,
)


def _arrays():
    values = np.arange(24).reshape(3, 4, 2).astype(float).astype(complex)
    metadata = [
        {
            "id": 11,
            "empty_ref_id": 12,
            "ant_rad": 20.0,
            "ant_height": -3.0,
            "tum_rad": 0.5,
        },
        {"id": 12, "empty_ref_id": np.nan, "ant_rad": 20.0},
        {"id": 13, "emp_ref_id": 12, "ant_rad": 19.5},
    ]
    return values, metadata


def test_P2U_1_um_bmid_arrays_gain_axes_si_metadata_and_geometry():
    values, metadata = _arrays()
    record = measurement_from_um_bmid_arrays(values, metadata)
    assert record.dims == ("scan", "frequency", "angle")
    assert record.coords["scan"].tolist() == [11, 12, 13]
    assert record.metadata[0]["empty_reference_id"] == 12
    assert record.metadata[0]["antenna_radius_m"] == pytest.approx(0.20)
    assert record.metadata[0]["tumor_radius_m"] == pytest.approx(0.005)
    assert record.coords["angle"][0] == pytest.approx(np.deg2rad(-102.5))
    assert np.allclose(record.coords["angle"], gen_one_angles_rad(2))
    assert record.geometry["antenna_position"].values.shape == (3, 2, 3)
    assert record.attrs["doi"] == "10.5281/zenodo.5120981"


def test_P2U_2_pickle_requires_an_explicit_trust_decision(tmp_path):
    path = tmp_path / "tiny.pickle"
    with path.open("wb") as handle:
        pickle.dump({"safe_test_value": 7}, handle)
    with pytest.raises(UntrustedPickleError, match="disabled by default"):
        load_trusted_pickle(path)
    assert load_trusted_pickle(path, trusted_pickle=True)["safe_test_value"] == 7


def test_P2U_3_mat_data_source_loads_without_pickle(tmp_path):
    values, metadata = _arrays()
    data_path = tmp_path / "fd_data_gen_one_s11.mat"
    metadata_path = tmp_path / "metadata_gen_one.mat"
    savemat(data_path, {"fd_data": values})
    savemat(metadata_path, {"metadata": metadata})
    record = UMBMIDDataSource(data_path, metadata_path).measurements()
    assert np.array_equal(record.values, values)
    assert record.coords["scan"].tolist() == [11, 12, 13]
    assert "um_bmid" in available("data_source")


def test_P2U_4_raw_text_real_imaginary_columns_are_paired(tmp_path):
    raw = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    path = tmp_path / "scan.txt"
    np.savetxt(path, raw)
    parsed = load_um_bmid_raw_txt(path)
    assert np.array_equal(parsed, np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]))


def test_P2U_5_safe_zip_extraction_rejects_path_traversal(tmp_path):
    good_zip = tmp_path / "good.zip"
    with ZipFile(good_zip, "w") as archive:
        archive.writestr("nested/value.txt", "ok")
    extracted = safe_extract_zip(good_zip, tmp_path / "good")
    assert extracted[0].read_text(encoding="utf-8") == "ok"
    assert verify_extracted_zip_members(good_zip, tmp_path / "good") == extracted
    extracted[0].write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong size|wrong CRC"):
        verify_extracted_zip_members(good_zip, tmp_path / "good")

    bad_zip = tmp_path / "bad.zip"
    with ZipFile(bad_zip, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe ZIP"):
        safe_extract_zip(bad_zip, tmp_path / "bad")

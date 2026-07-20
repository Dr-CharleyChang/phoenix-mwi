"""P2-B tests for ID-safe cohort/reference construction."""
from __future__ import annotations

import numpy as np

from mwisim.data.schema import MeasurementSet
from mwisim.evaluation.measured_benchmark import (
    approximate_adipose_radius_m,
    artifact_ablation_records,
    reference_subtracted_targets,
    select_scan_ids,
)


def _protocol_record() -> MeasurementSet:
    scan_ids = [10, 20, 101, 102, 103, 201, 202, 203]
    row_values = np.asarray([10, 20, 1, 2, 3, 4, 5, 6], dtype=float)
    values = np.broadcast_to(row_values[:, None, None], (8, 4, 3)).astype(complex)
    metadata = [
        {
            "sample_id": 10,
            "empty_reference_id": 101,
            "adipose_reference_id": 102,
            "healthy_reference_id": 103,
            "phant_id": "A1F1",
        },
        {
            "sample_id": 20,
            "empty_reference_id": 201,
            "adipose_reference_id": 202,
            "healthy_reference_id": 203,
            "phant_id": "A3F2",
        },
    ]
    metadata.extend(
        {"sample_id": sample_id} for sample_id in scan_ids[len(metadata) :]
    )
    return MeasurementSet(
        values=values,
        dims=("scan", "frequency", "angle"),
        coords={
            "scan": scan_ids,
            "frequency": np.linspace(1e9, 2e9, 4),
            "angle": np.arange(3),
        },
        metadata=metadata,
    )


def test_P2BP_1_selection_and_reference_subtraction_use_ids_not_row_adjacency():
    record = _protocol_record()
    reordered = select_scan_ids(record, [20, 10])
    assert reordered.coords["scan"].tolist() == [20, 10]
    result = reference_subtracted_targets(
        record, [20, 10], reference_key="empty_reference_id"
    )
    assert result.coords["scan"].tolist() == [20, 10]
    assert np.allclose(result.values[:, 0, 0], [16, 9])


def test_P2BP_2_ablation_variants_share_target_order_and_declared_roi_proxy():
    records = artifact_ablation_records(_protocol_record(), [10, 20])
    assert set(records) == {
        "empty_reference",
        "empty_plus_angular_mean",
        "empty_plus_low_rank_1",
        "adipose_reference",
        "healthy_reference",
    }
    assert all(record.coords["scan"].tolist() == [10, 20] for record in records.values())
    assert np.allclose(records["healthy_reference"].values[:, 0, 0], [7, 14])
    assert approximate_adipose_radius_m({"phant_id": "A1F9"}) == 0.05
    assert approximate_adipose_radius_m({"phant_id": "A3F9"}) == 0.07

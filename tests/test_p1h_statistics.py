"""P1-H integration tests: multi-seed aggregation and hardening report."""
from __future__ import annotations

import json

import pytest

from mwisim.evaluation.hardening import (
    aggregate_seed_statistics,
    hardening_scenario_specs,
    run_hardening_suite,
)
from mwisim.reporting import HardeningReporter


def test_P1H_10_seed_statistics_use_sample_standard_deviation():
    rows = [
        {"scenario": "s", "method": "born", "metrics": {"rel_l2": 1.0}},
        {"scenario": "s", "method": "born", "metrics": {"rel_l2": 3.0}},
        {"scenario": "s", "method": "born", "metrics": {"rel_l2": 5.0}},
    ]
    stats = aggregate_seed_statistics(rows)["s"]["born"]["rel_l2"]
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["std"] == pytest.approx(2.0)


@pytest.fixture(scope="module")
def two_seed_suite():
    spec = hardening_scenario_specs()["off_center"]
    return run_hardening_suite(
        scenario_specs={"off_center": spec},
        seeds=(0, 1),
    )


def test_P1H_11_two_seed_suite_runs_four_outputs_per_seed(two_seed_suite):
    suite = two_seed_suite
    assert len(suite["rows"]) == 2 * 4
    assert set(suite["summary"]["off_center"]) == {"das", "born", "dbim", "csi"}
    assert suite["summary"]["off_center"]["born"]["rel_l2"]["n"] == 2
    assert suite["acceptance"]["all_runs_complete"]
    assert suite["acceptance"]["all_numeric_metrics_finite"]
    assert suite["acceptance"]["corrupted_seeds_produce_variation"]


def test_P1H_12_hardening_reporter_writes_reusable_artifacts(two_seed_suite, tmp_path):
    paths = HardeningReporter(dpi=70).write(two_seed_suite, tmp_path)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    payload = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
    assert payload["seeds"] == [0, 1]
    assert "representatives" not in payload
    report = paths["report_md"].read_text(encoding="utf-8")
    assert "off_center" in report and "sample standard deviation" in report

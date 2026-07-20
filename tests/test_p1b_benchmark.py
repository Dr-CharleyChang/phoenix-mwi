"""P1-B tests: one reproducible benchmark and automatic report."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mwisim.evaluation.benchmark import run_phase1_benchmark
from mwisim.inverse.born import BornInverter
from mwisim.inverse.csi import CSIInverter
from mwisim.inverse.dbim import DBIMInverter
from mwisim.reporting import BenchmarkReporter


@pytest.fixture(scope="module")
def phase1_result():
    inverters = {
        "born": BornInverter(mu=1e-2, iter_lim=250),
        "dbim": DBIMInverter(mu=1e-2, max_outer=8, inner_iter=120, tol=1e-3),
        "csi": CSIInverter(
            mu_chi=1e-2, mu_w=1e-3, xi=1.0, max_outer=8, tol=1e-3, init="born"
        ),
    }
    return run_phase1_benchmark(
        problem_kwargs={"eps_r": 1.5, "n_per_lambda": 6, "n_views": 8, "n_rx": 20},
        inverters=inverters,
        warm_start=True,
    )


def test_P1B_1_benchmark_returns_all_platform_outputs(phase1_result):
    result = phase1_result
    assert set(result["methods"]) == {"born", "dbim", "csi"}
    assert result["das"]["image"].shape == result["problem"]["chi_true"].shape
    for record in result["methods"].values():
        assert record["estimate"].shape == result["problem"]["chi_true"].shape
        assert np.all(np.isfinite(record["estimate"]))
        assert {"rel_l2", "eps_r_rmse", "ssim", "support_iou", "data_residual"}.issubset(
            record["metrics"]
        )


def test_P1B_2_nonlinear_methods_improve_the_common_data_fit(phase1_result):
    metrics = {key: value["metrics"] for key, value in phase1_result["methods"].items()}
    assert metrics["dbim"]["data_residual"] < metrics["born"]["data_residual"]
    assert metrics["csi"]["data_residual"] < metrics["born"]["data_residual"]
    assert phase1_result["acceptance"]["das_localizes_inside_true_object"]
    assert phase1_result["acceptance"]["all_outputs_finite"]


def test_P1B_3_warm_start_runtime_is_accounted_for(phase1_result):
    born_time = phase1_result["methods"]["born"]["runtime_s"]
    for key in ("dbim", "csi"):
        record = phase1_result["methods"][key]
        assert record["runtime_s"] == pytest.approx(
            born_time + record["refinement_runtime_s"], rel=1e-12
        )


def test_P1B_4_reporter_writes_png_markdown_and_machine_readable_json(phase1_result, tmp_path):
    paths = BenchmarkReporter(dpi=80).write(phase1_result, tmp_path)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    report = paths["report_md"].read_text(encoding="utf-8")
    assert "| Born |" in report and "| DBIM |" in report and "| CSI |" in report
    payload = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
    assert set(payload["methods"]) == {"born", "dbim", "csi"}
    assert "estimate" not in payload["methods"]["born"]

"""P1-H integration tests: YAML SceneBuilder and concrete Phase1Pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mwisim.config import YamlSceneBuilder, load_yaml_spec
from mwisim.core.registry import available, build
from mwisim.phantoms.composite import CompositeCirclePhantom
from mwisim.pipeline import Phase1Pipeline

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "phase1_hardening.yaml"


def test_P1H_6_offline_yaml_loader_parses_nested_lists_and_scalars():
    spec = load_yaml_spec(EXAMPLE)
    assert spec["schema_version"] == 1
    assert spec["scene"]["inclusions"][1]["label"] == "high_contrast_core"
    assert spec["corruption"]["snr_db"] == pytest.approx(25.0)
    assert spec["algorithms"]["imager"]["params"]["sensitivity_correction"] is True


def test_P1H_7_yaml_scene_builder_implements_the_registry_contract():
    builder = YamlSceneBuilder()
    phantom = builder.build(EXAMPLE)
    assert isinstance(phantom, CompositeCirclePhantom)
    assert np.count_nonzero(phantom.contrast()) > 0
    assert "yaml" in available("scene_builder")
    rebuilt = build("scene_builder", "yaml").build(EXAMPLE)
    assert np.allclose(rebuilt.contrast(), phantom.contrast())


def test_P1H_8_pipeline_builds_model_mismatch_and_runs_all_methods(tmp_path):
    pipeline = Phase1Pipeline(EXAMPLE)
    problem = pipeline.build_problem(seed_override=3)
    assert problem["seed"] == 3
    assert problem["snr_db_achieved"] == pytest.approx(25.0, abs=1e-10)
    assert not np.allclose(problem["rx"], problem["rx_true"])

    result = pipeline.run(seed_override=3, output_dir=tmp_path)
    assert set(result["methods"]) == {"born", "dbim", "csi"}
    assert result["pipeline"]["name"] == "heterogeneous_noisy_geometry_demo"
    assert result["problem_summary"]["support_threshold"] == pytest.approx(0.2)
    assert set(result["artifacts"]) == {
        "method_figure",
        "residual_figure",
        "metrics_json",
        "report_md",
    }
    assert all(Path(path).exists() for path in result["artifacts"].values())


def test_P1H_9_schema_validation_rejects_unknown_version_and_missing_acquisition():
    with pytest.raises(ValueError, match="schema_version"):
        load_yaml_spec({"schema_version": 99, "scene": {}})
    pipeline = Phase1Pipeline(
        {
            "schema_version": 1,
            "scene": {
                "domain_size_m": 0.3,
                "cell_size_m": 0.05,
                "inclusions": [
                    {"center_m": [0.0, 0.0], "radius_m": 0.05, "eps_r": 1.4}
                ],
            },
        }
    )
    with pytest.raises(ValueError, match="acquisition"):
        pipeline.build_problem()

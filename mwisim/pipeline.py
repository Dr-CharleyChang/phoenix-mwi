"""Concrete Phase-1 YAML pipeline: scene -> data -> image/invert -> score -> report."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .config.yaml_scene import YamlSceneBuilder
from .core.registry import build
from .data.synthetic import SyntheticDataSource
from .evaluation.benchmark import run_phase1_benchmark
from .reporting import BenchmarkReporter

# Import concrete packages so their registration decorators run before build-by-name.
from . import imaging as _imaging  # noqa: F401
from . import inverse as _inverse  # noqa: F401


def _component(block, kind: str, default_name: str):
    if block is None:
        name, params = default_name, {}
    elif isinstance(block, str):
        name, params = block, {}
    elif isinstance(block, dict):
        name = str(block.get("name", default_name))
        params = dict(block.get("params", {}))
    else:
        raise TypeError(f"{kind} configuration must be a name string or mapping")
    return build(kind, name, **params)


class Phase1Pipeline:
    """Execute a complete Phase-1 run from a declarative schema-version-1 spec."""

    def __init__(self, spec, scene_builder=None):
        self.source = spec
        self.scene_builder = YamlSceneBuilder() if scene_builder is None else scene_builder
        self.spec = self.scene_builder.load(spec)
        self.base_dir = (
            Path(spec).resolve().parent
            if isinstance(spec, (str, Path)) and Path(spec).exists()
            else Path.cwd()
        )

    def build_problem(self, *, seed_override: int | None = None) -> dict:
        phantom = self.scene_builder.build(self.spec)
        acquisition = dict(self.spec.get("acquisition", {}))
        required = (
            "frequency_hz",
            "n_views",
            "n_receivers",
            "observation_radius_m",
        )
        missing = [key for key in required if key not in acquisition]
        if missing:
            raise ValueError(f"acquisition is missing required fields: {missing}")
        corruption = dict(self.spec.get("corruption", {}))
        seed = int(corruption.get("seed", 0) if seed_override is None else seed_override)
        source = SyntheticDataSource(
            phantom=phantom,
            frequency_hz=float(acquisition["frequency_hz"]),
            n_views=int(acquisition["n_views"]),
            n_receivers=int(acquisition["n_receivers"]),
            observation_radius_m=float(acquisition["observation_radius_m"]),
            snr_db=corruption.get("snr_db"),
            receiver_position_std_m=float(
                corruption.get("receiver_position_std_m", 0.0)
            ),
            seed=seed,
        )
        problem = source.measurements()
        problem["scenario_name"] = str(self.spec.get("name", problem["scene_name"]))
        return problem

    def _algorithms(self):
        algorithms = dict(self.spec.get("algorithms", {}))
        imager = _component(algorithms.get("imager"), "imager", "das")
        blocks = dict(algorithms.get("inverters", {}))
        defaults = {
            "born": {"name": "born", "params": {"mu": 1e-2, "iter_lim": 300}},
            "dbim": {
                "name": "dbim",
                "params": {"mu": 1e-2, "max_outer": 8, "inner_iter": 150, "tol": 1e-3},
            },
            "csi": {
                "name": "csi",
                "params": {
                    "mu_chi": 1e-2,
                    "mu_w": 1e-3,
                    "xi": 1.0,
                    "max_outer": 8,
                    "tol": 1e-3,
                },
            },
        }
        inverters = {
            key: _component(blocks.get(key, defaults[key]), "inverter", key)
            for key in ("born", "dbim", "csi")
        }
        return imager, inverters, bool(algorithms.get("warm_start", True))

    def run(
        self,
        *,
        seed_override: int | None = None,
        output_dir=None,
        write_report: bool = True,
    ) -> dict:
        problem = self.build_problem(seed_override=seed_override)
        imager, inverters, warm_start = self._algorithms()
        evaluation = dict(self.spec.get("evaluation", {}))
        result = run_phase1_benchmark(
            problem=problem,
            inverters=inverters,
            imager=imager,
            warm_start=warm_start,
            support_threshold=float(evaluation.get("support_threshold", 0.5)),
        )
        result["pipeline"] = {
            "schema_version": int(self.spec.get("schema_version", 1)),
            "name": str(self.spec.get("name", "phase1_pipeline")),
            "seed": int(problem["seed"]),
            "config": deepcopy(self.spec),
        }

        configured_output = self.spec.get("reporting", {}).get("output_dir")
        destination = output_dir if output_dir is not None else configured_output
        if write_report and destination is not None:
            destination = Path(destination)
            if not destination.is_absolute():
                # A path written inside YAML is relative to that YAML file. An explicit
                # Python/CLI argument follows normal process semantics and is relative to cwd.
                base = Path.cwd() if output_dir is not None else self.base_dir
                destination = base / destination
            paths = BenchmarkReporter().write(result, destination)
            result["artifacts"] = {key: str(path.resolve()) for key, path in paths.items()}
        return result


__all__ = ["Phase1Pipeline"]

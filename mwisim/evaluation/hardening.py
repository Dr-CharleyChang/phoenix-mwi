"""Monte-Carlo hardening suite for off-centre, dual, and heterogeneous scenes."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import numpy as np


def _algorithm_block() -> dict:
    return {
        "warm_start": True,
        "imager": {
            "name": "das",
            "params": {"power": 2.0, "sensitivity_correction": True},
        },
        "inverters": {
            "born": {"name": "born", "params": {"mu": 0.01, "iter_lim": 250}},
            "dbim": {
                "name": "dbim",
                "params": {
                    "mu": 0.02,
                    "max_outer": 8,
                    "inner_iter": 120,
                    "step": 0.8,
                    "tol": 0.001,
                },
            },
            "csi": {
                "name": "csi",
                "params": {
                    "mu_chi": 0.02,
                    "mu_w": 0.002,
                    "xi": 1.0,
                    "max_outer": 8,
                    "step": 0.8,
                    "tol": 0.001,
                },
            },
        },
    }


def _base_spec(name: str, inclusions: list[dict], support_threshold: float) -> dict:
    return {
        "schema_version": 1,
        "name": name,
        "scene": {
            "type": "composite_circles",
            "domain_size_m": 0.36,
            "cell_size_m": 0.04,
            "background_eps_r": 1.0,
            "overlap_policy": "last_wins",
            "inclusions": inclusions,
        },
        "acquisition": {
            "frequency_hz": 1e9,
            "n_views": 8,
            "n_receivers": 20,
            "observation_radius_m": 0.30,
        },
        "corruption": {
            "snr_db": 25.0,
            "receiver_position_std_m": 0.001,
            "seed": 0,
        },
        "algorithms": _algorithm_block(),
        "evaluation": {"support_threshold": support_threshold},
        "reporting": {"output_dir": None},
    }


def hardening_scenario_specs() -> dict[str, dict]:
    """Return the three canonical P1-H scenarios as detached configuration dicts."""
    off_center = _base_spec(
        "off_center",
        [
            {
                "label": "offset_target",
                "center_m": [0.06, -0.04],
                "radius_m": 0.06,
                "eps_r": 1.50,
            }
        ],
        support_threshold=0.5,
    )
    off_center["corruption"].update(
        {"snr_db": 30.0, "receiver_position_std_m": 0.0005}
    )

    dual = _base_spec(
        "dual_target",
        [
            {
                "label": "left_target",
                "center_m": [-0.07, 0.03],
                "radius_m": 0.05,
                "eps_r": 1.40,
            },
            {
                "label": "right_target",
                "center_m": [0.07, -0.03],
                "radius_m": 0.05,
                "eps_r": 1.65,
            },
        ],
        support_threshold=0.2,
    )

    heterogeneous = _base_spec(
        "heterogeneous_nested",
        [
            {
                "label": "host",
                "center_m": [0.02, 0.00],
                "radius_m": 0.08,
                "eps_r": 1.25,
            },
            {
                "label": "core",
                "center_m": [0.05, 0.02],
                "radius_m": 0.035,
                "eps_r": 1.70,
            },
        ],
        support_threshold=0.2,
    )
    return {
        "off_center": off_center,
        "dual_target": dual,
        "heterogeneous_nested": heterogeneous,
    }


def result_rows(result: dict, scenario: str, seed: int) -> list[dict]:
    """Flatten one benchmark result into one compact row per method/imager."""
    rows = []
    for key, record in result["methods"].items():
        metrics = dict(record["metrics"])
        metrics["runtime_s"] = float(record["runtime_s"])
        rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "method": key,
                "kind": "quantitative",
                "metrics": metrics,
            }
        )
    das_metrics = dict(result["das"]["metrics"])
    das_metrics["runtime_s"] = float(result["das"]["runtime_s"])
    rows.append(
        {
            "scenario": scenario,
            "seed": int(seed),
            "method": "das",
            "kind": "qualitative",
            "metrics": das_metrics,
        }
    )
    return rows


def aggregate_seed_statistics(rows: list[dict]) -> dict:
    """Compute count, mean, and sample standard deviation by scenario/method/metric."""
    grouped = defaultdict(list)
    for row in rows:
        for metric, value in row["metrics"].items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                grouped[(row["scenario"], row["method"], metric)].append(float(value))
    summary: dict = {}
    for (scenario, method, metric), values in grouped.items():
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        count = int(finite.size)
        mean = float(np.mean(finite)) if count else float("nan")
        std = float(np.std(finite, ddof=1)) if count > 1 else 0.0
        summary.setdefault(scenario, {}).setdefault(method, {})[metric] = {
            "n": count,
            "mean": mean,
            "std": std,
        }
    return summary


def run_hardening_suite(
    *,
    scenario_specs: dict[str, dict] | None = None,
    seeds=(0, 1, 2),
) -> dict:
    """Run every scenario/seed through the full YAML Pipeline and aggregate metrics."""
    # Local import keeps the module graph acyclic:
    # pipeline -> evaluation.benchmark, while hardening -> pipeline only when executed.
    from ..pipeline import Phase1Pipeline

    specs = hardening_scenario_specs() if scenario_specs is None else deepcopy(scenario_specs)
    seeds = tuple(int(seed) for seed in seeds)
    if not specs:
        raise ValueError("scenario_specs cannot be empty")
    if not seeds:
        raise ValueError("seeds cannot be empty")

    rows = []
    representatives = {}
    run_acceptance = {}
    for scenario, spec in specs.items():
        for seed in seeds:
            result = Phase1Pipeline(spec).run(
                seed_override=seed, write_report=False
            )
            rows.extend(result_rows(result, scenario, seed))
            run_acceptance[f"{scenario}/seed_{seed}"] = dict(result["acceptance"])
            representatives.setdefault(scenario, result)

    summary = aggregate_seed_statistics(rows)
    all_numeric_finite = all(
        np.isfinite(float(value))
        for row in rows
        for value in row["metrics"].values()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    expected_rows = len(specs) * len(seeds) * 4
    variation_values = [
        summary[scenario]["born"]["data_residual"]["std"]
        for scenario in specs
        if "data_residual" in summary[scenario]["born"]
    ]
    acceptance = {
        "all_runs_complete": len(rows) == expected_rows,
        "all_numeric_metrics_finite": bool(all_numeric_finite),
        "corrupted_seeds_produce_variation": bool(
            len(seeds) == 1 or any(value > 0 for value in variation_values)
        ),
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "scenario_specs": specs,
        "rows": rows,
        "summary": summary,
        "representatives": representatives,
        "run_acceptance": run_acceptance,
        "acceptance": acceptance,
    }


__all__ = [
    "hardening_scenario_specs",
    "result_rows",
    "aggregate_seed_statistics",
    "run_hardening_suite",
]

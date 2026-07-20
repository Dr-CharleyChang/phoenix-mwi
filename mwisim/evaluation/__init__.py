"""Evaluation implementations (Layer 8).

Importing this package registers the built-in evaluators.
"""
from __future__ import annotations

from .metrics import RelL2Evaluator
from .image_metrics import (
    ImageMetricsEvaluator,
    rmse,
    eps_r_rmse,
    ssim_2d,
    support_iou,
    support_component_count,
    component_count_error,
    localization_error,
    contrast_recovery_ratio,
    relative_data_residual,
)
from .benchmark import grid_shape, full_forward_prediction, run_phase1_benchmark
from .hardening import (
    hardening_scenario_specs,
    result_rows,
    aggregate_seed_statistics,
    run_hardening_suite,
)

__all__ = [
    "RelL2Evaluator",
    "ImageMetricsEvaluator",
    "rmse",
    "eps_r_rmse",
    "ssim_2d",
    "support_iou",
    "support_component_count",
    "component_count_error",
    "localization_error",
    "contrast_recovery_ratio",
    "relative_data_residual",
    "grid_shape",
    "full_forward_prediction",
    "run_phase1_benchmark",
    "hardening_scenario_specs",
    "result_rows",
    "aggregate_seed_statistics",
    "run_hardening_suite",
]

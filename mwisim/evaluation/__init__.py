"""Evaluation implementations (Layer 8).

Importing this package registers the built-in evaluators.
"""
from __future__ import annotations

from .metrics import RelL2Evaluator
from .measured import (
    frequency_to_time,
    relative_l2,
    reproduce_um_bmid_reference_example,
    um_bmid_iczt_reference,
)
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
from .measured_imaging import (
    aggregate_measured_metrics,
    circular_mask,
    localization_error_m,
    measured_image_metrics,
    peak_location_m,
    signal_to_clutter_ratio_db,
)
from .measured_benchmark import (
    DEFAULT_P2B_CALIBRATION_IDS,
    DEFAULT_P2B_EVALUATION_IDS,
    DEFAULT_P2B_SPEEDS_M_S,
    artifact_ablation_records,
    reference_subtracted_targets,
    run_p2b_benchmark,
    speed_sensitivity,
)

__all__ = [
    "RelL2Evaluator",
    "frequency_to_time",
    "relative_l2",
    "reproduce_um_bmid_reference_example",
    "um_bmid_iczt_reference",
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
    "aggregate_measured_metrics",
    "circular_mask",
    "localization_error_m",
    "measured_image_metrics",
    "peak_location_m",
    "signal_to_clutter_ratio_db",
    "DEFAULT_P2B_CALIBRATION_IDS",
    "DEFAULT_P2B_EVALUATION_IDS",
    "DEFAULT_P2B_SPEEDS_M_S",
    "artifact_ablation_records",
    "reference_subtracted_targets",
    "run_p2b_benchmark",
    "speed_sensitivity",
]

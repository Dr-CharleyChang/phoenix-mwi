"""Reproducible Phase-1 benchmark: DAS plus Born, DBIM, and CSI on one problem."""
from __future__ import annotations

from datetime import datetime
from time import perf_counter

import numpy as np

from .image_metrics import (
    ImageMetricsEvaluator,
    localization_error,
    relative_data_residual,
    ssim_2d,
    support_iou,
    support_mask,
    support_component_count,
    component_count_error,
)
from ..imaging.das import DASImager
from ..inverse.born import BornInverter
from ..inverse.csi import CSIInverter
from ..inverse.dbim import DBIMInverter, make_dbim_problem, simulate_scattered_data


def grid_shape(centers: np.ndarray) -> tuple[int, int]:
    """Infer ``(Ny, Nx)`` from Phoenix's regular cell-center grid."""
    centers = np.asarray(centers)
    nx = np.unique(np.round(centers[:, 0], 12)).size
    ny = np.unique(np.round(centers[:, 1], 12)).size
    if nx * ny != centers.shape[0]:
        raise ValueError(f"centers do not form a complete {ny}x{nx} regular grid")
    return ny, nx


def full_forward_prediction(problem: dict, chi: np.ndarray) -> np.ndarray:
    """Evaluate the honest nonlinear forward map ``F(chi)`` for common scoring."""
    d_side = float(np.sqrt(problem["dS"]))
    predicted, _ = simulate_scattered_data(
        problem["centers"],
        chi,
        problem["k_b"],
        d_side,
        problem["dS"],
        problem["E_inc_set"],
        problem["rx"],
    )
    return predicted


def _timed_reconstruction(inverter, problem: dict, x0=None):
    start = perf_counter()
    estimate, info = inverter.reconstruct(problem, x0=x0)
    return np.asarray(estimate), dict(info), perf_counter() - start


def _method_record(
    name: str,
    inverter,
    problem: dict,
    evaluator: ImageMetricsEvaluator,
    shape: tuple[int, int],
    *,
    x0=None,
    shared_runtime_s: float = 0.0,
    support_threshold: float = 0.5,
) -> dict:
    estimate, info, refinement_runtime = _timed_reconstruction(inverter, problem, x0=x0)
    scores = evaluator.score(
        estimate,
        problem["chi_true"],
        centers=problem["centers"],
        shape=shape,
        eps_b=problem.get("eps_b", 1.0),
        support_threshold=support_threshold,
    )
    predicted = full_forward_prediction(problem, estimate)
    scores["data_residual"] = relative_data_residual(predicted, problem["d"])
    return {
        "name": name,
        "estimate": estimate,
        "metrics": scores,
        "runtime_s": float(shared_runtime_s + refinement_runtime),
        "refinement_runtime_s": float(refinement_runtime),
        "info": info,
    }


def run_phase1_benchmark(
    problem: dict | None = None,
    *,
    problem_kwargs: dict | None = None,
    inverters: dict | None = None,
    imager=None,
    warm_start: bool = True,
    support_threshold: float = 0.5,
) -> dict:
    """Run the complete P1 benchmark and return structured arrays, metrics, and timing.

    The default is intentionally small enough for a laptop/CI run while still using full
    nonlinear physical data.  DBIM and CSI reuse the Born estimate as a documented warm
    start; their reported ``runtime_s`` includes that shared Born cost.
    """
    if problem is None:
        cfg = {
            "eps_r": 1.5,
            "n_per_lambda": 6,
            "n_views": 8,
            "n_rx": 20,
            "f": 1e9,
            "R_obs_factor": 3.0,
        }
        if problem_kwargs:
            cfg.update(problem_kwargs)
        problem = make_dbim_problem(**cfg)
    required = {"centers", "dS", "k_b", "rx", "E_inc_set", "d", "chi_true", "f"}
    missing = sorted(required.difference(problem))
    if missing:
        raise KeyError(f"benchmark problem is missing keys: {missing}")

    if inverters is None:
        inverters = {
            "born": BornInverter(mu=1e-2, iter_lim=300),
            "dbim": DBIMInverter(mu=1e-2, max_outer=10, inner_iter=150, tol=1e-3),
            "csi": CSIInverter(
                mu_chi=1e-2,
                mu_w=1e-3,
                xi=1.0,
                max_outer=10,
                tol=1e-3,
                init="born",
            ),
        }
    missing_methods = {"born", "dbim", "csi"}.difference(inverters)
    if missing_methods:
        raise ValueError(f"inverters must contain born, dbim, and csi; missing {sorted(missing_methods)}")

    centers = np.asarray(problem["centers"])
    shape = grid_shape(centers)
    support_threshold = float(support_threshold)
    if not 0.0 < support_threshold <= 1.0:
        raise ValueError("support_threshold must lie in (0, 1]")
    evaluator = ImageMetricsEvaluator(support_threshold=support_threshold, component="real")

    born = _method_record(
        "Born",
        inverters["born"],
        problem,
        evaluator,
        shape,
        support_threshold=support_threshold,
    )
    x0 = born["estimate"] if warm_start else None
    shared = born["runtime_s"] if warm_start else 0.0
    dbim = _method_record(
        "DBIM",
        inverters["dbim"],
        problem,
        evaluator,
        shape,
        x0=x0,
        shared_runtime_s=shared,
        support_threshold=support_threshold,
    )
    csi = _method_record(
        "CSI",
        inverters["csi"],
        problem,
        evaluator,
        shape,
        x0=x0,
        shared_runtime_s=shared,
        support_threshold=support_threshold,
    )
    methods = {"born": born, "dbim": dbim, "csi": csi}

    imager = DASImager() if imager is None else imager
    start = perf_counter()
    das_map = np.asarray(imager.image(problem), dtype=float).ravel()
    das_runtime = perf_counter() - start
    truth_display = np.abs(np.asarray(problem["chi_true"]))
    truth_peak = float(np.max(truth_display))
    if truth_peak > 0.0:
        truth_display = truth_display / truth_peak
    das_metrics = {
        "ssim": ssim_2d(das_map, truth_display, shape=shape, component="real", data_range=1.0),
        "support_iou": support_iou(
            das_map, truth_display, threshold_fraction=support_threshold
        ),
        "localization_error_m": localization_error(
            das_map, truth_display, centers, threshold_fraction=support_threshold
        ),
        "component_count": support_component_count(
            das_map, shape=shape, threshold_fraction=support_threshold
        ),
        "true_component_count": support_component_count(
            truth_display, shape=shape, threshold_fraction=support_threshold
        ),
        "component_count_error": component_count_error(
            das_map,
            truth_display,
            shape=shape,
            threshold_fraction=support_threshold,
        ),
    }

    true_mask = support_mask(problem["chi_true"], support_threshold)
    if not np.any(true_mask):
        raise ValueError("chi_true has empty support at the requested threshold")
    true_centroid = np.mean(centers[true_mask], axis=0)
    true_radius = float(np.max(np.linalg.norm(centers[true_mask] - true_centroid, axis=1)))
    acceptance = {
        "dbim_data_fit_better_than_born": bool(
            dbim["metrics"]["data_residual"] < born["metrics"]["data_residual"]
        ),
        "csi_data_fit_better_than_born": bool(
            csi["metrics"]["data_residual"] < born["metrics"]["data_residual"]
        ),
        "das_localizes_inside_true_object": bool(
            das_metrics["localization_error_m"] <= true_radius
        ),
        "all_outputs_finite": bool(
            all(np.all(np.isfinite(record["estimate"])) for record in methods.values())
            and np.all(np.isfinite(das_map))
        ),
    }

    chi_true = np.asarray(problem["chi_true"])
    problem_summary = {
        "frequency_hz": float(problem["f"]),
        "n_cells": int(centers.shape[0]),
        "grid_shape": list(shape),
        "n_views": int(np.asarray(problem["E_inc_set"]).shape[0]),
        "n_receivers": int(np.asarray(problem["rx"]).shape[0]),
        "cell_area_m2": float(problem["dS"]),
        "max_true_contrast": float(np.max(chi_true.real)),
        "max_true_eps_r_for_eps_b_1": float(1.0 + np.max(chi_true.real)),
        "warm_start": bool(warm_start),
        "support_threshold": support_threshold,
        "snr_db_requested": problem.get("snr_db_requested"),
        "receiver_position_std_m": float(problem.get("receiver_position_std_m", 0.0)),
        "seed": int(problem.get("seed", 0)),
        "scene_name": str(problem.get("scene_name", "phase1_problem")),
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "problem": problem,
        "problem_summary": problem_summary,
        "grid_shape": shape,
        "methods": methods,
        "das": {"name": "DAS", "image": das_map, "metrics": das_metrics, "runtime_s": das_runtime},
        "acceptance": acceptance,
    }


__all__ = ["grid_shape", "full_forward_prediction", "run_phase1_benchmark"]

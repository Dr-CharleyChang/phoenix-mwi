"""P1-A tests: unified quantitative image metrics."""
from __future__ import annotations

import numpy as np
import pytest

from mwisim.core.registry import build, available
from mwisim.evaluation.image_metrics import (
    ImageMetricsEvaluator,
    contrast_recovery_ratio,
    eps_r_rmse,
    localization_error,
    rmse,
    ssim_2d,
    support_iou,
)


def test_P1A_1_exact_arithmetic_metrics():
    truth = np.array([0.0, 1.0, 1.0, 0.0])
    estimate = np.array([0.0, 0.8, 1.2, 0.0])
    assert rmse(estimate, truth) == pytest.approx(np.sqrt(0.02))
    assert eps_r_rmse(estimate, truth, eps_b=1.0) == pytest.approx(np.sqrt(0.02))
    assert contrast_recovery_ratio(estimate, truth) == pytest.approx(1.0)


def test_P1A_2_identical_images_are_perfect():
    truth = np.zeros((9, 9), dtype=float)
    truth[3:6, 3:6] = 0.5
    assert ssim_2d(truth, truth) == pytest.approx(1.0, abs=1e-12)
    assert support_iou(truth, truth) == pytest.approx(1.0)


def test_P1A_3_localization_and_iou_have_physical_meaning():
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    truth = np.array([0.0, 1.0, 0.0, 0.0])
    shifted = np.array([0.0, 0.0, 1.0, 0.0])
    assert localization_error(shifted, truth, centers) == pytest.approx(1.0)
    assert support_iou(shifted, truth) == pytest.approx(0.0)


def test_P1A_4_evaluator_returns_the_phase1_scorecard_and_builds_by_name():
    y, x = np.mgrid[-1:1:5j, -1:1:5j]
    centers = np.column_stack([x.ravel(), y.ravel()])
    truth = ((x**2 + y**2) <= 0.6**2).astype(float).ravel()
    evaluator = build("evaluator", "image_metrics")
    scores = evaluator.score(truth, truth, centers=centers, shape=x.shape)
    assert "image_metrics" in available("evaluator")
    assert isinstance(evaluator, ImageMetricsEvaluator)
    assert scores["rel_l2"] == pytest.approx(0.0)
    assert scores["ssim"] == pytest.approx(1.0, abs=1e-12)
    assert scores["support_iou"] == pytest.approx(1.0)
    assert scores["contrast_recovery"] == pytest.approx(1.0)
    assert scores["localization_error_m"] == pytest.approx(0.0, abs=1e-12)

"""Image-quality metrics for quantitative microwave reconstructions.

The functions in this module deliberately depend only on NumPy/SciPy.  They accept
the flat contrast vectors used by the solvers and reshape them only when a genuinely
2-D metric (SSIM) needs an image.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from ..core.interfaces import Evaluator
from ..core.registry import register
from ..metrics import rel_l2_error


def _component(a: np.ndarray, component: str) -> np.ndarray:
    """Select the physically meaningful real-valued component of an array."""
    z = np.asarray(a)
    if component == "real":
        return z.real.astype(float, copy=False)
    if component == "imag":
        return z.imag.astype(float, copy=False)
    if component in ("magnitude", "abs"):
        return np.abs(z).astype(float, copy=False)
    raise ValueError("component must be 'real', 'imag', or 'magnitude'")


def _image(a: np.ndarray, shape=None, component: str = "real") -> np.ndarray:
    """Return a 2-D real image, inferring a square shape when possible."""
    x = _component(a, component)
    if x.ndim == 2:
        if shape is not None and tuple(shape) != x.shape:
            raise ValueError(f"requested shape {tuple(shape)} does not match image shape {x.shape}")
        return x
    if x.ndim != 1:
        raise ValueError(f"expected a flat vector or 2-D image, got shape {x.shape}")
    if shape is None:
        n_side = int(round(np.sqrt(x.size)))
        if n_side * n_side != x.size:
            raise ValueError("shape is required when the flat image is not square")
        shape = (n_side, n_side)
    if int(np.prod(shape)) != x.size:
        raise ValueError(f"shape {tuple(shape)} contains {np.prod(shape)} cells, not {x.size}")
    return x.reshape(tuple(shape))


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error; complex errors contribute through their magnitude."""
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    if estimate.shape != truth.shape:
        raise ValueError(f"estimate shape {estimate.shape} != truth shape {truth.shape}")
    return float(np.sqrt(np.mean(np.abs(estimate - truth) ** 2)))


def eps_r_rmse(estimate_chi: np.ndarray, truth_chi: np.ndarray, eps_b=1.0) -> float:
    """RMSE after converting contrast ``chi`` to relative permittivity ``eps_r``.

    Since ``chi = eps_r / eps_b - 1``, the conversion is
    ``eps_r = eps_b * (1 + chi)``.  For the Phase-1 vacuum background ``eps_b=1``.
    """
    estimate_eps = eps_b * (1.0 + np.asarray(estimate_chi))
    truth_eps = eps_b * (1.0 + np.asarray(truth_chi))
    return rmse(estimate_eps, truth_eps)


def ssim_2d(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    shape=None,
    component: str = "real",
    data_range: float | None = None,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) using Gaussian local statistics.

    This is the standard luminance-times-contrast/structure formula averaged over
    the image.  A score of 1 means identical images; values nearer 0 indicate less
    structural agreement.  No scikit-image dependency is required.
    """
    x = _image(estimate, shape=shape, component=component)
    y = _image(truth, shape=shape, component=component)
    if x.shape != y.shape:
        raise ValueError(f"estimate image shape {x.shape} != truth image shape {y.shape}")
    if data_range is None:
        data_range = float(np.max(y) - np.min(y))
        if data_range <= np.finfo(float).eps:
            data_range = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1.0))
    if data_range <= 0:
        raise ValueError("data_range must be positive")

    mu_x = gaussian_filter(x, sigma=sigma, mode="reflect")
    mu_y = gaussian_filter(y, sigma=sigma, mode="reflect")
    var_x = np.maximum(gaussian_filter(x * x, sigma=sigma, mode="reflect") - mu_x**2, 0.0)
    var_y = np.maximum(gaussian_filter(y * y, sigma=sigma, mode="reflect") - mu_y**2, 0.0)
    cov_xy = gaussian_filter(x * y, sigma=sigma, mode="reflect") - mu_x * mu_y
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    score = float(np.mean(numerator / denominator))
    return float(np.clip(score, -1.0, 1.0))


def support_mask(image: np.ndarray, threshold_fraction: float = 0.5) -> np.ndarray:
    """Threshold a map at a fraction of its maximum magnitude."""
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must lie in [0, 1]")
    values = np.abs(np.asarray(image)).ravel()
    peak = float(np.max(values)) if values.size else 0.0
    if peak == 0.0:
        return np.zeros(values.size, dtype=bool)
    return values >= threshold_fraction * peak


def support_iou(estimate: np.ndarray, truth: np.ndarray, threshold_fraction: float = 0.5) -> float:
    """Intersection-over-union of thresholded estimated and true supports."""
    est = support_mask(estimate, threshold_fraction)
    ref = support_mask(truth, threshold_fraction)
    if est.shape != ref.shape:
        raise ValueError(f"estimate has {est.size} cells but truth has {ref.size}")
    union = np.count_nonzero(est | ref)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(est & ref) / union)


def support_component_count(
    image: np.ndarray,
    *,
    shape=None,
    threshold_fraction: float = 0.5,
) -> int:
    """Count 8-connected objects in a thresholded 2-D support mask."""
    values = np.asarray(image)
    if values.ndim == 2:
        image_shape = values.shape
    elif values.ndim == 1:
        if shape is None:
            n_side = int(round(np.sqrt(values.size)))
            if n_side * n_side != values.size:
                raise ValueError("shape is required for a non-square flat image")
            image_shape = (n_side, n_side)
        else:
            image_shape = tuple(shape)
            if int(np.prod(image_shape)) != values.size:
                raise ValueError("shape does not match the flat image size")
    else:
        raise ValueError("image must be a flat vector or 2-D array")
    mask = support_mask(values, threshold_fraction).reshape(image_shape)
    _, count = label(mask, structure=np.ones((3, 3), dtype=int))
    return int(count)


def component_count_error(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    shape=None,
    threshold_fraction: float = 0.5,
) -> int:
    """Absolute difference between estimated and true connected-object counts."""
    return abs(
        support_component_count(
            estimate, shape=shape, threshold_fraction=threshold_fraction
        )
        - support_component_count(
            truth, shape=shape, threshold_fraction=threshold_fraction
        )
    )


def energy_centroid(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    threshold_fraction: float = 0.5,
    power: float = 1.0,
) -> np.ndarray:
    """Centroid of the strong part of a map in physical coordinates."""
    values = np.abs(np.asarray(image)).ravel()
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] != values.size:
        raise ValueError("centers must have shape (number_of_image_cells, spatial_dimension)")
    mask = support_mask(values, threshold_fraction)
    weights = np.where(mask, values**power, 0.0)
    total = float(np.sum(weights))
    if total == 0.0:
        return np.full(centers.shape[1], np.nan)
    return np.sum(centers * weights[:, None], axis=0) / total


def localization_error(
    estimate: np.ndarray,
    truth: np.ndarray,
    centers: np.ndarray,
    *,
    threshold_fraction: float = 0.5,
) -> float:
    """Euclidean distance between estimated and true strong-region centroids."""
    c_est = energy_centroid(estimate, centers, threshold_fraction=threshold_fraction)
    c_true = energy_centroid(truth, centers, threshold_fraction=threshold_fraction)
    if np.any(~np.isfinite(c_est)) or np.any(~np.isfinite(c_true)):
        return float("inf")
    return float(np.linalg.norm(c_est - c_true))


def contrast_recovery_ratio(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    threshold_fraction: float = 0.5,
) -> float:
    """Mean recovered real contrast inside the true support divided by the true mean."""
    estimate = np.asarray(estimate).real.ravel()
    truth = np.asarray(truth).real.ravel()
    if estimate.shape != truth.shape:
        raise ValueError(f"estimate has {estimate.size} cells but truth has {truth.size}")
    mask = support_mask(truth, threshold_fraction)
    if not np.any(mask):
        raise ValueError("truth has empty support; contrast recovery is undefined")
    truth_mean = float(np.mean(truth[mask]))
    if abs(truth_mean) <= np.finfo(float).eps:
        raise ValueError("truth mean contrast is zero; contrast recovery is undefined")
    return float(np.mean(estimate[mask]) / truth_mean)


def relative_data_residual(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Return ``||predicted - observed||_2 / ||observed||_2``."""
    return rel_l2_error(predicted, observed)


@register("evaluator", "image_metrics")
class ImageMetricsEvaluator(Evaluator):
    """Unified Phase-1 evaluator for quantitative contrast maps."""

    def __init__(self, support_threshold: float = 0.5, component: str = "real"):
        self.support_threshold = float(support_threshold)
        self.component = component

    def score(self, estimate, truth, **kwargs) -> dict:
        centers = kwargs.get("centers")
        shape = kwargs.get("shape")
        eps_b = kwargs.get("eps_b", 1.0)
        threshold = float(kwargs.get("support_threshold", self.support_threshold))
        metrics = {
            "rel_l2": rel_l2_error(estimate, truth),
            "rmse": rmse(estimate, truth),
            "eps_r_rmse": eps_r_rmse(estimate, truth, eps_b=eps_b),
            "ssim": ssim_2d(estimate, truth, shape=shape, component=self.component),
            "support_iou": support_iou(estimate, truth, threshold_fraction=threshold),
            "component_count": support_component_count(
                estimate, shape=shape, threshold_fraction=threshold
            ),
            "true_component_count": support_component_count(
                truth, shape=shape, threshold_fraction=threshold
            ),
            "component_count_error": component_count_error(
                estimate, truth, shape=shape, threshold_fraction=threshold
            ),
        }
        ratio = contrast_recovery_ratio(estimate, truth, threshold_fraction=threshold)
        metrics["contrast_recovery"] = ratio
        metrics["contrast_rel_error"] = abs(ratio - 1.0)
        if centers is not None:
            metrics["localization_error_m"] = localization_error(
                estimate, truth, centers, threshold_fraction=threshold
            )
        return metrics


__all__ = [
    "rmse",
    "eps_r_rmse",
    "ssim_2d",
    "support_mask",
    "support_iou",
    "support_component_count",
    "component_count_error",
    "energy_centroid",
    "localization_error",
    "contrast_recovery_ratio",
    "relative_data_residual",
    "ImageMetricsEvaluator",
]

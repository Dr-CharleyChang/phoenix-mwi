"""P1-C tests: qualitative DAS imaging behind the Imager interface."""
from __future__ import annotations

import numpy as np

from mwisim.core.registry import available, build
from mwisim.evaluation.image_metrics import localization_error, support_iou
from mwisim.imaging.das import DASImager, coherent_backprojection
from mwisim.inverse.born import BornOperator, make_born_problem

C0 = 299_792_458.0


def test_P1C_1_das_is_the_born_adjoint_up_to_the_common_physics_scalar():
    data = make_born_problem(
        eps_r=1.1, n_per_lambda=5, n_views=4, n_rx=12, mode="crime"
    )
    back = coherent_backprojection(data, sensitivity_correction=False)
    op = BornOperator(
        data["centers"], data["rx"], data["E_inc_set"], data["k_b"], data["dS"]
    )
    expected = np.conj(data["k_b"] ** 2 * data["dS"]) * back
    assert np.allclose(op.rmatvec(data["d"]), expected, rtol=1e-12, atol=1e-12)


def test_P1C_2_das_localizes_the_synthetic_cylinder():
    data = make_born_problem(
        eps_r=1.5, n_per_lambda=6, n_views=8, n_rx=20, mode="physical"
    )
    image = DASImager().image(data)
    radius = 0.3 * C0 / data["f"]
    assert image.shape == data["chi_true"].shape
    assert np.all(np.isfinite(image))
    assert np.min(image) >= 0.0 and np.max(image) <= 1.0 + 1e-12
    assert localization_error(image, data["chi_true"], data["centers"]) < radius
    assert support_iou(image, data["chi_true"], threshold_fraction=0.5) > 0.10


def test_P1C_3_das_builds_by_name_and_can_reshape():
    data = make_born_problem(
        eps_r=1.2, n_per_lambda=5, n_views=4, n_rx=12, mode="crime"
    )
    imager = build("imager", "das", power=1.0)
    ny = np.unique(data["centers"][:, 1]).size
    nx = np.unique(data["centers"][:, 0]).size
    image = imager.image(data, reshape=(ny, nx))
    assert "das" in available("imager")
    assert isinstance(imager, DASImager)
    assert image.shape == (ny, nx)

"""Quantitative inversion stage: Born, DBIM, and CSI.

Reuses mwisim.operators (A_op / AH_op) and LSMR/LSQR. Importing this package registers
the built-in inverters (so ``build('inverter', ...)`` works).

I1 = Born, I2 = DBIM, and I3 = CSI. Importing this package registers all three.
"""
from __future__ import annotations

from .born import (
    BornInverter,
    BornOperator,
    green_matrix,
    plane_wave_incidences,
    make_born_problem,
)
from .dbim import (
    DBIMInverter,
    simulate_scattered_data,
    distorted_green_matrix,
    build_frechet_operator,
    make_dbim_problem,
)
from .csi import (
    CSIInverter,
    receiver_operator,
    domain_green_matrix,
    simulate_csi_data,
    update_contrast_sources,
    update_contrast,
    make_csi_problem,
)

__all__ = [
    "BornInverter", "BornOperator", "green_matrix",
    "plane_wave_incidences", "make_born_problem",
    "DBIMInverter", "simulate_scattered_data", "distorted_green_matrix",
    "build_frechet_operator", "make_dbim_problem",
    "CSIInverter", "receiver_operator", "domain_green_matrix",
    "simulate_csi_data", "update_contrast_sources", "update_contrast",
    "make_csi_problem",
]

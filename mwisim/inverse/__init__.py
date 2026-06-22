"""Inversion stage (I1-I4): Born / BIM / DBIM / PnP-DBIM.

Reuses mwisim.operators (A_op / AH_op) and LSMR/LSQR. Importing this package registers
the built-in inverters (so ``build('inverter', ...)`` works).

I1 — Born linear inversion: see ``born.py`` and docs/I1_Tutorial_Born-linear-inversion.md.
"""
from __future__ import annotations

from .born import (
    BornInverter,
    BornOperator,
    green_matrix,
    plane_wave_incidences,
    make_born_problem,
)

__all__ = [
    "BornInverter", "BornOperator", "green_matrix",
    "plane_wave_incidences", "make_born_problem",
]

"""CirclePhantom — a dielectric circular cylinder, wrapping the existing F1 grid/contrast
behind the :class:`Phantom` interface. (Phase 0 refactor: same physics, now pluggable.)
"""
from __future__ import annotations

import numpy as np

from ..core.interfaces import Phantom
from ..core.registry import register
from ..grid import make_grid, assign_contrast   # the validated F1 building blocks

C0 = 299_792_458.0   # speed of light [m/s]


@register("phantom", "circle")
class CirclePhantom(Phantom):
    """Homogeneous dielectric cylinder of radius ``R_cyl`` in background ``eps_b``.

    The grid + contrast are built once in ``__init__`` (cached), so ``grid()`` and
    ``contrast()`` are cheap repeated lookups.

    Parameters (the implementation-specific "knobs" — see PROJECT_PLAN §11.3):
        R_cyl : cylinder radius [m]
        eps_r : cylinder relative permittivity (may be complex for loss)
        eps_b : background relative permittivity (default 1.0 = vacuum)
        d     : cell side [m] (required)
        domain_size : square domain side [m]; default = ``domain_factor * 2 * R_cyl``
        domain_factor : used only if ``domain_size`` is None
    """

    def __init__(self, R_cyl, eps_r, d, eps_b=1.0, domain_size=None, domain_factor=2.0):
        self.R_cyl = float(R_cyl)
        self.eps_r = eps_r
        self.eps_b = eps_b
        self.d = float(d)
        if domain_size is None:
            domain_size = domain_factor * 2 * self.R_cyl
        self.domain_size = float(domain_size)
        # Build once, cache. Full regular grid (no masking) so the FFT backend works.
        self._centers, self._dS = make_grid(self.domain_size, self.d)
        self._chi = assign_contrast(self._centers, self.R_cyl, self.eps_r, self.eps_b)

    # -- Phantom interface -------------------------------------------------
    def grid(self):
        return self._centers, self._dS

    def contrast(self, freq: float | None = None):
        # Non-dispersive cylinder: chi does not depend on frequency (yet).
        return self._chi

    def background_wavenumber(self, freq: float):
        return 2 * np.pi * freq / C0 * np.sqrt(self.eps_b)

    # -- extra (not in the ABC): handy for Mie validation in tests ---------
    def inside_wavenumber(self, freq: float):
        """Wavenumber inside the cylinder (used by the analytic Mie ground truth)."""
        return 2 * np.pi * freq / C0 * np.sqrt(self.eps_r)

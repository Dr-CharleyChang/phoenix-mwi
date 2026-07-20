"""Composite 2-D circular phantoms for off-centre, multi-target, and heterogeneous scenes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.interfaces import Phantom
from ..core.registry import register
from ..grid import make_grid

C0 = 299_792_458.0


@dataclass(frozen=True)
class CircularInclusion:
    """One homogeneous circular material region inside a composite scene."""

    center: tuple[float, float]
    radius: float
    eps_r: complex
    label: str = "inclusion"

    def __post_init__(self):
        center = tuple(float(v) for v in self.center)
        if len(center) != 2 or not np.all(np.isfinite(center)):
            raise ValueError("center must contain two finite coordinates")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be a positive finite number")
        if not np.isfinite(complex(self.eps_r)):
            raise ValueError("eps_r must be finite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "eps_r", complex(self.eps_r))


@register("phantom", "composite_circles")
class CompositeCirclePhantom(Phantom):
    """A regular-grid scene containing one or more circular inclusions.

    Inclusions are applied in list order. With overlap_policy="last_wins" a later
    inclusion overwrites an earlier one, which naturally represents a high-contrast core
    embedded in a lower-contrast host region. overlap_policy="error" rejects overlap.
    """

    def __init__(
        self,
        domain_size: float,
        d: float,
        inclusions,
        eps_b: complex = 1.0,
        overlap_policy: str = "last_wins",
        name: str = "composite",
    ):
        self.domain_size = float(domain_size)
        self.d = float(d)
        self.eps_b = complex(eps_b)
        self.name = str(name)
        self.overlap_policy = overlap_policy
        if self.domain_size <= 0 or self.d <= 0:
            raise ValueError("domain_size and d must be positive")
        if self.eps_b == 0:
            raise ValueError("eps_b must be nonzero")
        if overlap_policy not in ("last_wins", "error"):
            raise ValueError("overlap_policy must be 'last_wins' or 'error'")
        self.inclusions = tuple(
            item if isinstance(item, CircularInclusion) else CircularInclusion(**item)
            for item in inclusions
        )
        if not self.inclusions:
            raise ValueError("at least one circular inclusion is required")
        half = self.domain_size / 2
        for item in self.inclusions:
            cx, cy = item.center
            if abs(cx) + item.radius > half + 1e-12 or abs(cy) + item.radius > half + 1e-12:
                raise ValueError(
                    f"inclusion {item.label!r} extends outside the square domain; "
                    "increase domain_size or move/reduce the inclusion"
                )

        self._centers, self._dS = make_grid(self.domain_size, self.d)
        self._chi = np.zeros(self._centers.shape[0], dtype=complex)
        self._material_index = np.full(self._centers.shape[0], -1, dtype=int)
        occupied = np.zeros(self._centers.shape[0], dtype=bool)
        for index, item in enumerate(self.inclusions):
            delta = self._centers - np.asarray(item.center)[None, :]
            mask = np.sum(delta**2, axis=1) <= item.radius**2
            if not np.any(mask):
                raise ValueError(
                    f"inclusion {item.label!r} contains no grid-cell center; "
                    "reduce d or enlarge the inclusion"
                )
            if overlap_policy == "error" and np.any(occupied & mask):
                raise ValueError(f"inclusion {item.label!r} overlaps an earlier inclusion")
            self._chi[mask] = item.eps_r / self.eps_b - 1.0
            self._material_index[mask] = index
            occupied |= mask

    def grid(self):
        return self._centers, self._dS

    def contrast(self, freq: float | None = None):
        return self._chi

    def background_wavenumber(self, freq: float):
        return 2 * np.pi * float(freq) / C0 * np.sqrt(self.eps_b)

    def material_index(self):
        """Return -1 for background and the winning inclusion index for every cell."""
        return self._material_index.copy()

    def inclusion_masks(self):
        """Return one geometric Boolean mask per inclusion before overlap precedence."""
        masks = []
        for item in self.inclusions:
            delta = self._centers - np.asarray(item.center)[None, :]
            masks.append(np.sum(delta**2, axis=1) <= item.radius**2)
        return tuple(masks)


__all__ = ["CircularInclusion", "CompositeCirclePhantom"]

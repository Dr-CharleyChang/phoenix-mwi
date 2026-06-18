"""Forward-solver implementations (Layer 2).

Importing this package registers the built-in solvers.
"""
from __future__ import annotations

from .mom2d import MoM2D

__all__ = ["MoM2D"]

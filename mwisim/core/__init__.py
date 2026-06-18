"""Core of the phoenix-mwi platform: the abstract interfaces (contracts) and the
registry that make every layer pluggable.

Phase 0 deliverable. Nothing here does physics — these are the *shapes* that concrete
implementations (MoM forward solver, DBIM inverter, ...) must fit into, plus the
registry that lets a user select an implementation by name.

See PROJECT_PLAN.md §4 and §11.3.
"""
from __future__ import annotations

from .interfaces import (
    Phantom,
    SceneBuilder,
    ForwardSolver,
    DataSource,
    Preprocessor,
    Imager,
    Inverter,
    Reconstructor,
    Evaluator,
)
from .registry import register, build, available, REGISTRY

__all__ = [
    "Phantom", "SceneBuilder", "ForwardSolver", "DataSource", "Preprocessor",
    "Imager", "Inverter", "Reconstructor", "Evaluator",
    "register", "build", "available", "REGISTRY",
]

"""Evaluation implementations (Layer 8).

Importing this package registers the built-in evaluators.
"""
from __future__ import annotations

from .metrics import RelL2Evaluator

__all__ = ["RelL2Evaluator"]

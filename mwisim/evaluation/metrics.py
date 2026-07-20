"""Small evaluator adapters retained from Phase 0.

The complete Phase-1 image metric suite lives in :mod:`mwisim.evaluation.image_metrics`.
"""
from __future__ import annotations

from ..core.interfaces import Evaluator
from ..core.registry import register
from ..metrics import rel_l2_error   # the validated F1 helper


@register("evaluator", "rel_l2")
class RelL2Evaluator(Evaluator):
    """Relative L2 error ||estimate - truth|| / ||truth|| (complex-aware)."""

    def score(self, estimate, truth, **kwargs) -> dict:
        return {"rel_l2": rel_l2_error(estimate, truth)}

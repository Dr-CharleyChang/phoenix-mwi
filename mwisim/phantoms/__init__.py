"""Scene / Phantom implementations (Layer 1).

Importing this package registers the built-in phantoms (so ``build('phantom', ...)`` works).
"""
from __future__ import annotations

from .circle import CirclePhantom
from .composite import CircularInclusion, CompositeCirclePhantom

__all__ = ["CirclePhantom", "CircularInclusion", "CompositeCirclePhantom"]

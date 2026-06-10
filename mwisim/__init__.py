"""phoenix-mwi core library.

F1 forward solver (2D TM MoM) + Mie analytic validation.
Convention: time-harmonic e^{+jωt}, outgoing wave ~ H^(2). Keep Green / incident /
Mie all in this same convention (see docs/F1 tutorial §3.4).
"""

__all__ = ["grid", "green", "mom", "operators", "mie", "metrics"]
__version__ = "0.1.0"

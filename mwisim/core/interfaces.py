"""Abstract interfaces (the "contracts") for every platform layer.

An *abstract base class* (ABC) defines the methods a layer must provide, but gives no
implementation. You cannot instantiate an ABC directly; you must subclass it and
implement every ``@abstractmethod``. The payoff is polymorphism: high-level code (a
Pipeline, an Inverter) calls these methods on *any* conforming object without knowing or
caring which concrete class it is. That is what makes the platform pluggable.

Design rules (PROJECT_PLAN §2, §11.3):
  - Method signatures stay minimal and universal; per-implementation knobs go in the
    subclass ``__init__`` (e.g. ``MoM2D(method="cgfft", tol=1e-8)``).
  - Every interface method carries a ``**kwargs`` escape hatch so a subclass can accept
    extra options without breaking the shared contract.

Newcomer note: ``...`` (Ellipsis) is just a "no body here" placeholder; ``ABC`` and
``abstractmethod`` come from Python's standard ``abc`` module. See CODE_GUIDE Appendix D.4.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


# --------------------------------------------------------------------------
# Layer 1 — Scene / Phantom
# --------------------------------------------------------------------------
class Phantom(ABC):
    """A scene: the discretized geometry + its electromagnetic contrast.

    A Phantom answers three questions the forward solver needs: *where are the cells*,
    *what is the contrast in each*, and *what is the background wavenumber at a frequency*.
    """

    @abstractmethod
    def grid(self):
        """Return ``(centers, dS)``: cell-center coordinates ``(N, 2)`` and cell area."""
        ...

    @abstractmethod
    def contrast(self, freq: float | None = None):
        """Return the contrast vector ``chi`` ``(N,)`` (frequency-dependent if dispersive)."""
        ...

    @abstractmethod
    def background_wavenumber(self, freq: float):
        """Return the background wavenumber ``k_b`` at frequency ``freq`` [Hz]."""
        ...


class SceneBuilder(ABC):
    """Build a :class:`Phantom` (+ array geometry) from a declarative spec.

    Phase 1 implements a YAML/txt loader (gprMax-style) so users define problems without
    writing Python. Stubbed here so the contract exists from the start (PROJECT_PLAN §11.2).
    """

    @abstractmethod
    def build(self, spec, **kwargs):
        """Parse ``spec`` (path or dict) and return a :class:`Phantom`."""
        ...


# --------------------------------------------------------------------------
# Layer 2 — Forward solver
# --------------------------------------------------------------------------
class ForwardSolver(ABC):
    """The engine: given a scene, compute fields.

    Implementations: 2D MoM (dense + CG-FFT) now; wrapped gprMax/openEMS (3D) later;
    a Zynq/FPGA adapter later still — all behind this same contract.
    """

    @abstractmethod
    def solve_total_field(self, phantom: Phantom, freq: float, **kwargs):
        """Solve for the in-domain total field. Return ``(E_tot, info)``.

        ``info`` is a dict (iterations, residual, method, N, ...) for the Reporting layer.
        """
        ...

    @abstractmethod
    def scattered_field(self, phantom: Phantom, E_tot, rx, freq: float, **kwargs):
        """Radiate the in-domain field to exterior receivers ``rx`` ``(Nrx, 2)``.

        Return ``E_sc`` ``(Nrx,)``.
        """
        ...


# --------------------------------------------------------------------------
# Layer 3 — Data (ingest + preprocessing)
# --------------------------------------------------------------------------
class DataSource(ABC):
    """Ingest synthetic or real measurements into the unified schema (no alteration)."""

    @abstractmethod
    def measurements(self, **kwargs):
        """Return a dict: fields/S-params, tx/rx geometry, frequencies, calibration meta."""
        ...


class Preprocessor(ABC):
    """One step of the calibration / artifact-removal pipeline (the moat layer).

    Concrete steps chain: Calibrate -> ReferenceSubtract -> ArtifactRemoval.
    """

    @abstractmethod
    def apply(self, data, **kwargs):
        """Return processed ``data`` in the same schema."""
        ...


# --------------------------------------------------------------------------
# Layer 4 — Qualitative imaging
# --------------------------------------------------------------------------
class Imager(ABC):
    """Qualitative reconstruction (DAS / DMAS / confocal / ...): an energy map."""

    @abstractmethod
    def image(self, data, geom, **kwargs):
        """Return a qualitative energy/intensity map over the imaging grid."""
        ...


# --------------------------------------------------------------------------
# Layer 5 — Quantitative inversion
# --------------------------------------------------------------------------
class Inverter(ABC):
    """Quantitative inversion (BIM/DBIM/CSI/...): recover the contrast map."""

    @abstractmethod
    def reconstruct(self, data, forward: ForwardSolver, x0=None, **kwargs):
        """Reconstruct and return ``(chi, info)`` (the χ / ε_r map + an info dict)."""
        ...


# --------------------------------------------------------------------------
# Layer 6 — AI reconstruction
# --------------------------------------------------------------------------
class Reconstructor(ABC):
    """Learned reconstruction / prior (Phase 4)."""

    @abstractmethod
    def predict(self, data, **kwargs):
        """Return a learned reconstruction (or a prior to feed an Inverter)."""
        ...


# --------------------------------------------------------------------------
# Layer 8 — Evaluation
# --------------------------------------------------------------------------
class Evaluator(ABC):
    """Score an estimate against ground truth (SSIM, εr-RMSE, localization, ...)."""

    @abstractmethod
    def score(self, estimate, truth, **kwargs) -> dict:
        """Return a dict of metric_name -> value."""
        ...

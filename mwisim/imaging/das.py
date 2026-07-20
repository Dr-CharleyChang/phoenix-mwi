"""Frequency-domain Delay-and-Sum (DAS) / coherent back-projection.

Phoenix Phase 1 uses plane-wave transmit views rather than point transmit antennas.
For a candidate cell n, DAS conjugates the known incident phase and the cell-to-receiver
Green phase, then coherently sums all receiver/view measurements.  This is the matched
filter ``A^H d`` of the Born forward model, used as a qualitative location/energy map.
"""
from __future__ import annotations

import numpy as np

from ..core.interfaces import Imager
from ..core.registry import register
from ..inverse.born import green_matrix


def _validated_arrays(data: dict):
    required = ("centers", "rx", "E_inc_set", "k_b", "d")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"DAS data is missing required keys: {missing}")
    centers = np.asarray(data["centers"], dtype=float)
    rx = np.asarray(data["rx"], dtype=float)
    E_inc_set = np.atleast_2d(np.asarray(data["E_inc_set"], dtype=complex))
    d_data = np.asarray(data["d"], dtype=complex).ravel()
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (N, 2)")
    if rx.ndim != 2 or rx.shape[1] != 2:
        raise ValueError("rx must have shape (M, 2)")
    if E_inc_set.shape[1] != centers.shape[0]:
        raise ValueError("E_inc_set must have shape (N_views, N_cells)")
    expected = E_inc_set.shape[0] * rx.shape[0]
    if d_data.size != expected:
        raise ValueError(f"d contains {d_data.size} samples; expected N_views*M = {expected}")
    return centers, rx, E_inc_set, complex(data["k_b"]), d_data


def coherent_backprojection(data: dict, *, sensitivity_correction: bool = True) -> np.ndarray:
    """Back-propagate all measurements to the grid and return a complex map ``(N,)``.

    Ignoring the common scalar ``k_b**2*dS``, the operation is

    ``b[n] = sum_i conj(E_inc[i,n]) * sum_m conj(G_tr[m,n]) * d[i,m]``.

    With sensitivity correction, each cell is divided by the norm of its Born-operator
    column so near/strongly illuminated cells do not win merely because of geometry.
    """
    centers, rx, E_inc_set, k_b, d_data = _validated_arrays(data)
    nv, n_cells = E_inc_set.shape
    n_rx = rx.shape[0]
    G_tr = green_matrix(rx, centers, k_b)
    data_blocks = d_data.reshape(nv, n_rx)
    receiver_back = data_blocks @ G_tr.conj()  # row i equals (G_tr^H d_i)^T
    backprojection = np.sum(np.conj(E_inc_set) * receiver_back, axis=0)

    if sensitivity_correction:
        column_energy = np.sum(np.abs(E_inc_set) ** 2, axis=0) * np.sum(np.abs(G_tr) ** 2, axis=0)
        scale = np.sqrt(np.maximum(column_energy, np.finfo(float).eps))
        backprojection = backprojection / scale
    if backprojection.shape != (n_cells,):
        raise RuntimeError("internal DAS shape error")
    return backprojection


def das_intensity(
    data: dict,
    *,
    power: float = 2.0,
    sensitivity_correction: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Return nonnegative DAS intensity ``|backprojection|**power`` on the flat grid."""
    if power <= 0:
        raise ValueError("power must be positive")
    coherent = coherent_backprojection(data, sensitivity_correction=sensitivity_correction)
    intensity = np.abs(coherent) ** float(power)
    if normalize:
        peak = float(np.max(intensity)) if intensity.size else 0.0
        if peak > 0.0:
            intensity = intensity / peak
    return intensity.astype(float, copy=False)


@register("imager", "das")
class DASImager(Imager):
    """Plane-wave frequency-domain DAS imager.

    ``image`` returns a flat ``(N,)`` intensity map by default so it stays aligned with
    ``centers`` and ``chi``.  Pass ``reshape=(Ny, Nx)`` to request a display-ready image.
    """

    def __init__(self, power: float = 2.0, sensitivity_correction: bool = True, normalize: bool = True):
        self.power = float(power)
        self.sensitivity_correction = bool(sensitivity_correction)
        self.normalize = bool(normalize)

    def image(self, data, geom=None, **kwargs):
        problem = dict(data)
        if geom is not None:
            if not isinstance(geom, dict):
                raise TypeError("geom must be a dict when supplied")
            problem.update(geom)
        result = das_intensity(
            problem,
            power=float(kwargs.get("power", self.power)),
            sensitivity_correction=bool(
                kwargs.get("sensitivity_correction", self.sensitivity_correction)
            ),
            normalize=bool(kwargs.get("normalize", self.normalize)),
        )
        shape = kwargs.get("reshape")
        if shape is not None:
            if int(np.prod(shape)) != result.size:
                raise ValueError(f"reshape {tuple(shape)} does not contain {result.size} cells")
            result = result.reshape(tuple(shape))
        return result


__all__ = ["coherent_backprojection", "das_intensity", "DASImager"]

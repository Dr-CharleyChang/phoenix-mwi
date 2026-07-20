"""Full-wave synthetic data with controlled noise and receiver-geometry mismatch."""
from __future__ import annotations

import numpy as np

from ..core.interfaces import DataSource, Phantom
from ..core.registry import register
from ..inverse.born import plane_wave_incidences
from ..inverse.dbim import simulate_scattered_data


def receiver_ring(radius: float, n_receivers: int) -> np.ndarray:
    """Return equally spaced receiver coordinates on a circular ring."""
    radius = float(radius)
    n_receivers = int(n_receivers)
    if radius <= 0 or n_receivers < 1:
        raise ValueError("radius must be positive and n_receivers must be at least 1")
    angles = np.linspace(0.0, 2.0 * np.pi, n_receivers, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def achieved_snr_db(signal: np.ndarray, noisy: np.ndarray) -> float:
    """Measure SNR as 20 log10(norm(signal) / norm(noisy-signal))."""
    signal = np.asarray(signal)
    noise = np.asarray(noisy) - signal
    noise_norm = np.linalg.norm(noise)
    if noise_norm == 0:
        return float("inf")
    signal_norm = np.linalg.norm(signal)
    if signal_norm == 0:
        raise ValueError("SNR is undefined for a zero signal")
    return float(20.0 * np.log10(signal_norm / noise_norm))


def add_complex_gaussian_noise(
    signal: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Add circular complex Gaussian noise scaled to an exact finite-vector SNR.

    The random direction is Gaussian. Its norm is then rescaled so
    norm(signal) / norm(noise) = 10**(snr_db/20) exactly for this realization.
    """
    signal = np.asarray(signal, dtype=complex)
    if snr_db is None:
        noise = np.zeros_like(signal)
        return signal.copy(), noise, float("inf")
    snr_db = float(snr_db)
    signal_norm = np.linalg.norm(signal)
    if signal_norm == 0:
        raise ValueError("cannot add noise at a requested SNR to a zero signal")
    raw = rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    raw_norm = np.linalg.norm(raw)
    if raw_norm == 0:
        raise RuntimeError("random noise realization unexpectedly has zero norm")
    target_noise_norm = signal_norm / (10.0 ** (snr_db / 20.0))
    noise = raw * (target_noise_norm / raw_norm)
    noisy = signal + noise
    return noisy, noise, achieved_snr_db(signal, noisy)


@register("data_source", "synthetic")
class SyntheticDataSource(DataSource):
    """Generate one Phoenix problem dictionary from a Phantom.

    rx is the geometry assumed by imaging/inversion. Data are generated at
    rx_true = rx + jitter. Therefore a nonzero receiver-position standard deviation
    creates a genuine model mismatch instead of moving both data and model together.
    """

    def __init__(
        self,
        phantom: Phantom,
        frequency_hz: float,
        n_views: int,
        n_receivers: int,
        observation_radius_m: float,
        *,
        snr_db: float | None = None,
        receiver_position_std_m: float = 0.0,
        seed: int = 0,
        view_angles_rad=None,
    ):
        self.phantom = phantom
        self.frequency_hz = float(frequency_hz)
        self.n_views = int(n_views)
        self.n_receivers = int(n_receivers)
        self.observation_radius_m = float(observation_radius_m)
        self.snr_db = None if snr_db is None else float(snr_db)
        self.receiver_position_std_m = float(receiver_position_std_m)
        self.seed = int(seed)
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if self.n_views < 1 or self.n_receivers < 1:
            raise ValueError("n_views and n_receivers must be at least 1")
        if self.receiver_position_std_m < 0:
            raise ValueError("receiver_position_std_m cannot be negative")
        if view_angles_rad is None:
            self.view_angles_rad = np.linspace(
                0.0, 2.0 * np.pi, self.n_views, endpoint=False
            )
        else:
            self.view_angles_rad = np.asarray(view_angles_rad, dtype=float).ravel()
            if self.view_angles_rad.size != self.n_views:
                raise ValueError("view_angles_rad length must equal n_views")
        self._cache = None

    def measurements(self, **kwargs):
        refresh = bool(kwargs.get("refresh", False))
        if self._cache is not None and not refresh:
            return self._cache

        centers, dS = self.phantom.grid()
        centers = np.asarray(centers, dtype=float)
        chi_true = np.asarray(self.phantom.contrast(self.frequency_hz), dtype=complex)
        k_b = complex(self.phantom.background_wavenumber(self.frequency_hz))
        d_side = float(np.sqrt(dS))
        E_inc_set = plane_wave_incidences(centers, k_b, self.view_angles_rad)
        rx_assumed = receiver_ring(self.observation_radius_m, self.n_receivers)

        geometry_seed, noise_seed = np.random.SeedSequence(self.seed).spawn(2)
        geometry_rng = np.random.default_rng(geometry_seed)
        noise_rng = np.random.default_rng(noise_seed)
        rx_error = geometry_rng.normal(
            loc=0.0,
            scale=self.receiver_position_std_m,
            size=rx_assumed.shape,
        )
        rx_true = rx_assumed + rx_error
        d_clean, E_tot_set = simulate_scattered_data(
            centers,
            chi_true,
            k_b,
            d_side,
            dS,
            E_inc_set,
            rx_true,
        )
        d_observed, noise, snr_achieved = add_complex_gaussian_noise(
            d_clean, self.snr_db, noise_rng
        )

        eps_b = complex(getattr(self.phantom, "eps_b", 1.0))
        self._cache = {
            "centers": centers,
            "dS": float(dS),
            "k_b": k_b,
            "rx": rx_assumed,
            "rx_true": rx_true,
            "rx_error": rx_error,
            "E_inc_set": E_inc_set,
            "E_tot_true_set": E_tot_set,
            "d": d_observed,
            "d_clean": d_clean,
            "noise": noise,
            "chi_true": chi_true,
            "f": self.frequency_hz,
            "eps_b": eps_b,
            "seed": self.seed,
            "snr_db_requested": self.snr_db,
            "snr_db_achieved": snr_achieved,
            "receiver_position_std_m": self.receiver_position_std_m,
            "receiver_position_rmse_m": float(np.sqrt(np.mean(rx_error**2))),
            "scene_name": str(getattr(self.phantom, "name", type(self.phantom).__name__)),
        }
        return self._cache


__all__ = [
    "receiver_ring",
    "achieved_snr_db",
    "add_complex_gaussian_noise",
    "SyntheticDataSource",
]

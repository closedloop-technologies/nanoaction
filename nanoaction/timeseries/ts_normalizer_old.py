import numpy as np
import math
from dataclasses import dataclass

SQRT2 = math.sqrt(2.0)


def _huber_clip(residual, scale, c=2.5):
    if np.isscalar(scale):
        thresh = c * max(scale, 1e-12)
    else:
        thresh = c * np.maximum(scale, 1e-12)
    return np.clip(residual, -thresh, thresh)


@dataclass
class RobustEMAState:
    mu: np.ndarray
    mad: np.ndarray
    sigma_min: float
    k_tanh: float
    huber_c: float = 2.5
    tau: float = 128.0

    def _alpha(self, dt: float) -> float:
        dt = max(float(dt), 0.0)
        return 1.0 - math.exp(-dt / max(self.tau, 1e-6))

    def forward_step(self, x_t: np.ndarray, dt: float) -> np.ndarray:
        sigma = np.maximum(self.mad / 0.6745, self.sigma_min)
        y_t = np.tanh((x_t - self.mu) / (self.k_tanh * sigma))

        a = self._alpha(dt)
        r = x_t - self.mu
        r_clip = _huber_clip(r, sigma, c=self.huber_c)
        self.mu = self.mu + a * r_clip

        abs_r = np.abs(r)
        abs_r_clip = _huber_clip(abs_r, sigma, c=self.huber_c)
        self.mad = (1.0 - a) * self.mad + a * abs_r_clip

        return y_t

    def inverse_step(self, y_t: np.ndarray, dt: float) -> np.ndarray:
        sigma = np.maximum(self.mad / 0.6745, self.sigma_min)
        y_t = np.clip(y_t, -0.999999, 0.999999)
        x_t = self.mu + self.k_tanh * sigma * np.arctanh(y_t)

        a = self._alpha(dt)
        r = x_t - self.mu
        r_clip = _huber_clip(r, sigma, c=self.huber_c)
        self.mu = self.mu + a * r_clip

        abs_r = np.abs(r)
        abs_r_clip = _huber_clip(abs_r, sigma, c=self.huber_c)
        self.mad = (1.0 - a) * self.mad + a * abs_r_clip

        return x_t


class SingleViewNormalizer:
    def __init__(
        self,
        n_channels: int,
        k_tanh: float = 3.0,
        sigma_min: float = 1e-6,
        huber_c: float = 2.5,
        tau: float = 128.0,
        init_mu: float = 0.0,
        init_mad: float = 1.0,
    ):
        mu = np.full((n_channels,), init_mu, dtype=np.float64)
        mad = np.full((n_channels,), init_mad, dtype=np.float64)
        self.state = RobustEMAState(
            mu=mu, mad=mad, sigma_min=sigma_min, k_tanh=k_tanh, huber_c=huber_c, tau=tau
        )

    def forward_step(self, x_t: np.ndarray, dt: float) -> np.ndarray:
        return self.state.forward_step(np.asarray(x_t, dtype=np.float64), dt)

    def inverse_step(self, y_t: np.ndarray, dt: float) -> np.ndarray:
        return self.state.inverse_step(np.asarray(y_t, dtype=np.float64), dt)


class WaveletHaarNormalizer:
    def __init__(
        self,
        n_channels: int,
        k_tanh_low: float = 3.0,
        k_tanh_high: float = 3.0,
        sigma_min: float = 1e-6,
        huber_c: float = 2.5,
        tau_low: float = 128.0,
        tau_high: float = 64.0,
        init_mu: float = 0.0,
        init_mad: float = 1.0,
    ):
        mu_low = np.full((n_channels,), init_mu, dtype=np.float64)
        mad_low = np.full((n_channels,), init_mad, dtype=np.float64)
        mu_high = np.full((n_channels,), init_mu, dtype=np.float64)
        mad_high = np.full((n_channels,), init_mad, dtype=np.float64)
        self.state_low = RobustEMAState(
            mu_low, mad_low, sigma_min, k_tanh_low, huber_c, tau_low
        )
        self.state_high = RobustEMAState(
            mu_high, mad_high, sigma_min, k_tanh_high, huber_c, tau_high
        )

        self.prev_x = None

    def forward_step(self, x_t: np.ndarray, dt: float):
        x_t = np.asarray(x_t, dtype=np.float64)
        if self.prev_x is None:
            self.prev_x = x_t.copy()
            low = (x_t + self.prev_x) / SQRT2
            high = (x_t - self.prev_x) / SQRT2
        else:
            low = (x_t + self.prev_x) / SQRT2
            high = (x_t - self.prev_x) / SQRT2

        y_low = self.state_low.forward_step(low, dt)
        y_high = self.state_high.forward_step(high, dt)

        self.prev_x = x_t.copy()
        return y_low, y_high

    def inverse_step(self, y_low: np.ndarray, y_high: np.ndarray, dt: float):
        low = self.state_low.inverse_step(np.asarray(y_low, dtype=np.float64), dt)
        high = self.state_high.inverse_step(np.asarray(y_high, dtype=np.float64), dt)
        x_t = (low + high) / SQRT2
        self.prev_x = x_t.copy()
        return x_t

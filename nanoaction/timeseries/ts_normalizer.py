import numpy as np, math
from dataclasses import dataclass


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


class Lifting53Normalizer:
    def __init__(
        self,
        n_channels: int,
        k_tanh_low: float = 3.0,
        k_tanh_high: float = 3.0,
        sigma_min: float = 1e-6,
        huber_c: float = 2.5,
        tau_low: float = 256.0,
        tau_high: float = 64.0,
    ):
        self.state_low = RobustEMAState(
            mu=np.zeros(n_channels),
            mad=np.ones(n_channels),
            sigma_min=sigma_min,
            k_tanh=k_tanh_low,
            huber_c=huber_c,
            tau=tau_low,
        )
        self.state_high = RobustEMAState(
            mu=np.zeros(n_channels),
            mad=np.ones(n_channels),
            sigma_min=sigma_min,
            k_tanh=k_tanh_high,
            huber_c=huber_c,
            tau=tau_high,
        )
        self.have_even = False
        self.prev_detail = np.zeros(n_channels)
        self.even_buf = np.zeros(n_channels)

    def forward_step(self, x_t: np.ndarray, dt: float):
        x_t = np.asarray(x_t, float)
        if not self.have_even:
            self.even_buf = x_t.copy()
            self.have_even = True
            return None, None
        else:
            s_e = self.even_buf
            s_o = x_t
            d = s_o - s_e
            s = s_e + 0.5 * self.prev_detail
            y_low = self.state_low.forward_step(s, dt)
            y_high = self.state_high.forward_step(d, dt)
            self.prev_detail = d.copy()
            self.have_even = False
            return y_low, y_high

    def inverse_step(self, y_low, y_high, dt: float):
        s = self.state_low.inverse_step(np.asarray(y_low, float), dt)
        d = self.state_high.inverse_step(np.asarray(y_high, float), dt)
        s_e = s - 0.5 * self.prev_detail
        s_o = d + s_e
        self.prev_detail = d.copy()
        return s_e, s_o


class StreamingHistogramCDF:
    def __init__(self, bin_edges: np.ndarray, half_life: float = 512.0):
        self.edges = np.asarray(bin_edges, float)
        assert np.all(np.diff(self.edges) > 0)
        self.counts = np.ones(len(self.edges) - 1) * 1e-3
        self.half_life = max(half_life, 1.0)

    def _decay(self, dt: float):
        lam = 0.5 ** (dt / self.half_life)
        self.counts *= lam

    def update(self, x: float, dt: float):
        self._decay(dt)
        idx = np.searchsorted(self.edges, x, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.counts) - 1))
        self.counts[idx] += 1.0

    def cdf(self, x: float) -> float:
        idx = np.searchsorted(self.edges, x, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.counts) - 1))
        cum = self.counts.cumsum()
        total = cum[-1]
        left_mass = cum[idx - 1] if idx > 0 else 0.0
        left_edge = self.edges[idx]
        right_edge = self.edges[idx + 1]
        bin_mass = self.counts[idx]
        if bin_mass <= 0:
            frac = 0.0
        else:
            frac = (x - left_edge) / (right_edge - left_edge)
        return float((left_mass + bin_mass * np.clip(frac, 0.0, 1.0)) / total)

    def ppf(self, u: float) -> float:
        u = float(np.clip(u, 1e-6, 1 - 1e-6))
        cum = self.counts.cumsum()
        total = cum[-1]
        target = u * total
        idx = int(np.searchsorted(cum, target, side="right"))
        idx = np.clip(idx, 0, len(self.counts) - 1)
        left_mass = cum[idx - 1] if idx > 0 else 0.0
        bin_mass = self.counts[idx]
        left_edge = self.edges[idx]
        right_edge = self.edges[idx + 1]
        if bin_mass <= 0:
            return float(left_edge)
        frac = np.clip((target - left_mass) / bin_mass, 0.0, 1.0)
        return float(left_edge + frac * (right_edge - left_edge))


class QuantileSketchState:
    def __init__(
        self,
        n_channels: int,
        bins: int = 257,
        vmin: float = -10.0,
        vmax: float = 10.0,
        half_life: float = 512.0,
    ):
        edges = np.linspace(vmin, vmax, bins)
        self.sketches = [
            StreamingHistogramCDF(edges, half_life=half_life) for _ in range(n_channels)
        ]

    def forward_step(self, z_t: np.ndarray, dt: float) -> np.ndarray:
        z_t = np.asarray(z_t, float)
        y = np.empty_like(z_t)
        for c, s in enumerate(self.sketches):
            u = s.cdf(z_t[c])
            y[c] = 2.0 * u - 1.0
            s.update(z_t[c], dt)
        return y

    def inverse_step(self, y_t: np.ndarray, dt: float) -> np.ndarray:
        y_t = np.asarray(y_t, float)
        z = np.empty_like(y_t)
        for c, s in enumerate(self.sketches):
            u = 0.5 * (np.clip(y_t[c], -0.999999, 0.999999) + 1.0)
            z[c] = s.ppf(u)
            s.update(z[c], dt)
        return z

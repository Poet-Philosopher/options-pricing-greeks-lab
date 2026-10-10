"""Vectorised Monte Carlo pricing under geometric Brownian motion.

Features: antithetic variates (standard errors are computed on *pair
averages*, the statistically correct unit), a closed-form geometric-Asian
control variate for arithmetic-average Asians, convergence series and path
generation for fan charts.

Under the risk-neutral measure ``S_t = S_0 exp((r - q - sigma^2/2) t + sigma W_t)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .bsm import OptionKind, check_kind

Z95 = 1.959963984540054  # two-sided 95% normal quantile
_MAX_CHUNK_ELEMS = 4_000_000


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with uncertainty.

    ``n_paths`` counts simulated paths (an antithetic pair counts as two).
    ``n_samples`` is the number of independent samples the SE is based on.
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_samples: int
    method: str

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the 95% confidence interval."""
        return self.ci_low <= value <= self.ci_high


def _summarise(samples: np.ndarray, disc: float, n_paths: int, method: str) -> MCResult:
    n = samples.size
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    price, se = disc * mean, disc * se
    return MCResult(price, se, price - Z95 * se, price + Z95 * se, n_paths, n, method)


def _payoff(x: np.ndarray, K: float, kind: str) -> np.ndarray:
    return np.maximum(x - K, 0.0) if kind == "call" else np.maximum(K - x, 0.0)


def _validate(S: float, K: float, T: float, sigma: float, n_paths: int) -> None:
    if min(S, K, T, sigma) <= 0:
        raise ValueError("S, K, T and sigma must be positive")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")


def mc_european(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    kind: OptionKind = "call", n_paths: int = 100_000, antithetic: bool = True, seed: int | None = 42,
) -> MCResult:
    """Monte Carlo price of a European option using exact terminal GBM sampling.

    With ``antithetic=True``, ``n_paths // 2`` draws ``Z`` are paired with
    ``-Z`` and the estimator averages each pair's payoff before computing the
    standard error.
    """
    check_kind(kind)
    _validate(S, K, T, sigma, n_paths)
    rng = np.random.default_rng(seed)
    drift, vol = (r - q - 0.5 * sigma**2) * T, sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if antithetic:
        n_pairs = max(n_paths // 2, 2)
        z = rng.standard_normal(n_pairs)
        y = 0.5 * (_payoff(S * np.exp(drift + vol * z), K, kind) + _payoff(S * np.exp(drift - vol * z), K, kind))
        return _summarise(y, disc, 2 * n_pairs, "antithetic")
    z = rng.standard_normal(n_paths)
    return _summarise(_payoff(S * np.exp(drift + vol * z), K, kind), disc, n_paths, "plain")


def convergence_series(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    kind: OptionKind = "call", n_max: int = 200_000, antithetic: bool = True,
    seed: int | None = 42, n_points: int = 60, n_min: int = 100,
) -> pd.DataFrame:
    """Running MC estimate and 95% CI at log-spaced sample sizes (one simulation).

    Columns: ``n_paths, price, std_error, ci_low, ci_high``.
    """
    check_kind(kind)
    _validate(S, K, T, sigma, n_max)
    rng = np.random.default_rng(seed)
    drift, vol = (r - q - 0.5 * sigma**2) * T, sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    per = 2 if antithetic else 1
    m = max(n_max // per, 3)
    z = rng.standard_normal(m)
    if antithetic:
        y = 0.5 * (_payoff(S * np.exp(drift + vol * z), K, kind) + _payoff(S * np.exp(drift - vol * z), K, kind))
    else:
        y = _payoff(S * np.exp(drift + vol * z), K, kind)
    cs, cs2 = np.cumsum(y), np.cumsum(y * y)
    idx = np.unique(np.geomspace(max(n_min // per, 2), m, n_points).astype(int))
    n = idx.astype(float)
    mean = cs[idx - 1] / n
    var = np.maximum((cs2[idx - 1] - n * mean**2) / (n - 1), 0.0)
    se = np.sqrt(var / n)
    price = disc * mean
    return pd.DataFrame({
        "n_paths": idx * per, "price": price, "std_error": disc * se,
        "ci_low": price - Z95 * disc * se, "ci_high": price + Z95 * disc * se,
    })


def simulate_gbm_paths(
    S: float, r: float, q: float, sigma: float, T: float,
    n_paths: int = 200, n_steps: int = 100, seed: int | None = 42, antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GBM paths on an equally spaced grid.

    Returns ``(times, paths)`` with ``paths`` of shape ``(n_paths, n_steps + 1)``
    and ``paths[:, 0] = S``.
    """
    if min(S, T, sigma) <= 0 or n_paths < 1 or n_steps < 1:
        raise ValueError("S, T, sigma, n_paths and n_steps must be positive")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])[:n_paths]
    else:
        z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return np.linspace(0.0, T, n_steps + 1), S * np.exp(log_paths)


def geometric_asian_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, n_steps: int, kind: OptionKind = "call"
) -> float:
    """Closed-form price of a discretely monitored *geometric*-average Asian option.

    Monitoring dates ``t_k = kT/m``, ``k = 1..m``. ``ln G`` is normal with
    ``mu_G = ln S + (r - q - sigma^2/2) T (m+1)/(2m)`` and
    ``sigma_G^2 = sigma^2 T (m+1)(2m+1)/(6 m^2)`` (Kemna-Vorst style).
    """
    check_kind(kind)
    m = n_steps
    mu = math.log(S) + (r - q - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    var = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m * m)
    sg = math.sqrt(var)
    d1 = (mu + var - math.log(K)) / sg
    d2 = d1 - sg
    eg = math.exp(mu + 0.5 * var)
    disc = math.exp(-r * T)
    if kind == "call":
        return disc * (eg * float(ndtr(d1)) - K * float(ndtr(d2)))
    return disc * (K * float(ndtr(-d2)) - eg * float(ndtr(-d1)))


def mc_asian_arithmetic(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    kind: OptionKind = "call", n_paths: int = 100_000, n_steps: int = 50,
    antithetic: bool = True, control_variate: bool = False, seed: int | None = 42,
) -> MCResult:
    """Monte Carlo price of a discretely monitored arithmetic-average Asian option.

    Payoff: ``max(A - K, 0)`` (call) or ``max(K - A, 0)`` (put) with
    ``A = (1/m) sum_{k=1..m} S_{t_k}``. There is no closed form.

    ``control_variate=True`` uses the geometric-average Asian (whose price is
    known exactly) as a control: ``X - beta (C - E[C])`` with
    ``beta = Cov(X, C)/Var(C)`` estimated from the same sample.
    """
    check_kind(kind)
    _validate(S, K, T, sigma, n_paths)
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift, vol = (r - q - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt)
    disc = math.exp(-r * T)
    n_units = max(n_paths // 2, 2) if antithetic else n_paths
    chunk = max(1, _MAX_CHUNK_ELEMS // n_steps)
    xs, cs = [], []
    done = 0
    while done < n_units:
        c = min(chunk, n_units - done)
        z = rng.standard_normal((c, n_steps))
        pairs = (z, -z) if antithetic else (z,)
        x_acc = np.zeros(c)
        g_acc = np.zeros(c)
        for zz in pairs:
            log_s = math.log(S) + np.cumsum(drift + vol * zz, axis=1)
            s = np.exp(log_s)
            x_acc += _payoff(s.mean(axis=1), K, kind)
            if control_variate:
                g_acc += _payoff(np.exp(log_s.mean(axis=1)), K, kind)
        xs.append(x_acc / len(pairs))
        cs.append(g_acc / len(pairs))
        done += c
    x = np.concatenate(xs)
    label = "antithetic" if antithetic else "plain"
    total_paths = n_units * (2 if antithetic else 1)
    if control_variate:
        g = np.concatenate(cs)
        mu_c = geometric_asian_price(S, K, T, r, q, sigma, n_steps, kind) / disc
        cov = np.cov(x, g, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        return _summarise(x - beta * (g - mu_c), disc, total_paths, f"{label} + control variate")
    return _summarise(x, disc, total_paths, label)

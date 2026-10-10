"""Black-Scholes-Merton pricing and analytic Greeks for European options.

Model: the underlying follows geometric Brownian motion with continuous
dividend yield ``q`` and constant volatility ``sigma``; the risk-free rate
``r`` is constant (all rates continuously compounded, ``T`` in years).

All functions accept scalars *or* NumPy arrays (they broadcast), so the
Greeks-explorer charts can evaluate a whole grid in one vectorised call.

Greeks are returned in **raw (calculus) units** by :func:`bs_greeks`:

* ``delta`` -- dV/dS
* ``gamma`` -- d2V/dS2
* ``vega``  -- dV/d(sigma), per 1.00 (= 100 vol points) of volatility
* ``theta`` -- dV/dt, per year (negative of dV/dT)
* ``rho``   -- dV/dr, per 1.00 (= 100 percentage points) of rate

:meth:`Greeks.display_units` converts to the trader-friendly units shown in
the app: vega per 1 vol point, theta per calendar day, rho per 1 rate point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from scipy.special import ndtr

OptionKind = Literal["call", "put"]
ArrayLike = Union[float, np.ndarray]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
_EPS = 1e-12
DAYS_PER_YEAR = 365.0


def check_kind(kind: str) -> None:
    """Raise ``ValueError`` unless ``kind`` is ``"call"`` or ``"put"``."""
    if kind not in ("call", "put"):
        raise ValueError(f"option kind must be 'call' or 'put', got {kind!r}")


def norm_pdf(x: ArrayLike) -> np.ndarray:
    """Standard normal probability density."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: ArrayLike) -> np.ndarray:
    """Standard normal cumulative distribution (via ``scipy.special.ndtr``)."""
    return ndtr(np.asarray(x, dtype=float))


def _prep(*args: ArrayLike) -> tuple[np.ndarray, ...]:
    return np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in args))


def _ret(x: np.ndarray) -> ArrayLike:
    """Return a Python float for 0-d results, otherwise the array."""
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


def d1_d2(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """Return the Black-Scholes ``d1`` and ``d2`` terms.

    ``d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))`` and
    ``d2 = d1 - sigma sqrt(T)``. Undefined (inf/nan) when ``sigma*sqrt(T) = 0``;
    the pricing functions handle that limit separately.
    """
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        vst = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vst
        d2 = d1 - vst
    return _ret(d1), _ret(d2)


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    kind: OptionKind = "call",
) -> ArrayLike:
    """Black-Scholes-Merton price of a European call or put.

    ``C = S e^{-qT} N(d1) - K e^{-rT} N(d2)``;
    ``P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)``.

    When ``T <= 0`` or ``sigma <= 0`` the option is deterministic and the price
    collapses to the discounted forward intrinsic value.
    """
    check_kind(kind)
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    Tc = np.maximum(T, 0.0)
    pv_s = S * np.exp(-q * Tc)
    pv_k = K * np.exp(-r * Tc)
    vst = sigma * np.sqrt(Tc)
    degenerate = vst <= _EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * Tc) / vst
        d2 = d1 - vst
    if kind == "call":
        normal = pv_s * ndtr(d1) - pv_k * ndtr(d2)
        det = np.maximum(pv_s - pv_k, 0.0)
    else:
        normal = pv_k * ndtr(-d2) - pv_s * ndtr(-d1)
        det = np.maximum(pv_k - pv_s, 0.0)
    return _ret(np.where(degenerate, det, normal))


@dataclass(frozen=True)
class Greeks:
    """Container for the five first/second-order Greeks (raw calculus units)."""

    delta: ArrayLike
    gamma: ArrayLike
    vega: ArrayLike
    theta: ArrayLike
    rho: ArrayLike

    def display_units(self) -> "Greeks":
        """Vega per 1 vol point, theta per calendar day, rho per 1 rate point."""
        return Greeks(
            delta=self.delta,
            gamma=self.gamma,
            vega=np.asarray(self.vega) / 100.0 if np.ndim(self.vega) else self.vega / 100.0,
            theta=np.asarray(self.theta) / DAYS_PER_YEAR if np.ndim(self.theta) else self.theta / DAYS_PER_YEAR,
            rho=np.asarray(self.rho) / 100.0 if np.ndim(self.rho) else self.rho / 100.0,
        )

    def as_dict(self) -> dict[str, ArrayLike]:
        """Return the Greeks as ``{name: value}``."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


def bs_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    kind: OptionKind = "call",
) -> Greeks:
    """Analytic BSM Greeks in raw calculus units (see module docstring).

    Closed forms (``phi`` = normal pdf, ``N`` = normal cdf)::

        delta_call = e^{-qT} N(d1)            delta_put = -e^{-qT} N(-d1)
        gamma      = e^{-qT} phi(d1) / (S sigma sqrt(T))
        vega       = S e^{-qT} phi(d1) sqrt(T)
        theta_call = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
        theta_put  = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)
        rho_call   = K T e^{-rT} N(d2)        rho_put   = -K T e^{-rT} N(-d2)

    In the deterministic limit (``sigma sqrt(T) = 0``) the Greeks of the
    discounted-forward-intrinsic payoff are returned (gamma = vega = 0).
    """
    check_kind(kind)
    S, K, T, r, q, sigma = _prep(S, K, T, r, q, sigma)
    Tc = np.maximum(T, 0.0)
    eqt = np.exp(-q * Tc)
    ert = np.exp(-r * Tc)
    pv_s, pv_k = S * eqt, K * ert
    vst = sigma * np.sqrt(Tc)
    degenerate = vst <= _EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * Tc) / vst
        d2 = d1 - vst
        pdf1 = norm_pdf(d1)
        gamma = eqt * pdf1 / (S * vst)
        vega = pv_s * pdf1 * np.sqrt(Tc)
        common = -pv_s * pdf1 * sigma / (2.0 * np.sqrt(Tc))
    if kind == "call":
        delta = eqt * ndtr(d1)
        theta = common - r * pv_k * ndtr(d2) + q * pv_s * ndtr(d1)
        rho = K * Tc * ert * ndtr(d2)
        itm = pv_s > pv_k
        d_delta, d_theta, d_rho = eqt * itm, np.where(itm, q * pv_s - r * pv_k, 0.0), np.where(itm, K * Tc * ert, 0.0)
    else:
        delta = -eqt * ndtr(-d1)
        theta = common + r * pv_k * ndtr(-d2) - q * pv_s * ndtr(-d1)
        rho = -K * Tc * ert * ndtr(-d2)
        itm = pv_k > pv_s
        d_delta, d_theta, d_rho = -eqt * itm, np.where(itm, r * pv_k - q * pv_s, 0.0), np.where(itm, -K * Tc * ert, 0.0)
    zero = np.zeros_like(delta)
    return Greeks(
        delta=_ret(np.where(degenerate, d_delta, delta)),
        gamma=_ret(np.where(degenerate, zero, gamma)),
        vega=_ret(np.where(degenerate, zero, vega)),
        theta=_ret(np.where(degenerate, d_theta, theta)),
        rho=_ret(np.where(degenerate, d_rho, rho)),
    )


def put_call_parity_gap(
    call: ArrayLike, put: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike
) -> ArrayLike:
    """Return ``C - P - (S e^{-qT} - K e^{-rT})``; zero when parity holds."""
    call, put, S, K, T, r, q = _prep(call, put, S, K, T, r, q)
    return _ret(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T)))

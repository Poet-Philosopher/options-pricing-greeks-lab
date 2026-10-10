"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bsm import OptionKind, bs_price, check_kind


@dataclass(frozen=True)
class CRRResult:
    """CRR tree output (``delta``/``gamma`` read off the first two time steps)."""

    price: float
    delta: float
    gamma: float
    u: float
    d: float
    p: float
    n_steps: int
    american: bool


def crr_tree(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    n_steps: int = 200, kind: OptionKind = "call", american: bool = False,
) -> CRRResult:
    """Price an option on a CRR recombining tree via vectorised backward induction.

    Parameters of the tree: ``u = exp(sigma sqrt(dt))``, ``d = 1/u``,
    risk-neutral up probability ``p = (e^{(r-q)dt} - d)/(u - d)``. With
    ``american=True`` each node takes ``max(continuation, immediate exercise)``.

    Raises:
        ValueError: on invalid inputs or if ``p`` falls outside ``[0, 1]``
            (volatility too low for the number of steps -- use fewer steps).
    """
    check_kind(kind)
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if min(S, K, T, sigma) <= 0:
        raise ValueError("S, K, T and sigma must be positive")
    dt = T / n_steps
    vs = sigma * math.sqrt(dt)
    u, d = math.exp(vs), math.exp(-vs)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} is outside [0,1]: volatility is too low for "
            f"{n_steps} steps at these rates. Reduce the number of steps or raise volatility."
        )
    disc = math.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    def payoff(s: np.ndarray) -> np.ndarray:
        return np.maximum(sign * (s - K), 0.0)

    j = np.arange(n_steps + 1)
    v = payoff(S * np.exp((2 * j - n_steps) * vs))
    v1 = v2 = None
    for i in range(n_steps - 1, -1, -1):
        v = disc * (p * v[1 : i + 2] + (1.0 - p) * v[: i + 1])
        if american:
            v = np.maximum(v, payoff(S * np.exp((2 * np.arange(i + 1) - i) * vs)))
        if i == 2:
            v2 = v.copy()
        elif i == 1:
            v1 = v.copy()
    assert v1 is not None and v2 is not None
    s_u, s_d = S * u, S * d
    delta = (v1[1] - v1[0]) / (s_u - s_d)
    s_uu, s_ud, s_dd = S * u * u, S, S * d * d
    delta_up = (v2[2] - v2[1]) / (s_uu - s_ud)
    delta_dn = (v2[1] - v2[0]) / (s_ud - s_dd)
    gamma = (delta_up - delta_dn) / (0.5 * (s_uu - s_dd))
    return CRRResult(float(v[0]), float(delta), float(gamma), u, d, p, n_steps, american)


def crr_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    n_steps: int = 200, kind: OptionKind = "call", american: bool = False,
) -> float:
    """Convenience wrapper returning only the CRR price."""
    return crr_tree(S, K, T, r, q, sigma, n_steps, kind, american).price


def early_exercise_premium(
    S: float, K: float, T: float, r: float, q: float, sigma: float, n_steps: int = 200, kind: OptionKind = "put"
) -> float:
    """American minus European CRR price on the same tree (>= 0)."""
    a = crr_price(S, K, T, r, q, sigma, n_steps, kind, True)
    e = crr_price(S, K, T, r, q, sigma, n_steps, kind, False)
    return a - e


def crr_convergence(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    steps: list[int], kind: OptionKind = "put",
) -> pd.DataFrame:
    """CRR European/American prices over a list of step counts vs analytic BSM.

    Columns: ``steps, european, american, bsm, euro_error, premium``.
    """
    bsm = float(bs_price(S, K, T, r, q, sigma, kind))
    rows = []
    for n in steps:
        e = crr_price(S, K, T, r, q, sigma, n, kind, False)
        a = crr_price(S, K, T, r, q, sigma, n, kind, True)
        rows.append({"steps": n, "european": e, "american": a, "bsm": bsm, "euro_error": e - bsm, "premium": a - e})
    return pd.DataFrame(rows)

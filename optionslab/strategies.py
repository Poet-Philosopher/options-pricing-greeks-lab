"""Multi-leg option strategy payoff / P&L analytics.

A strategy is a list of :class:`Leg` objects. Positions are signed: positive
``qty`` is long, negative is short. Entry premiums default to the theoretical
Black-Scholes-Merton price but can be overridden per leg (e.g. with a quote).

P&L figures are *per unit of underlying* unless multiplied by a lot size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bsm import bs_price

LegType = Literal["call", "put", "stock"]
PRESET_NAMES = ("Covered call", "Bull call spread", "Straddle", "Strangle", "Iron condor")


@dataclass(frozen=True)
class Leg:
    """One position in a strategy.

    Attributes:
        kind: ``"call"``, ``"put"`` or ``"stock"`` (underlying position).
        qty: Signed quantity (+ long, - short), in units of the underlying.
        strike: Strike price (ignored for ``"stock"``).
        premium: Entry price per unit; ``None`` => theoretical BSM price
            (for stock: the current spot).
    """

    kind: LegType
    qty: float
    strike: float = 0.0
    premium: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put", "stock"):
            raise ValueError(f"unknown leg kind {self.kind!r}")
        if self.qty == 0 or not math.isfinite(self.qty):
            raise ValueError("leg quantity must be a non-zero finite number")
        if self.kind != "stock" and not (self.strike > 0 and math.isfinite(self.strike)):
            raise ValueError("option legs need a positive strike")
        if self.premium is not None and (self.premium < 0 or not math.isfinite(self.premium)):
            raise ValueError("premium must be a non-negative number")


@dataclass(frozen=True)
class StrategySummary:
    """Key figures for a strategy held to expiry (per unit of underlying)."""

    entry_cost: float            # net debit (+) / credit (-)
    max_profit: float            # may be +inf
    max_loss: float              # most negative P&L; may be -inf
    breakevens: tuple[float, ...]
    unlimited_profit: bool
    unlimited_loss: bool


def entry_price(leg: Leg, spot: float, T: float, r: float, q: float, sigma: float) -> float:
    """Per-unit entry price for ``leg`` (override, spot for stock, or BSM)."""
    if leg.premium is not None:
        return float(leg.premium)
    if leg.kind == "stock":
        return float(spot)
    return float(bs_price(spot, leg.strike, T, r, q, sigma, leg.kind))


def entry_cost(legs: list[Leg], spot: float, T: float, r: float, q: float, sigma: float) -> float:
    """Net debit (+) or credit (-) of opening the strategy."""
    return sum(l.qty * entry_price(l, spot, T, r, q, sigma) for l in legs)


def expiry_value(legs: list[Leg], s_grid: np.ndarray) -> np.ndarray:
    """Gross value of the strategy at expiry across a grid of terminal spots."""
    s = np.asarray(s_grid, dtype=float)
    total = np.zeros_like(s)
    for l in legs:
        if l.kind == "call":
            total += l.qty * np.maximum(s - l.strike, 0.0)
        elif l.kind == "put":
            total += l.qty * np.maximum(l.strike - s, 0.0)
        else:
            total += l.qty * s
    return total


def value_at(
    legs: list[Leg], s_grid: np.ndarray, T_remaining: float, r: float, q: float, sigma: float
) -> np.ndarray:
    """Theoretical (BSM) gross value of the strategy with ``T_remaining`` years left."""
    s = np.asarray(s_grid, dtype=float)
    total = np.zeros_like(s)
    for l in legs:
        if l.kind == "stock":
            total += l.qty * s
        else:
            total += l.qty * np.asarray(bs_price(s, l.strike, max(T_remaining, 0.0), r, q, sigma, l.kind))
    return total


def expiry_pnl(legs: list[Leg], s_grid: np.ndarray, cost: float) -> np.ndarray:
    """P&L at expiry = expiry value minus net entry cost."""
    return expiry_value(legs, s_grid) - cost


def pnl_at(
    legs: list[Leg], s_grid: np.ndarray, T_remaining: float, r: float, q: float, sigma: float, cost: float
) -> np.ndarray:
    """Theoretical P&L with ``T_remaining`` years left (``T_remaining=T`` => today)."""
    return value_at(legs, s_grid, T_remaining, r, q, sigma) - cost


def analyze(legs: list[Leg], spot: float, T: float, r: float, q: float, sigma: float) -> StrategySummary:
    """Exact breakevens, max profit and max loss at expiry.

    The expiry P&L is piecewise linear with kinks only at the strikes, so the
    extremes lie at ``S = 0`` or a strike, or are unbounded as ``S -> inf``
    (decided by the slope beyond the highest strike = long calls + stock).
    """
    if not legs:
        raise ValueError("strategy has no legs")
    cost = entry_cost(legs, spot, T, r, q, sigma)
    vertices = sorted({0.0, *[l.strike for l in legs if l.kind != "stock"]})
    pnl_v = [float(expiry_pnl(legs, np.array([v]), cost)[0]) for v in vertices]
    end_slope = sum(l.qty for l in legs if l.kind in ("call", "stock"))
    tol = 1e-9
    unlimited_profit = end_slope > tol
    unlimited_loss = end_slope < -tol
    max_profit = math.inf if unlimited_profit else max(pnl_v)
    max_loss = -math.inf if unlimited_loss else min(pnl_v)

    bes: list[float] = []

    def add(x: float) -> None:
        if x > 1e-9 and all(abs(x - b) > 1e-7 * max(1.0, x) for b in bes):
            bes.append(x)

    for (x0, p0), (x1, p1) in zip(zip(vertices, pnl_v), zip(vertices[1:], pnl_v[1:])):
        if abs(p0) < tol:
            if abs(p1) >= tol:  # skip flat-zero segments
                add(x0)
        elif p0 * p1 < 0:
            add(x0 + (x1 - x0) * (-p0) / (p1 - p0))
    if abs(pnl_v[-1]) < tol:
        if abs(end_slope) > tol:
            add(vertices[-1])
    elif abs(end_slope) > tol and pnl_v[-1] * end_slope < 0:
        add(vertices[-1] - pnl_v[-1] / end_slope)
    return StrategySummary(cost, max_profit, max_loss, tuple(sorted(bes)), unlimited_profit, unlimited_loss)


def round_to_step(x: float, step: float) -> float:
    """Round ``x`` to the nearest multiple of ``step``."""
    return round(x / step) * step


def preset_legs(name: str, spot: float, step: float, width_pct: float = 0.05) -> list[Leg]:
    """Build a standard strategy around ``spot`` with strikes snapped to ``step``.

    ``width_pct`` is the distance of the first OTM strikes from spot; iron
    condor wings sit at ``2 * width_pct``.
    """
    if name not in PRESET_NAMES:
        raise ValueError(f"unknown preset {name!r}; choose from {PRESET_NAMES}")
    atm = round_to_step(spot, step)
    up1 = round_to_step(spot * (1 + width_pct), step)
    dn1 = round_to_step(spot * (1 - width_pct), step)
    up2 = round_to_step(spot * (1 + 2 * width_pct), step)
    dn2 = round_to_step(spot * (1 - 2 * width_pct), step)
    if name == "Covered call":
        return [Leg("stock", 1), Leg("call", -1, up1)]
    if name == "Bull call spread":
        return [Leg("call", 1, atm), Leg("call", -1, up1)]
    if name == "Straddle":
        return [Leg("call", 1, atm), Leg("put", 1, atm)]
    if name == "Strangle":
        return [Leg("put", 1, dn1), Leg("call", 1, up1)]
    # Short iron condor: sell inner put/call, buy outer wings.
    return [Leg("put", 1, dn2), Leg("put", -1, dn1), Leg("call", -1, up1), Leg("call", 1, up2)]

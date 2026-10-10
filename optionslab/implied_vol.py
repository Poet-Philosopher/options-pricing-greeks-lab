"""Implied-volatility solver: Newton-Raphson with a Brent bisection fallback.

The solver never raises on bad market data; it returns an :class:`IVResult`
whose ``status`` and ``message`` explain exactly what went wrong (price below
intrinsic, above the no-arbitrage cap, no bracketing root, ...).
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .bsm import OptionKind, bs_greeks, bs_price, check_kind

SIGMA_FLOOR = 1e-6
SIGMA_CAP = 10.0  # 1000% vol


@dataclass(frozen=True)
class IVResult:
    """Outcome of an implied-volatility solve.

    Attributes:
        iv: Implied volatility as a decimal, or ``None`` if no solution.
        converged: ``True`` when ``iv`` reprices the option within tolerance.
        method: ``"newton"``, ``"brentq"`` (fallback), ``"bound"`` or ``"none"``.
        iterations: Iterations used by the method that produced the answer.
        status: Machine-readable code (``ok``, ``below_intrinsic``, ...).
        message: Plain-English explanation.
        warnings: Non-fatal cautions (e.g. price almost insensitive to vol).
    """

    iv: float | None
    converged: bool
    method: str
    iterations: int
    status: str
    message: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float, kind: OptionKind) -> tuple[float, float]:
    """Return ``(lower, upper)`` no-arbitrage price bounds for a European option."""
    check_kind(kind)
    pv_s, pv_k = S * math.exp(-q * T), K * math.exp(-r * T)
    if kind == "call":
        return max(pv_s - pv_k, 0.0), pv_s
    return max(pv_k - pv_s, 0.0), pv_k


def _fail(status: str, message: str) -> IVResult:
    return IVResult(None, False, "none", 0, status, message)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kind: OptionKind = "call",
    tol: float = 1e-10,  # relative price tolerance
    max_iter: int = 50,
) -> IVResult:
    """Solve ``BS(sigma) = price`` for ``sigma``.

    Strategy: Newton-Raphson from a Brenner-Subrahmanyam style seed, using
    analytic vega. If vega vanishes, a step leaves ``[SIGMA_FLOOR, SIGMA_CAP]``,
    or Newton fails to converge, fall back to Brent's bracketing method on the
    full range (guaranteed to converge when a root is bracketed).
    """
    check_kind(kind)
    vals = (price, S, K, T, r, q)
    if any(v is None or not math.isfinite(v) for v in vals):
        return _fail("invalid_input", "All inputs must be finite numbers.")
    if S <= 0 or K <= 0 or T <= 0:
        return _fail("invalid_input", "Spot, strike and time to expiry must be positive.")
    if price <= 0:
        return _fail("non_positive_price", "Option price must be positive; a zero or negative premium has no implied volatility.")

    lower, upper = no_arbitrage_bounds(S, K, T, r, q, kind)
    slack = 1e-9 * max(1.0, S)
    if price < lower - slack:
        return _fail(
            "below_intrinsic",
            f"Price {price:.4f} is below the discounted intrinsic value {lower:.4f}. "
            "No volatility can produce it (arbitrage / stale or mis-entered quote).",
        )
    if price > upper + slack:
        return _fail(
            "above_upper_bound",
            f"Price {price:.4f} exceeds the no-arbitrage upper bound {upper:.4f} "
            f"({'spot' if kind == 'call' else 'discounted strike'}). Check the quote.",
        )

    def f(sig: float) -> float:
        return float(bs_price(S, K, T, r, q, sig, kind)) - price

    f_lo, f_hi = f(SIGMA_FLOOR), f(SIGMA_CAP)
    if f_lo >= 0:
        if abs(f_lo) <= max(tol, slack):
            return IVResult(SIGMA_FLOOR, True, "bound", 0, "at_lower_bound",
                            "Price equals discounted intrinsic value: time value is zero, so IV is effectively 0.")
        return _fail("iv_below_floor", f"Implied volatility is below {SIGMA_FLOOR:.0e}: the price is essentially intrinsic value with no time value.")
    if f_hi < 0:
        return _fail("iv_above_cap", f"Implied volatility exceeds {SIGMA_CAP:.0%}; the price is too close to the upper bound to solve reliably.")

    # --- Newton-Raphson ---------------------------------------------------
    seed = math.sqrt(2.0 * math.pi / T) * price / (S * math.exp(-q * T))
    sigma = min(max(seed, 0.05), 3.0)
    # Relative price tolerance (an absolute one would accept any sigma for tiny OTM prices),
    # floored at the floating-point noise level of the pricing formula.
    ptol = max(tol * price, 1e-14 * (S + K))
    newton_ok = False
    iters = 0
    for iters in range(1, max_iter + 1):
        diff = f(sigma)
        if abs(diff) < ptol:
            newton_ok = True
            break
        vega = float(bs_greeks(S, K, T, r, q, sigma, kind).vega)
        if not math.isfinite(vega) or vega < 1e-10:
            break
        new = sigma - diff / vega
        if not math.isfinite(new) or new <= SIGMA_FLOOR or new >= SIGMA_CAP:
            break
        if abs(new - sigma) < 1e-14:
            sigma = new
            newton_ok = abs(f(sigma)) < 100 * ptol
            break
        sigma = new
    method = "newton"
    if not newton_ok:
        try:
            root, info = brentq(f, SIGMA_FLOOR, SIGMA_CAP, xtol=1e-14, rtol=1e-13, maxiter=500, full_output=True)
        except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
            return _fail("no_convergence", f"Neither Newton-Raphson nor bisection converged ({exc}).")
        if not info.converged:  # pragma: no cover - defensive
            return _fail("no_convergence", "Bisection fallback did not converge within its iteration limit.")
        sigma, iters, method = float(root), int(info.iterations), "brentq"

    warns: list[str] = []
    vega_sol = float(bs_greeks(S, K, T, r, q, sigma, kind).vega)
    if vega_sol < 1e-4:
        warns.append("Vega is tiny at the solution (deep ITM/OTM or very short expiry): the price barely depends on volatility, so the IV is poorly determined.")
    if (price - lower) < 1e-6 * max(1.0, S):
        warns.append("Almost all of the price is intrinsic value; the IV is very sensitive to rounding in the quote.")
    msg = f"Converged via {'Newton-Raphson' if method == 'newton' else 'Brent bisection fallback'} in {iters} iteration(s)."
    return IVResult(float(sigma), True, method, iters, "ok", msg, tuple(warns))


def implied_vol_smile(
    strikes: list[float] | np.ndarray,
    prices: list[float] | np.ndarray,
    kinds: list[str] | None,
    S: float,
    T: float,
    r: float,
    q: float,
) -> pd.DataFrame:
    """Solve IV for many ``(strike, price, kind)`` quotes sharing one expiry.

    Returns a DataFrame with columns ``strike, price, kind, iv, moneyness
    (K/S), method, status, message``; failed rows have ``iv = NaN``.
    """
    strikes = list(strikes)
    prices = list(prices)
    kinds = list(kinds) if kinds is not None else ["call"] * len(strikes)
    rows = []
    for K, p, k in zip(strikes, prices, kinds):
        res = implied_vol(p, S, K, T, r, q, k)  # type: ignore[arg-type]
        rows.append({
            "strike": K, "price": p, "kind": k,
            "iv": res.iv if res.iv is not None else np.nan,
            "moneyness": K / S, "method": res.method, "status": res.status,
            "message": res.message + (" " + " ".join(res.warnings) if res.warnings else ""),
        })
    return pd.DataFrame(rows)


def parse_quotes(text: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse pasted quotes: one ``strike, price[, call|put]`` per line.

    Separators may be commas, semicolons, tabs or spaces. Header lines and
    blank lines are skipped. Returns ``(DataFrame[strike, price, kind], errors)``.
    """
    rows: list[dict] = []
    errors: list[str] = []
    for n, raw in enumerate(io.StringIO(text), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in re.split(r"[,;\t ]+", line) if p]
        try:
            strike, price = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            if n == 1 or not rows:
                continue  # treat as header
            errors.append(f"Line {n}: could not read strike and price from {line!r}.")
            continue
        kind = "call"
        if len(parts) >= 3:
            k = parts[2].lower()
            if k in ("c", "call", "ce"):
                kind = "call"
            elif k in ("p", "put", "pe"):
                kind = "put"
            else:
                errors.append(f"Line {n}: unknown option type {parts[2]!r} (use call/put).")
                continue
        if strike <= 0 or price <= 0:
            errors.append(f"Line {n}: strike and price must be positive.")
            continue
        rows.append({"strike": strike, "price": price, "kind": kind})
    return pd.DataFrame(rows, columns=["strike", "price", "kind"]), errors


def synthetic_smile_example(
    S: float, T: float, r: float, q: float, strikes: list[float], base_vol: float = 0.16,
    skew: float = -0.25, curvature: float = 1.2,
) -> pd.DataFrame:
    """Generate a *synthetic* smile (OTM options priced from a made-up vol curve).

    ``vol(K) = base_vol + skew*x + curvature*x^2`` with ``x = ln(K/F)``. This is
    demonstration data produced by code -- it is **not** market data.
    """
    fwd = S * math.exp((r - q) * T)
    rows = []
    for K in strikes:
        x = math.log(K / fwd)
        vol = max(base_vol + skew * x + curvature * x * x, 0.02)
        kind = "put" if K < fwd else "call"
        rows.append({"strike": K, "price": round(float(bs_price(S, K, T, r, q, vol, kind)), 2), "kind": kind})
    return pd.DataFrame(rows)

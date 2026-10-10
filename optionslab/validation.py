"""Input validation shared by the library and the UI (no Streamlit imports)."""
from __future__ import annotations

import math


def validate_market_inputs(
    S: float, K: float, T: float, r: float, q: float, sigma: float | None = None
) -> list[str]:
    """Return a list of human-readable problems with the inputs (empty if valid).

    Rates and volatility are decimals (0.05 = 5%); ``T`` is in years.
    """
    problems: list[str] = []
    named = {"spot": S, "strike": K, "time to expiry": T, "risk-free rate": r, "dividend yield": q}
    if sigma is not None:
        named["volatility"] = sigma
    for name, value in named.items():
        if value is None or not math.isfinite(value):
            problems.append(f"{name.capitalize()} must be a finite number.")
    if problems:
        return problems
    if S <= 0:
        problems.append("Spot price must be positive.")
    if K <= 0:
        problems.append("Strike must be positive.")
    if T <= 0:
        problems.append("Time to expiry must be positive (use at least 1 day).")
    if sigma is not None and sigma <= 0:
        problems.append("Volatility must be positive.")
    if sigma is not None and sigma > 5:
        problems.append("Volatility above 500% is outside the supported range.")
    if not -0.5 <= r <= 1.0:
        problems.append("Risk-free rate should be between -50% and 100%.")
    if not 0.0 <= q <= 1.0:
        problems.append("Dividend yield should be between 0% and 100%.")
    return problems

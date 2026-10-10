"""Market presets, including an India / Nifty 50 index-option preset.

IMPORTANT: every number below is an **illustrative, editable assumption** --
not market data. Nothing here is fetched from any exchange or data vendor.
Users should overwrite spot, rate, dividend yield and volatility with their own
current figures. Contract specifications (lot size, strike interval) are set by
NSE and are revised periodically -- verify them on nseindia.com.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketPreset:
    """A named set of default inputs (all editable in the UI)."""

    name: str
    spot: float
    strike: float
    days_to_expiry: int
    rate_pct: float
    div_yield_pct: float
    vol_pct: float
    strike_step: float
    lot_size: int | None
    currency: str
    notes: str


GENERIC = MarketPreset(
    name="Generic equity",
    spot=100.0, strike=100.0, days_to_expiry=90, rate_pct=5.0, div_yield_pct=1.0, vol_pct=20.0,
    strike_step=5.0, lot_size=None, currency="",
    notes="Textbook-style defaults. All values are assumptions.",
)

NIFTY50 = MarketPreset(
    name="Nifty 50 (India)",
    spot=25000.0, strike=25000.0, days_to_expiry=30, rate_pct=6.5, div_yield_pct=1.3, vol_pct=14.0,
    strike_step=50.0, lot_size=65, currency="INR",
    notes=(
        "ASSUMPTIONS, not market data: spot is a round placeholder; the rate is an indicative "
        "Indian short-term rate; the dividend yield is a rough index-level assumption; the vol is a "
        "placeholder in the range India VIX has historically traded. Nifty options are European-style "
        "and cash-settled, so Black-Scholes-Merton is a natural fit. Lot size 65 was the NSE figure "
        "reported from the January 2026 series and is editable - verify the current lot size and strike "
        "interval on NSE before relying on them."
    ),
)

PRESETS: dict[str, MarketPreset] = {p.name: p for p in (GENERIC, NIFTY50)}

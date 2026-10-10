"""optionslab: a small, dependency-light options pricing library.

Pure Python + NumPy/SciPy/pandas. No Streamlit or network imports, so every
function is unit-testable in isolation.
"""
from .binomial import CRRResult, crr_convergence, crr_price, crr_tree, early_exercise_premium
from .bsm import (
    DAYS_PER_YEAR, Greeks, bs_greeks, bs_price, d1_d2, put_call_parity_gap,
)
from .implied_vol import (
    IVResult, implied_vol, implied_vol_smile, no_arbitrage_bounds, parse_quotes, synthetic_smile_example,
)
from .india import GENERIC, NIFTY50, PRESETS, MarketPreset
from .monte_carlo import (
    MCResult, convergence_series, geometric_asian_price, mc_asian_arithmetic, mc_european, simulate_gbm_paths,
)
from .strategies import (
    PRESET_NAMES, Leg, StrategySummary, analyze, entry_cost, entry_price, expiry_pnl, expiry_value,
    pnl_at, preset_legs, round_to_step, value_at,
)
from .validation import validate_market_inputs

__version__ = "1.0.0"
__all__ = [n for n in dir() if not n.startswith("_")]

"""Copy shared across pages: one-line Greek explanations and the disclaimer."""

GREEK_HELP = {
    "Delta": "Change in option value for a 1-unit move in the underlying; also the share-equivalent hedge ratio.",
    "Gamma": "Change in delta for a 1-unit move in the underlying (convexity) - peaks at-the-money, rising into expiry.",
    "Vega": "Change in option value for a 1-percentage-point rise in implied volatility.",
    "Theta": "Option value lost per calendar day from the passage of time, all else equal.",
    "Rho": "Change in option value for a 1-percentage-point rise in the risk-free rate.",
}
GREEK_UNITS = {
    "Delta": "per 1 unit of spot",
    "Gamma": "delta per 1 unit of spot",
    "Vega": "per 1 vol point",
    "Theta": "per calendar day",
    "Rho": "per 1 rate point",
}
DISCLAIMER = (
    "Educational tool only - **not investment advice**. Models make simplifying assumptions "
    "(constant volatility and rates, no transaction costs, taxes or margin). All default inputs "
    "are illustrative, editable assumptions, not market data."
)
SIDEBAR_NOTE = (
    "Defaults are **editable assumptions**, not live market data. Time is measured Act/365."
)

"""Methodology: formulas, conventions, algorithms and limitations."""
import streamlit as st

from app_utils import ui

ui.setup("Methodology", "📚", "Models, formulas, numerical methods and their limits.")

st.markdown("### 1. Black-Scholes-Merton (European options)")
st.markdown("Underlying follows geometric Brownian motion with continuous dividend yield $q$, constant volatility $\\sigma$ "
            "and a constant continuously compounded rate $r$ (risk-neutral measure):")
st.latex(r"dS_t = (r-q)\,S_t\,dt + \sigma S_t\,dW_t")
st.latex(r"d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad d_2=d_1-\sigma\sqrt T")
st.latex(r"C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2),\qquad P = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1)")
st.latex(r"C - P = S e^{-qT} - K e^{-rT}\quad\text{(put-call parity)}")

st.markdown("### 2. Greeks")
st.latex(r"\Delta_C = e^{-qT}N(d_1),\quad \Delta_P = -e^{-qT}N(-d_1),\quad \Gamma = \frac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt T},\quad \mathcal V = S e^{-qT}\varphi(d_1)\sqrt T")
st.latex(r"\Theta_C = -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt T} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)")
st.latex(r"\Theta_P = -\frac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt T} + rKe^{-rT}N(-d_2) - qSe^{-qT}N(-d_1)")
st.latex(r"\rho_C = KTe^{-rT}N(d_2),\qquad \rho_P = -KTe^{-rT}N(-d_2)")
st.markdown("**Display units:** vega per 1 volatility point (÷100), theta per calendar day (÷365), rho per 1 rate point (÷100). "
            "The library returns raw calculus units; tests verify every Greek against central finite differences of the price.")

st.markdown("### 3. Implied volatility")
st.markdown("Solve $\\mathrm{BS}(\\sigma)=\\text{price}$. Price is strictly increasing in $\\sigma$ (vega > 0), so the root is unique when it exists, "
            "which requires the price to lie strictly between the discounted-intrinsic lower bound and the upper bound ($Se^{-qT}$ for calls, $Ke^{-rT}$ for puts).")
st.latex(r"\sigma_{n+1}=\sigma_n-\frac{\mathrm{BS}(\sigma_n)-\text{price}}{\mathcal V(\sigma_n)}")
st.markdown("Newton-Raphson is fast (quadratic) but can overshoot when vega is small, so the solver **falls back to Brent's method** "
            "(bisection + secant + inverse quadratic interpolation) on $[10^{-6},10]$. Convergence is judged on a *relative* price tolerance; "
            "a vega-based warning flags quotes whose IV is poorly determined (deep ITM/OTM, tiny time value).")

st.markdown("### 4. Strategy payoff analytics")
st.markdown("Each leg's expiry value is $q\\max(\\pm(S_T-K),0)$ (or $qS_T$ for stock); P&L subtracts the net entry cost "
            "(theoretical BSM price unless overridden). Because the expiry P&L is **piecewise linear** with kinks only at strikes, "
            "extremes occur at $S=0$ or a strike, or are unbounded as $S\\to\\infty$ (decided by the slope beyond the highest strike). "
            "Breakevens are the exact zero-crossings of each linear segment - no grid search.")

st.markdown("### 5. Monte Carlo")
st.latex(r"S_T = S_0\exp\!\Big[(r-q-\tfrac12\sigma^2)T+\sigma\sqrt T\,Z\Big],\quad Z\sim N(0,1)")
st.latex(r"\hat V = e^{-rT}\frac1n\sum_{i=1}^n f(S_T^{(i)}),\qquad \mathrm{SE}=e^{-rT}\frac{s}{\sqrt n},\qquad \text{95\% CI}=\hat V\pm1.96\,\mathrm{SE}")
st.markdown("**Antithetic variates:** each draw $Z$ is paired with $-Z$; the estimator averages the two payoffs and the "
            "standard error is computed on the *pair averages* (treating the 2n correlated paths as independent would understate the error). "
            "**Asian option:** payoff on the arithmetic average of $m$ discretely monitored fixings, simulated path-by-path in memory-bounded chunks. "
            "**Control variate:** the geometric-average Asian has a closed form,")
st.latex(r"\ln G\sim N(\mu_G,\sigma_G^2),\ \ \mu_G=\ln S+(r-q-\tfrac12\sigma^2)T\tfrac{m+1}{2m},\ \ \sigma_G^2=\sigma^2T\tfrac{(m+1)(2m+1)}{6m^2}")
st.markdown("and the estimator $X-\\beta\\,(C-\\mathbb E[C])$, with $\\beta=\\mathrm{Cov}(X,C)/\\mathrm{Var}(C)$, removes most of the variance because the two payoffs are highly correlated.")

st.markdown("### 6. Cox-Ross-Rubinstein binomial tree")
st.latex(r"u=e^{\sigma\sqrt{\Delta t}},\quad d=1/u,\quad p=\frac{e^{(r-q)\Delta t}-d}{u-d},\quad V_{i,j}=e^{-r\Delta t}\big[pV_{i+1,j+1}+(1-p)V_{i+1,j}\big]")
st.markdown("For American options each node takes $\\max(V_{i,j},\\ \\text{intrinsic}_{i,j})$. The early-exercise premium is the American minus the "
            "European value on the same tree. An American call on a non-dividend-paying asset is never exercised early; an American put can be, "
            "because exercising earns interest on the strike. Tree delta and gamma are read from the first two layers.")

st.markdown("### 7. India context")
st.markdown("The Nifty 50 preset is a convenience. **Every number is an editable assumption, not market data**: spot is a round placeholder, "
            "the rate, dividend yield and volatility are indicative, and the lot size reflects the NSE figure reported from the January 2026 series (verify on nseindia.com; "
            "NSE revises these periodically). Nifty index options are European-style and cash-settled, so BSM is appropriate; "
            "no trading costs, STT, margins, or the forward-vs-spot basis are modelled.")

st.markdown("### 8. Limitations")
st.markdown(
    "- Constant volatility and rates: real markets exhibit a volatility **smile/skew** and stochastic rates (see the smile view).\n"
    "- Continuous dividend yield only; discrete dividends, borrow costs and early-exercise of dividend-paying calls are not modelled in BSM.\n"
    "- Log-normal returns: no jumps or fat tails.\n"
    "- Strategy P&L ignores transaction costs, taxes, bid-ask spreads and margin.\n"
    "- Monte Carlo results are statistical estimates with sampling error; always read the standard error.")

st.markdown("### References")
st.markdown("- Black & Scholes (1973); Merton (1973) - option pricing with dividends.\n"
            "- Cox, Ross & Rubinstein (1979) - binomial option pricing.\n"
            "- Kemna & Vorst (1990) - geometric Asian option and control variate.\n"
            "- Hull, *Options, Futures, and Other Derivatives* - benchmark values used in the test suite.\n"
            "- Glasserman, *Monte Carlo Methods in Financial Engineering* - antithetic and control variates.")
ui.footer()

---
name: vasicek-gauss-plus
description: Price, simulate, and calibrate the Vasicek one-factor and Gauss+ three-factor (two-source-of-risk) short-rate term-structure models from Tuckman & Serrat, *Fixed Income Securities, 4e*, Chapter 9 and Appendix A9. Use this skill whenever the user asks to price zero-coupon bonds, build a yield curve, decompose forwards into expectations / risk premium / convexity, simulate the short rate, calibrate to observed yields or volatilities, or extract risk premia under either of these models. Triggers include "Vasicek", "Gauss+", "Gauss plus", "cascade form", "mean-reverting short rate", "two-factor term structure", "affine term structure", and related queries about pricing or estimating mean-reverting Gaussian short-rate models.
---

# Vasicek and Gauss+ Term-Structure Models

Numerical and analytical toolkit for the two short-rate models presented in *Fixed Income Securities, 4e* (Tuckman & Serrat), Chapter 9 and Appendix A9. The toolkit handles both the classic Vasicek model and the cascade-form Gauss+ model with three factors and two sources of risk.

## When to use this skill

Reach for it whenever the user needs to:

- Price zero-coupon bonds analytically under Vasicek or Gauss+.
- Compute model spot and forward curves and their decomposition into expectations, risk premium, and convexity.
- Simulate short-rate paths (Euler discretization or exact Vasicek transition).
- Build a recombining binomial tree for Vasicek (Appendix A9.1 construction).
- Calibrate Vasicek (k, θ, σ, r₀, λ) to a target curve, ATM volatility, or 10y rate level.
- Calibrate the Gauss+ parameter vector P = (αr, αm, αl, σm, σl, ρ, μ) in the staged procedure of Appendix A9.2.2.
- Extract Gauss+ factors (mt, lt) by exact fit at the 2y and 10y forwards.
- Compute the implied price of risk λt (Appendix A9.2.3) and decompose forwards into expected short rate, risk premium, and convexity.

If a related question is conceptual ("what does αr mean?", "why does Gauss+ have a hump in volatility?"), answer from the reference content first, then offer code.

## File layout

The skill ships with the following files inside this directory. Read them on demand — do not load everything up front.

- `SKILL.md` — this file. Index and decision rules.
- `vasicek.py` — analytic and numerical Vasicek implementation: `E[rt]`, `V[rt]`, forward `f(t)`, spot `r̂(t)`, zero-coupon bond price, Monte Carlo, binomial tree, calibration helpers.
- `gauss_plus.py` — Gauss+ implementation in cascade form: A(α), Ω(σ), B(τ,α), C(τ,α,σ), Υ(τ,α), yields, forwards, factor extraction, staged calibration, λt extraction.
- `reference.md` — full derivations from Appendix A9 (Vasicek tree construction, Gauss+ reduced/cascade-form derivation, yield/forward affine representation, staged estimation, implied risk premia).
- `examples/` — runnable examples:
  - **Textbook scenarios** (using known parameters):
    - `vasicek_textbook.py` — reproduces Figures 9.1–9.4 numbers (r₀=2%, θ=11%, k=0.0165, σ=0.95%).
    - `gauss_plus_textbook.py` — uses the Table 9.1 parameters to build yield and forward curves and to compute the implied risk-premium decomposition.
  - **Calibration examples** (fitting parameters to data):
    - `vasicek_calibration.py` — demonstrates three calibration workflows: unweighted yield-curve fit, weighted fit (short-rate emphasis), and sigma-only calibration.
    - `gauss_plus_calibration.py` — three-stage cascade-form calibration on synthetic SDE-generated yield data with visualization of alpha, sigma, and mu convergence and factor loadings.

## How to use

### Vasicek

```python
from vasicek import Vasicek

m = Vasicek(r0=0.02, k=0.0165, theta=0.11, sigma=0.0095)
m.spot_rate(10)         # continuously-compounded 10y zero yield
m.forward_rate(10)      # instantaneous forward at t=10
m.bond_price(10)        # P(0, 10)
m.expected_rate(10)     # E[r_10]
m.rate_variance(10)
paths = m.simulate(T=10, n_steps=120, n_paths=10_000, seed=0)
tree  = m.binomial_tree(n_steps=3, dt=1.0)   # Appendix A9.1
```

For real-world (P-measure) dynamics with risk premium λ:

```python
m = Vasicek(r0=0.02, k=0.0165, theta=0.11, sigma=0.0095, risk_premium=0.00125)
m.decompose_forward(t=10)   # returns (expectation, risk_premium, convexity)
```

### Gauss+

```python
from gauss_plus import GaussPlus, TABLE_9_1

g = GaussPlus(**TABLE_9_1)           # use textbook estimates
g.set_factors(r=0.0008, m=0.01, l=0.02)
g.yield_curve(taus=[1, 2, 5, 10, 20, 30])
g.forward_curve(taus=[0, 1, 2, 5, 10], tenor=1.0)
g.loadings(tau=10)                   # Υ(10, α) for (r, m, l)

# Extract (m, l) so the model matches the observed 2y/10y forwards exactly.
g.fit_factors(observed_forwards={2: 0.0125, 10: 0.022}, tenor=1.0, r_obs=0.0008)

# Implied price of risk and forward decomposition (Appendix A9.2.3).
lam_t = g.implied_lambda(f_tau=0.024, f_tau_prime=0.0245, tau=14, tau_prime=15, tenor=1.0)
g.decompose_forward(tau=10, tenor=1.0, lambda_t=lam_t)
```

### Calibration

```python
# Vasicek: fit (k, theta, sigma) to a yield curve.
from vasicek import calibrate_vasicek
params = calibrate_vasicek(taus=taus_array, market_yields=y_array, r0=0.02)

# Gauss+: staged estimator from Appendix A9.2.2.
from gauss_plus import estimate_gauss_plus
fit = estimate_gauss_plus(yields_array, taus_array, short_rate_series, decay=0.8)
```

## Decision tree for picking the model

- One factor is enough, problem is long-dated, simple analytics matter → Vasicek.
- Need to match a humped term structure of volatility, fit short and long ends jointly, or do relative-value trading vs. the 2y and 10y → Gauss+.
- User mentions a binomial tree → Vasicek (Appendix A9.1). Gauss+ in this toolkit is solved analytically (cascade form) and via Monte Carlo, not by tree.

## Conventions

- All rates, volatilities, and parameters are **annualized** and quoted as decimals (e.g., 95 bps/yr is `sigma=0.0095`, not `95`).
- All `tau` arguments are in years.
- Bond prices are unit-face, continuously compounded.
- Mean-reversion parameters `k`, `αr`, `αm`, `αl` are in 1/year.
- The Gauss+ short rate `r` is treated as observed (fed funds target style) — `set_factors` and `fit_factors` both require it.

## Sources

Tuckman, B. and Serrat, A. *Fixed Income Securities: Tools for Today's Markets*, 4th ed., Chapter 9 "The Vasicek and Gauss+ Models" and Appendix A9 "The Vasicek and Gauss+ Models". All formulas in this skill are referenced by their book equation number (e.g., Eq. 9.4, A9.13).

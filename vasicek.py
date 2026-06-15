"""Vasicek one-factor short-rate model.

Implements the formulas from Chapter 9 (Eqs. 9.1-9.8) and the binomial-tree
construction from Appendix A9.1 of Tuckman & Serrat, Fixed Income Securities, 4e.

Risk-neutral dynamics:
    dr = k * (theta - r) * dt + sigma * dW

Real-world dynamics with a constant risk premium lambda (Eq. 9.2-9.3):
    dr = k * (r_inf - r) * dt + lambda * dt + sigma * dW
where theta = r_inf + lambda / k.

All rates and parameters are annualized decimals; all maturities in years.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class Vasicek:
    r0: float
    k: float
    theta: float
    sigma: float
    risk_premium: float = 0.0  # lambda, in absolute rate units per year

    @property
    def r_inf(self) -> float:
        """Long-run mean under the real-world measure (Eq. 9.3)."""
        return self.theta - self.risk_premium / self.k

    def expected_rate(self, t: float) -> float:
        """E[r_t] under the risk-neutral measure (Eq. 9.4)."""
        return self.r0 * np.exp(-self.k * t) + self.theta * (1 - np.exp(-self.k * t))

    def rate_variance(self, t: float) -> float:
        """Var[r_t] (Eq. 9.5)."""
        return self.sigma**2 * (1 - np.exp(-2 * self.k * t)) / (2 * self.k)

    def rate_std(self, t: float) -> float:
        return np.sqrt(self.rate_variance(t))

    def forward_rate(self, t: float) -> float:
        """Instantaneous continuously-compounded forward rate (Eq. 9.6)."""
        k, theta, sigma, r0 = self.k, self.theta, self.sigma, self.r0
        convexity = (sigma**2) / (2 * k**2) * (1 + np.exp(-2 * k * t) - 2 * np.exp(-k * t))
        return theta + np.exp(-k * t) * (r0 - theta) - convexity

    def spot_rate(self, t: float) -> float:
        """Continuously-compounded zero yield (Eq. 9.7)."""
        if t == 0:
            return self.r0
        k, theta, sigma, r0 = self.k, self.theta, self.sigma, self.r0
        decay = (1 - np.exp(-k * t)) / (k * t)
        convex = (sigma**2) / (2 * k**2) * (
            1 + (1 - np.exp(-2 * k * t)) / (2 * k * t) - 2 * (1 - np.exp(-k * t)) / (k * t)
        )
        return theta + decay * (r0 - theta) - convex

    def bond_price(self, t: float) -> float:
        """P(0, t) = exp(-spot_rate(t) * t)."""
        return float(np.exp(-self.spot_rate(t) * t))

    def forward_volatility(self, t: float) -> float:
        """Instantaneous volatility of the t-maturity forward, sigma * exp(-k * t)."""
        return self.sigma * np.exp(-self.k * t)

    def half_life(self) -> float:
        """Half-life of a shock to r (Eq. 9.8)."""
        return np.log(2) / self.k

    # ---------------------------------------------------------------- decompose
    def decompose_forward(self, t: float) -> dict:
        """Decompose f(t) into expectation, risk premium, and convexity.

        Under the real-world measure with constant risk premium lambda:
            E_P[r_t]    = r_inf + e^{-kt} (r0 - r_inf)
            risk prem   = (lambda / k) * (1 - e^{-kt})
            convexity   = -(sigma^2 / (2 k^2)) (1 - e^{-kt})^2

        The three pieces sum to the model forward rate (Eq. 9.6) with theta = r_inf + lambda/k.
        Reproduces Figure 9.4 of the text.
        """
        k, sigma = self.k, self.sigma
        decay = 1 - np.exp(-k * t)
        expectation = self.r_inf + np.exp(-k * t) * (self.r0 - self.r_inf)
        risk_prem = (self.risk_premium / k) * decay
        convexity = -(sigma**2) / (2 * k**2) * decay**2
        return {
            "expectation": float(expectation),
            "risk_premium": float(risk_prem),
            "convexity": float(convexity),
            "forward": float(expectation + risk_prem + convexity),
        }

    # ----------------------------------------------------------- calibration
    def calibrate(self, taus: np.ndarray, market_yields: np.ndarray,
                 weights: Optional[np.ndarray] = None,
                 x0: Optional[tuple] = None) -> dict:
        """Calibrate (k, theta, sigma) to a yield curve.

        Fits the model to observed yields by least-squares optimization.

        taus : array of maturities (years).
        market_yields : array of observed zero-coupon yields (decimals).
        weights : optional weights for each maturity (default: uniform).
        x0 : initial guess (k, theta, sigma). Default [0.05, 0.04, 0.01].

        Returns a dict with calibrated k, theta, sigma and RSS.
        """
        taus = np.asarray(taus, dtype=float)
        yields = np.asarray(market_yields, dtype=float)
        if weights is None:
            weights = np.ones_like(taus)
        else:
            weights = np.asarray(weights, dtype=float)

        if x0 is None:
            x0 = (0.05, 0.04, 0.01)

        def loss(params):
            k, theta, sigma = params
            if k <= 0 or sigma <= 0:
                return 1e6
            temp_v = Vasicek(r0=self.r0, k=k, theta=theta, sigma=sigma,
                            risk_premium=self.risk_premium)
            model = np.array([temp_v.spot_rate(t) for t in taus])
            return float(np.sum(weights * (model - yields) ** 2))

        res = minimize(loss, x0=x0, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-12})
        self.k, self.theta, self.sigma = res.x
        return {
            "k": float(self.k),
            "theta": float(self.theta),
            "sigma": float(self.sigma),
            "rss": float(res.fun),
            "success": bool(res.success),
        }

    def calibrate_sigma(self, taus: np.ndarray, market_yields: np.ndarray,
                       weights: Optional[np.ndarray] = None,
                       x0: Optional[float] = None) -> float:
        """Calibrate sigma while holding k and theta fixed.

        Fits volatility to minimize yield curve fit errors.

        taus : array of maturities (years).
        market_yields : array of observed zero-coupon yields (decimals).
        weights : optional weights for each maturity (default: uniform).
        x0 : initial guess for sigma. Default 0.01.

        Returns the calibrated sigma value.
        """
        taus = np.asarray(taus, dtype=float)
        yields = np.asarray(market_yields, dtype=float)
        if weights is None:
            weights = np.ones_like(taus)
        else:
            weights = np.asarray(weights, dtype=float)

        if x0 is None:
            x0 = 0.01

        def loss(sigma_arr):
            sigma = float(sigma_arr[0])
            if sigma <= 0:
                return 1e6
            temp_v = Vasicek(r0=self.r0, k=self.k, theta=self.theta,
                            sigma=sigma, risk_premium=self.risk_premium)
            model = np.array([temp_v.spot_rate(t) for t in taus])
            return float(np.sum(weights * (model - yields) ** 2))

        res = minimize(loss, x0=[x0], method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-12})
        self.sigma = float(res.x[0])
        return self.sigma

    # ----------------------------------------------------------- Monte Carlo
    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None,
        exact: bool = True,
    ) -> np.ndarray:
        """Simulate short-rate paths under the risk-neutral measure.

        If exact=True, samples from the exact Gaussian transition
        (zero discretization error for the marginal). Otherwise uses Euler.
        Returns an (n_paths, n_steps + 1) array of rates including t=0.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        r = np.empty((n_paths, n_steps + 1))
        r[:, 0] = self.r0
        if exact:
            mean_decay = np.exp(-self.k * dt)
            cond_var = self.sigma**2 * (1 - np.exp(-2 * self.k * dt)) / (2 * self.k)
            cond_std = np.sqrt(cond_var)
            for i in range(n_steps):
                z = rng.standard_normal(n_paths)
                r[:, i + 1] = r[:, i] * mean_decay + self.theta * (1 - mean_decay) + cond_std * z
        else:
            sqrt_dt = np.sqrt(dt)
            for i in range(n_steps):
                z = rng.standard_normal(n_paths)
                r[:, i + 1] = (
                    r[:, i]
                    + self.k * (self.theta - r[:, i]) * dt
                    + self.sigma * sqrt_dt * z
                )
        return r

    def bond_price_mc(self, T: float, n_steps: int, n_paths: int, seed: Optional[int] = None) -> tuple[float, float]:
        """Monte Carlo bond price and standard error. Sanity-checks bond_price."""
        r = self.simulate(T, n_steps, n_paths, seed=seed)
        dt = T / n_steps
        # Trapezoidal integral of r along each path.
        integral = 0.5 * dt * (r[:, 0] + r[:, -1]) + dt * r[:, 1:-1].sum(axis=1)
        df = np.exp(-integral)
        return float(df.mean()), float(df.std(ddof=1) / np.sqrt(n_paths))

    # ----------------------------------------------------------- Binomial tree
    def binomial_tree(self, n_steps: int, dt: float = 1.0) -> dict:
        """Recombining binomial tree, Appendix A9.1.

        Returns a dict with:
            'rates'  : list of arrays, one per date, of rates at each state.
            'probs'  : list of (p_up, p_down) tuples per node, dates 0..n_steps-1.

        Construction (textbook):
            From a node with rate r and expected next-period rate r + k(theta-r)dt,
            create up/down nodes at r + k(theta-r)dt +/- sigma*sqrt(dt) on date 1.
            For subsequent dates, the "middle" expected rate is recomputed and
            up/down probabilities are solved so that drift and volatility match.
        """
        sqrt_dt = np.sqrt(dt)
        rates = [np.array([self.r0])]
        probs = []

        # Date 1: simple +/- sigma sqrt(dt) around expected drift.
        next_mean = self.r0 + self.k * (self.theta - self.r0) * dt
        rates.append(np.array([next_mean + self.sigma * sqrt_dt, next_mean - self.sigma * sqrt_dt]))
        probs.append([(0.5, 0.5)])  # one parent node, equal probs

        for date in range(1, n_steps):
            prev_rates = rates[-1]
            new_states = []
            new_probs_at_date = []
            # For each adjacent pair of parent nodes, build the shared child.
            for i in range(len(prev_rates)):
                r_i = prev_rates[i]
                drift = r_i + self.k * (self.theta - r_i) * dt  # E[r | r_i]
                if i == 0:
                    # leftmost parent: produce its "up" child (shared) and
                    # the rightmost "down" child of the leftmost only on first iteration
                    new_states.append(drift - self.sigma * sqrt_dt)  # dd of leftmost
                    new_states.append(drift + self.sigma * sqrt_dt)  # ud / shared with next
                else:
                    # next parent: only its "up" child is new
                    new_states.append(drift + self.sigma * sqrt_dt)
            rates.append(np.array(sorted(new_states, reverse=True)))

            # Solve probabilities at each parent so drift and vol match.
            new_layer_probs = []
            for i, r_i in enumerate(prev_rates):
                drift = r_i + self.k * (self.theta - r_i) * dt
                # children indices: parent i -> children [i, i+1] in sorted-descending order
                r_up = rates[-1][i]
                r_dn = rates[-1][i + 1]
                # p * r_up + (1-p) * r_dn = drift
                p = (drift - r_dn) / (r_up - r_dn)
                p = float(np.clip(p, 0.0, 1.0))
                new_layer_probs.append((p, 1 - p))
            probs.append(new_layer_probs)

        return {"rates": rates, "probs": probs, "dt": dt}


# ------------------------------------------------------------------ Calibration
def calibrate_vasicek(
    taus: np.ndarray,
    market_yields: np.ndarray,
    r0: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    x0: tuple = (0.05, 0.04, 0.01),
) -> dict:
    """Least-squares fit of (k, theta, sigma) to a yield curve.

    If r0 is None, it is fixed to market_yields[0]. Returns the fitted
    parameters and the residual sum of squares.
    """
    taus = np.asarray(taus, dtype=float)
    yields = np.asarray(market_yields, dtype=float)
    if r0 is None:
        r0 = float(yields[0])
    if weights is None:
        weights = np.ones_like(taus)

    def loss(params):
        k, theta, sigma = params
        if k <= 0 or sigma <= 0:
            return 1e6
        m = Vasicek(r0=r0, k=k, theta=theta, sigma=sigma)
        model = np.array([m.spot_rate(t) for t in taus])
        return float(np.sum(weights * (model - yields) ** 2))

    res = minimize(loss, x0=x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12})
    k_, theta_, sigma_ = res.x
    return {
        "r0": r0,
        "k": float(k_),
        "theta": float(theta_),
        "sigma": float(sigma_),
        "rss": float(res.fun),
        "success": bool(res.success),
    }

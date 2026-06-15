"""Gauss+ three-factor (two-source-of-risk) cascade-form short-rate model.

Reference: Tuckman & Serrat, *Fixed Income Securities*, 4e, Chapter 9.2 and
Appendix A9.2. Equation numbers in comments refer to that text.

Cascade-form dynamics (Eqs. 9.9-9.12; the textbook prints leading minus signs
that are inconsistent with mean-reversion — we use the equivalent positive form):

    dr_t = alpha_r (m_t - r_t) dt
    dm_t = alpha_m (l_t - m_t) dt + sigma_m (rho dW1 + sqrt(1-rho^2) dW2)
    dl_t = alpha_l (mu - l_t) dt + sigma_l dW1
    E[dW1 dW2] = 0

Reduced form (X_i mean-reverts to 0):
    dX = -diag(alpha) X dt + A(alpha)^{-1} Omega dW
    r_t = mu + 1' X_t,  x_t = A(alpha) X_t + mu * 1   (Eq. A9.5)

The text's eq. (A9.6) is heavily mis-rendered in the PDF text layer; the
matrix used here is the one consistent with A^{-1} in (A9.12) and with the
identity r = mu + 1' X:

    A[0, :] = (1, 1, 1)
    A[1, 1] = (alpha_r - alpha_m) / alpha_r
    A[1, 2] = (alpha_r - alpha_l) / alpha_r
    A[2, 2] = (alpha_r - alpha_l) (alpha_m - alpha_l) / (alpha_r alpha_m)

Affine yield (A9.13):
    y_t(tau) = mu (1 - Upsilon(tau, alpha) 1) - C(tau, alpha, sigma)
             + Upsilon(tau, alpha) x_t
    Upsilon(tau, alpha) = B(tau, alpha) A(alpha)^{-1}
    B_i(tau) = (1 - exp(-alpha_i tau)) / (alpha_i tau),  B_i(0) = 1.

For C we use the sign that is consistent with y(0) = r (so C(0) = 0) and
with the standard one-factor Vasicek convexity (text Eq. 9.7):
    C(tau, alpha, sigma) = sum_{i,j} Sigma_red[i,j] / (2 alpha_i alpha_j)
        * (1 - B_i(tau) - B_j(tau) + (1 - exp(-(alpha_i + alpha_j) tau))
                                       / ((alpha_i + alpha_j) tau))
    Sigma_red = A^{-1} Omega Omega' A^{-T}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


TABLE_9_1 = dict(
    alpha_r=1.0547,
    alpha_m=0.6358,
    alpha_l=0.0165,
    sigma_m=0.01092,
    sigma_l=0.00964,
    rho=0.212,
    mu=0.10555,
)


@dataclass
class GaussPlus:
    alpha_r: float
    alpha_m: float
    alpha_l: float
    sigma_m: float
    sigma_l: float
    rho: float
    mu: float
    r: float = 0.0
    m: float = 0.0
    l: float = 0.0

    # ---------------------------------------------------------------- matrices
    @property
    def alpha(self) -> np.ndarray:
        return np.array([self.alpha_r, self.alpha_m, self.alpha_l])

    def A(self) -> np.ndarray:
        ar, am, al = self.alpha_r, self.alpha_m, self.alpha_l
        a22 = (ar - al) * (am - al) / (ar * am)
        return np.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, (ar - am) / ar, (ar - al) / ar],
                [0.0, 0.0, a22],
            ]
        )

    def A_inv(self) -> np.ndarray:
        return np.linalg.inv(self.A())

    def Omega(self) -> np.ndarray:
        """Diffusion matrix of cascade-form factors x = (r, m, l).

        Two independent sources of risk; a zero third column is appended so
        that Omega is 3x3 (consistent with A9.8).
        """
        sm, sl, rho = self.sigma_m, self.sigma_l, self.rho
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [rho * sm, np.sqrt(max(1 - rho**2, 0.0)) * sm, 0.0],
                [sl, 0.0, 0.0],
            ]
        )

    def Sigma_red(self) -> np.ndarray:
        """Var(dX)/dt = A^{-1} Omega Omega' A^{-T}."""
        Ai = self.A_inv()
        Om = self.Omega()
        return Ai @ Om @ Om.T @ Ai.T

    # ---------------------------------------------------------- yields & forwards
    def B(self, tau: float) -> np.ndarray:
        """Vector of B_i(tau) = (1 - exp(-alpha_i tau)) / (alpha_i tau)."""
        if tau <= 0:
            return np.ones(3)
        a = self.alpha
        return (1 - np.exp(-a * tau)) / (a * tau)

    def C(self, tau: float) -> float:
        """C(tau, alpha, sigma), A9.11 (sign chosen for C(0) = 0)."""
        if tau <= 0:
            return 0.0
        a = self.alpha
        Sigma = self.Sigma_red()
        total = 0.0
        for i in range(3):
            for j in range(3):
                if Sigma[i, j] == 0:
                    continue
                Bi = (1 - np.exp(-a[i] * tau)) / (a[i] * tau)
                Bj = (1 - np.exp(-a[j] * tau)) / (a[j] * tau)
                aij = a[i] + a[j]
                third = (1 - np.exp(-aij * tau)) / (aij * tau)
                total += Sigma[i, j] / (2 * a[i] * a[j]) * (1 - Bi - Bj + third)
        return float(total)

    def upsilon(self, tau: float) -> np.ndarray:
        """Yield loadings Upsilon(tau, alpha) = B(tau) A^{-1}."""
        return self.B(tau) @ self.A_inv()

    def loadings(self, tau: float) -> Dict[str, float]:
        u = self.upsilon(tau)
        return {"r": float(u[0]), "m": float(u[1]), "l": float(u[2])}

    def yield_(self, tau: float, r: Optional[float] = None,
               m: Optional[float] = None, l: Optional[float] = None) -> float:
        """Zero-coupon yield (Eq. A9.13)."""
        x = np.array([
            r if r is not None else self.r,
            m if m is not None else self.m,
            l if l is not None else self.l,
        ])
        u = self.upsilon(tau)
        return float(self.mu * (1 - u.sum()) - self.C(tau) + u @ x)

    def bond_price(self, tau: float, **kw) -> float:
        return float(np.exp(-self.yield_(tau, **kw) * tau))

    def upsilon_prime(self, tau: float, tenor: float) -> np.ndarray:
        """Forward-rate loadings on x: ((tau+tenor) B(tau+tenor) - tau B(tau)) A^{-1} / tenor.

        Derived from forward * tenor = (tau+tenor) y(tau+tenor) - tau y(tau).
        The text's A9.16 prints (B(tau+tenor) - B(tau)) A^{-1} / tenor — the
        tau multipliers were dropped in the PDF typesetting.
        """
        b_long = (1 - np.exp(-self.alpha * (tau + tenor))) / self.alpha  # (tau+tenor) B(tau+tenor)
        b_short = (1 - np.exp(-self.alpha * tau)) / self.alpha            # tau B(tau)
        return (b_long - b_short) @ self.A_inv() / tenor

    def C_prime(self, tau: float, tenor: float) -> float:
        """C'(tau, alpha, sigma, tenor) = ((tau+tenor) C(tau+tenor) - tau C(tau)) / tenor.

        Same correction as upsilon_prime applied to A9.17.
        """
        return ((tau + tenor) * self.C(tau + tenor) - tau * self.C(tau)) / tenor

    def forward_rate(self, tau: float, tenor: float = 1.0,
                     r: Optional[float] = None,
                     m: Optional[float] = None,
                     l: Optional[float] = None) -> float:
        """Continuously-compounded forward rate (Eq. A9.15)."""
        x = np.array([
            r if r is not None else self.r,
            m if m is not None else self.m,
            l if l is not None else self.l,
        ])
        up = self.upsilon_prime(tau, tenor)
        cp = self.C_prime(tau, tenor)
        return float(self.mu * (1 - up.sum()) + up @ x - cp)

    def yield_curve(self, taus: Sequence[float]) -> Dict[float, float]:
        return {float(t): self.yield_(t) for t in taus}

    def forward_curve(self, taus: Sequence[float], tenor: float = 1.0) -> Dict[float, float]:
        return {float(t): self.forward_rate(t, tenor=tenor) for t in taus}

    # ------------------------------------------------- factor extraction
    def set_factors(self, r: float, m: float, l: float) -> None:
        self.r, self.m, self.l = float(r), float(m), float(l)

    def fit_factors(self, observed_forwards: Dict[float, float],
                    tenor: float = 1.0,
                    r_obs: Optional[float] = None) -> Dict[str, float]:
        """Solve for (m, l) so the model fits two observed forwards exactly.

        observed_forwards maps tau -> f_t(tau, tenor). Must contain exactly two
        entries. r is taken from r_obs (else self.r).
        """
        if len(observed_forwards) != 2:
            raise ValueError("Need exactly two observed forwards (e.g. {2: f2y, 10: f10y}).")
        if r_obs is not None:
            self.r = float(r_obs)
        taus = sorted(observed_forwards)

        # forward = mu (1 - sum(up)) + up @ x - cp.  Linear in x.
        rows = []
        rhs = []
        for tau in taus:
            up = self.upsilon_prime(tau, tenor)
            cp = self.C_prime(tau, tenor)
            const = self.mu * (1 - up.sum()) - cp
            # observed = const + up[0] r + up[1] m + up[2] l
            rhs.append(observed_forwards[tau] - const - up[0] * self.r)
            rows.append(up[1:])
        M = np.array(rows)
        b = np.array(rhs)
        m_l = np.linalg.solve(M, b)
        self.m, self.l = float(m_l[0]), float(m_l[1])
        return {"r": self.r, "m": self.m, "l": self.l}

    # ----------------------------------------------- calibration
    def calibrate_alpha(self, yields: np.ndarray, taus: Sequence[float],
                       benchmark_taus: Tuple[float, float] = (2.0, 10.0),
                       x0: Optional[Sequence[float]] = None) -> Dict[str, float]:
        """Stage 1: Calibrate mean-reversion speeds (alpha_r, alpha_m, alpha_l).

        Matches OLS yield-change slopes at all maturities to model-implied slopes.

        yields : T x N array of observed zero yields (decimals).
        taus : length-N maturities (years).
        benchmark_taus : maturities pinned exactly for OLS regression.
        x0 : initial guess [alpha_r, alpha_m, alpha_l]. Default [1.0, 0.5, 0.02].
        """
        yields = np.asarray(yields, dtype=float)
        taus = np.asarray(taus, dtype=float)
        dy = np.diff(yields, axis=0)
        bench_idx = [int(np.argmin(np.abs(taus - t))) for t in benchmark_taus]
        dyb = dy[:, bench_idx]
        beta_hat = np.linalg.lstsq(dyb, dy, rcond=None)[0].T  # N x 2

        def loss(params):
            ar, am, al = params
            if not (0 < al < am < ar):
                return 1e6
            temp_g = GaussPlus(ar, am, al, self.sigma_m, self.sigma_l,
                              self.rho, self.mu)
            Upsb = np.vstack([temp_g.upsilon(t)[1:] for t in benchmark_taus])  # 2 x 2
            if np.linalg.cond(Upsb) > 1e10:
                return 1e6
            try:
                Upsb_inv = np.linalg.inv(Upsb)
            except np.linalg.LinAlgError:
                return 1e6
            loss_val = 0.0
            for i, t in enumerate(taus):
                implied = temp_g.upsilon(t)[1:] @ Upsb_inv
                loss_val += float(np.sum((implied - beta_hat[i]) ** 2))
            return loss_val

        if x0 is None:
            x0 = [1.0, 0.5, 0.02]
        res = minimize(loss, x0=x0, method="Nelder-Mead")
        self.alpha_r, self.alpha_m, self.alpha_l = res.x
        return {
            "alpha_r": float(self.alpha_r),
            "alpha_m": float(self.alpha_m),
            "alpha_l": float(self.alpha_l),
        }

    def calibrate_sigma(self, yields: np.ndarray, taus: Sequence[float],
                       benchmark_taus: Tuple[float, float] = (2.0, 10.0),
                       x0: Optional[Sequence[float]] = None) -> Dict[str, float]:
        """Stage 2: Calibrate volatility parameters (sigma_m, sigma_l, rho).

        Matches model-implied yield-change covariance at benchmarks to realized.

        yields : T x N array of observed zero yields (decimals).
        taus : length-N maturities (years).
        benchmark_taus : maturities used for covariance matching.
        x0 : initial guess [sigma_m, sigma_l, rho]. Default [0.01, 0.01, 0.2].
        """
        yields = np.asarray(yields, dtype=float)
        taus = np.asarray(taus, dtype=float)
        T = yields.shape[0]
        dy = np.diff(yields, axis=0)
        bench_idx = [int(np.argmin(np.abs(taus - t))) for t in benchmark_taus]
        realized_cov = (dy.T @ dy) * (252.0 / (T - 1))
        bench_cov = realized_cov[np.ix_(bench_idx, bench_idx)]

        def loss(params):
            sm, sl, rho = params
            if sm <= 0 or sl <= 0 or abs(rho) > 0.999:
                return 1e6
            ups_b = np.vstack([self.upsilon(t)[1:] for t in benchmark_taus])
            # Cascade-form m/l diffusion covariance: Omega_sub = [[sm², rho*sm*sl],[rho*sm*sl, sl²]]
            Omega_sub = np.array([[sm**2, rho * sm * sl], [rho * sm * sl, sl**2]])
            model = ups_b @ Omega_sub @ ups_b.T
            return float(np.sum((model - bench_cov) ** 2))

        if x0 is None:
            x0 = [0.01, 0.01, 0.2]
        res = minimize(loss, x0=x0, method="Nelder-Mead")
        self.sigma_m, self.sigma_l, self.rho = res.x
        return {
            "sigma_m": float(self.sigma_m),
            "sigma_l": float(self.sigma_l),
            "rho": float(self.rho),
        }

    def calibrate_mu(self, yields: np.ndarray, taus: Sequence[float],
                    short_rate: np.ndarray,
                    benchmark_taus: Tuple[float, float] = (2.0, 10.0),
                    decay: float = 0.8,
                    x0: Optional[Sequence[float]] = None) -> float:
        """Stage 3: Calibrate long-run mean (mu).

        Exact fit at benchmark maturities, weighted by exponential decay.

        yields : T x N array of observed zero yields (decimals).
        taus : length-N maturities (years).
        short_rate : length-T short-rate series (decimals).
        benchmark_taus : maturities pinned exactly in each fit.
        decay : exponential decay 0.8^(years_old) applied to historical residuals.
        x0 : initial guess [mu]. Default [0.05].
        """
        yields = np.asarray(yields, dtype=float)
        taus = np.asarray(taus, dtype=float)
        short_rate = np.asarray(short_rate, dtype=float)
        T = yields.shape[0]
        bench_idx = [int(np.argmin(np.abs(taus - t))) for t in benchmark_taus]
        ages = np.arange(T)[::-1] / 252.0
        weights = decay ** ages

        def loss(mu_arr):
            mu_val = float(mu_arr[0])
            temp_g = GaussPlus(self.alpha_r, self.alpha_m, self.alpha_l,
                              self.sigma_m, self.sigma_l, self.rho, mu_val)
            total = 0.0
            for t in range(T):
                temp_g.r = float(short_rate[t])
                try:
                    up_b = np.vstack([temp_g.upsilon(tau)[1:] for tau in benchmark_taus])
                    c_b = np.array([temp_g.C(tau) for tau in benchmark_taus])
                    ups0 = np.array([temp_g.upsilon(tau)[0] for tau in benchmark_taus])
                    ub_sum = np.array([temp_g.upsilon(tau).sum() for tau in benchmark_taus])
                    y_b = yields[t, bench_idx]
                    rhs = y_b - mu_val * (1 - ub_sum) + c_b - ups0 * temp_g.r
                    m_l = np.linalg.solve(up_b, rhs)
                    temp_g.m, temp_g.l = float(m_l[0]), float(m_l[1])
                except np.linalg.LinAlgError:
                    return 1e6
                model_y = np.array([temp_g.yield_(tau) for tau in taus])
                total += float(weights[t]) * float(np.sum((model_y - yields[t]) ** 2))
            return total

        if x0 is None:
            x0 = [self.mu]
        res = minimize(loss, x0=x0, method="Nelder-Mead")
        self.mu = float(res.x[0])
        return self.mu

    # ----------------------------------------------- simulation
    def simulate(self, T: float, n_steps: int, n_paths: int,
                 seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        r = np.empty((n_paths, n_steps + 1))
        m = np.empty((n_paths, n_steps + 1))
        l = np.empty((n_paths, n_steps + 1))
        r[:, 0], m[:, 0], l[:, 0] = self.r, self.m, self.l
        rho = self.rho
        sqrt1m = np.sqrt(max(1 - rho**2, 0.0))
        for i in range(n_steps):
            dw1 = rng.standard_normal(n_paths) * sqrt_dt
            dw2 = rng.standard_normal(n_paths) * sqrt_dt
            r[:, i + 1] = r[:, i] + self.alpha_r * (m[:, i] - r[:, i]) * dt
            m[:, i + 1] = (
                m[:, i] + self.alpha_m * (l[:, i] - m[:, i]) * dt
                + self.sigma_m * (rho * dw1 + sqrt1m * dw2)
            )
            l[:, i + 1] = (
                l[:, i] + self.alpha_l * (self.mu - l[:, i]) * dt + self.sigma_l * dw1
            )
        return {"r": r, "m": m, "l": l}

    # ----------------------------------------------- risk premia (A9.2.3)
    def amount_of_risk(self, tau: float, tenor: float = 1.0) -> float:
        """RP_lambda(tau, tenor) = integral_0^tenor (tau+tenor-s) Upsilon_3 sigma_l ds.

        Pure risk-premium contribution per unit price-of-risk lambda_t. Independent
        of the factor levels (Eq. A9.27 with only the long factor priced).
        """
        steps = max(int(tenor * 200), 50)
        s_grid = np.linspace(0.0, tenor, steps + 1)
        vals = np.empty_like(s_grid)
        for i, s in enumerate(s_grid):
            T = tau + tenor - s
            if T <= 0:
                vals[i] = 0.0
                continue
            u3 = self.upsilon(T)[2]
            vals[i] = T * u3 * self.sigma_l
        ds = s_grid[1] - s_grid[0]
        return float(0.5 * ds * (vals[0] + vals[-1]) + ds * vals[1:-1].sum())

    def implied_lambda(self, f_tau: float, f_tau_prime: float,
                       tau: float, tau_prime: float, tenor: float = 1.0) -> float:
        """Implied price of risk lambda_t (Eq. A9.28).

        Assumes that beyond some long maturity the real-world expectation
        of the short rate is approximately constant, so the difference of
        forwards at tau and tau_prime is pure risk-premium difference.
        """
        rp_tau = self.amount_of_risk(tau, tenor) / tenor
        rp_tp = self.amount_of_risk(tau_prime, tenor) / tenor
        denom = rp_tp - rp_tau
        if denom == 0:
            raise ZeroDivisionError("Pick (tau, tau_prime) with different RP loadings.")
        return (f_tau_prime - f_tau) / denom

    def decompose_forward(self, tau: float, tenor: float, lambda_t: float) -> Dict[str, float]:
        """Decompose forward into expectation (P-measure), risk premium, and convexity.

            forward = expectation_P + risk_premium - convexity
            => expectation_P = forward - risk_premium + convexity
        """
        forward = self.forward_rate(tau, tenor)
        convex = self.C_prime(tau, tenor)
        risk_prem = lambda_t * self.amount_of_risk(tau, tenor)
        return {
            "forward": float(forward),
            "expectation_real_world": float(forward - risk_prem + convex),
            "risk_premium": float(risk_prem),
            "convexity": float(convex),
        }


# ------------------------------------------------- staged estimator (A9.2.2)
def estimate_gauss_plus(
    yields: np.ndarray,
    taus: Sequence[float],
    short_rate: np.ndarray,
    benchmark_taus: Tuple[float, float] = (2.0, 10.0),
    decay: float = 0.8,
    mu_init: float = 0.05,
) -> dict:
    """Three-stage estimator from Appendix A9.2.2.

    yields : T x N array of observed zero yields (decimals).
    taus   : length-N maturities (years).
    short_rate : length-T short-rate series (decimals).
    benchmark_taus : maturities pinned exactly (default 2y, 10y).
    decay : exponential decay 0.8^(years_old) applied in stage 3.
    """
    yields = np.asarray(yields, dtype=float)
    taus = np.asarray(taus, dtype=float)
    short_rate = np.asarray(short_rate, dtype=float)
    T, N = yields.shape
    bench_idx = [int(np.argmin(np.abs(taus - t))) for t in benchmark_taus]

    dy = np.diff(yields, axis=0)
    dyb = dy[:, bench_idx]

    # Stage 0 (A9.2.2): the short rate r_t is observed and treated as a known
    # input; we exclude maturities at or below the shortest benchmark from the
    # alpha-matching loss so the policy-rate-driven short end does not collapse
    # the cascade ordering.  The OLS betas are computed only from 2Y+ yields
    # regressed on the two benchmark (2Y, 10Y) yield changes — exactly (A9.21).
    min_bench = min(benchmark_taus)
    long_mask = taus >= min_bench
    long_idx  = np.where(long_mask)[0]
    beta_hat  = np.linalg.lstsq(dyb, dy[:, long_idx], rcond=None)[0].T  # len(long_idx) x 2
    taus_fit  = taus[long_idx]

    def stage1_loss(params):
        ar, am, al = params
        if not (0 < al < am < ar):
            return 1e6
        g = GaussPlus(ar, am, al, sigma_m=0.01, sigma_l=0.01, rho=0.0, mu=mu_init)
        Upsb = np.vstack([g.upsilon(t)[1:] for t in benchmark_taus])  # 2 x 2
        if np.linalg.cond(Upsb) > 1e10:
            return 1e6
        try:
            Upsb_inv = np.linalg.inv(Upsb)
        except np.linalg.LinAlgError:
            return 1e6
        loss = 0.0
        for i, t in enumerate(taus_fit):
            implied = g.upsilon(t)[1:] @ Upsb_inv
            loss += float(np.sum((implied - beta_hat[i]) ** 2))
        return loss

    res1 = minimize(stage1_loss, x0=[1.0, 0.5, 0.02], method="Nelder-Mead")
    alpha_r, alpha_m, alpha_l = res1.x

    # Stage 2 (A9.22): match realized benchmark yield-change covariance.
    # Model covariance at benchmarks = Υ_b Σ_red Υ_b^T where
    # Σ_red = A^{-1} Ω Ω^T A^{-T} is the full 3×3 reduced-form covariance.
    realized_cov = (dy.T @ dy) * (252.0 / (T - 1))
    bench_cov = realized_cov[np.ix_(bench_idx, bench_idx)]

    def stage2_loss(params):
        sm, sl, rho = params
        if sm <= 0 or sl <= 0 or abs(rho) > 0.999:
            return 1e6
        g = GaussPlus(alpha_r, alpha_m, alpha_l, sm, sl, rho, mu_init)
        ups_b = np.vstack([g.upsilon(t) for t in benchmark_taus])      # 2 x 3
        Sigma = g.Sigma_red()                                           # 3 x 3
        model = ups_b @ Sigma @ ups_b.T
        return float(np.sum((model - bench_cov) ** 2))

    res2 = minimize(stage2_loss, x0=[0.01, 0.01, 0.2], method="Nelder-Mead")
    sigma_m, sigma_l, rho = res2.x

    ages = np.arange(T)[::-1] / 252.0
    weights = decay ** ages

    def stage3_loss(mu_arr):
        mu_val = float(mu_arr[0])
        g = GaussPlus(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu_val)
        total = 0.0
        for t in range(T):
            g.r = float(short_rate[t])
            try:
                # Exact fit at the two benchmark yield maturities (not forwards here).
                up_b = np.vstack([g.upsilon(tau)[1:] for tau in benchmark_taus])
                c_b = np.array([g.C(tau) for tau in benchmark_taus])
                ups0 = np.array([g.upsilon(tau)[0] for tau in benchmark_taus])
                ub_sum = np.array([g.upsilon(tau).sum() for tau in benchmark_taus])
                y_b = yields[t, bench_idx]
                rhs = y_b - mu_val * (1 - ub_sum) + c_b - ups0 * g.r
                m_l = np.linalg.solve(up_b, rhs)
                g.m, g.l = float(m_l[0]), float(m_l[1])
            except np.linalg.LinAlgError:
                return 1e6
            model_y = np.array([g.yield_(tau) for tau in taus])
            total += float(weights[t]) * float(np.sum((model_y - yields[t]) ** 2))
        return total

    res3 = minimize(stage3_loss, x0=[mu_init], method="Nelder-Mead")
    mu_hat = float(res3.x[0])

    g = GaussPlus(alpha_r, alpha_m, alpha_l, sigma_m, sigma_l, rho, mu_hat)
    return {
        "alpha_r": float(alpha_r),
        "alpha_m": float(alpha_m),
        "alpha_l": float(alpha_l),
        "sigma_m": float(sigma_m),
        "sigma_l": float(sigma_l),
        "rho": float(rho),
        "mu": mu_hat,
        "model": g,
        "stage1_rss": float(res1.fun),
        "stage2_rss": float(res2.fun),
        "stage3_rss": float(res3.fun),
    }

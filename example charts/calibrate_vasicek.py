"""Calibrate Vasicek parameters from NY Fed zero-coupon yield data.

Two-stage procedure:
  Stage 1 -- k, sigma: OLS on daily short-rate changes (Eq. 9.1 discretized)
                        Δr_t = k*(theta - r_{t-1})*dt + sigma*sqrt(dt)*eps
             => k from slope of Δr on r_{t-1}; sigma from residual std.
  Stage 2 -- theta:    minimize weighted yield-fitting error on the
                        time-averaged observed yield curve.

Saves calibrated parameters to calibrated_vasicek_params.py.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from vasicek import Vasicek


# ── load data ─────────────────────────────────────────────────────────────────
def load_yields():
    xlsx = os.path.join(os.path.dirname(__file__), "..", "zero_coupon_yields.xlsx")
    df = pd.read_excel(xlsx, sheet_name="Table Data", header=0)
    df = df.iloc[1:].copy()
    df.columns = ["Date", "y10", "y1", "y2", "y3", "y5", "y6",
                  "y7", "y8", "y9", "y12", "y15", "y17", "y20", "y25", "y27", "y30"]
    df["Date"] = pd.to_datetime(df["Date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100
    df = df.dropna(subset=["y1", "y2", "y3", "y5", "y7", "y10"]).reset_index(drop=True)
    return df


def main():
    df = load_yields()
    print(f"Loaded {len(df)} trading days: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")

    r = df["y1"].values          # short rate proxy
    dt = 1.0 / 252.0

    # ── Stage 1: estimate k and sigma from short-rate time series ─────────────
    # OLS: Δr_t = a + b * r_{t-1} + eps
    # b = -k*dt  =>  k = -b/dt
    # a = k*theta*dt  =>  theta = a / (k*dt)  (use Stage 2 for theta instead)
    dr = np.diff(r)
    r_lag = r[:-1]

    X = np.column_stack([np.ones(len(r_lag)), r_lag])
    beta, _, _, _ = np.linalg.lstsq(X, dr, rcond=None)
    a_hat, b_hat = beta

    k = -b_hat / dt
    theta_ols = a_hat / (k * dt)        # OLS implied theta (used as init only)

    # Residual std → annualized sigma
    dr_hat = X @ beta
    resid = dr - dr_hat
    sigma = resid.std() / np.sqrt(dt)

    print(f"\nStage 1 (OLS on dr):")
    print(f"  k     = {k:.6f}  (half-life {np.log(2)/k:.1f} years)")
    print(f"  sigma = {sigma:.6f}  ({sigma*100:.2f}%/yr)")
    print(f"  theta_ols = {theta_ols:.6f}  (Stage 2 refits)")

    # ── Stage 2: fit theta to time-averaged yield curve ───────────────────────
    taus = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    cols = ["y1", "y2", "y3", "y5", "y7", "y10"]
    mean_yields = df[cols].mean().values
    r0 = float(r[-1])           # use last observed short rate

    print(f"\nStage 2 (theta from average yield curve):")
    print(f"  r0    = {r0:.6f}  (last observed 1Y rate)")
    print(f"  Mean yield curve: " + ", ".join(f"{y*100:.2f}%" for y in mean_yields))

    # Grid search + fine minimize over theta
    def yield_sse(theta):
        v = Vasicek(r0=r0, k=k, theta=theta, sigma=sigma)
        model = np.array([v.spot_rate(t) for t in taus])
        return float(np.sum((model - mean_yields) ** 2))

    from scipy.optimize import minimize_scalar
    res2 = minimize_scalar(yield_sse, bounds=(0.0, 0.25), method="bounded")
    theta = float(res2.x)

    print(f"  theta = {theta:.6f}  (SSE={res2.fun:.2e})")

    # Final model
    v_final = Vasicek(r0=r0, k=k, theta=theta, sigma=sigma)
    fitted = np.array([v_final.spot_rate(t) for t in taus])
    resids_bps = (fitted - mean_yields) * 1e4

    print(f"\n=== CALIBRATED VASICEK PARAMETERS ===")
    print(f"  k     = {k:.8f}")
    print(f"  theta = {theta:.8f}")
    print(f"  sigma = {sigma:.8f}")
    print(f"  r0    = {r0:.8f}")
    print(f"\nFit to mean yield curve (residuals in bps):")
    for tau, obs, fit, res in zip(taus, mean_yields, fitted, resids_bps):
        print(f"  {tau:4.0f}Y  obs={obs*100:.3f}%  fit={fit*100:.3f}%  res={res:+.1f}bps")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "calibrated_vasicek_params.py")
    with open(out_path, "w") as f:
        f.write('"""Vasicek parameters calibrated from NY Fed zero-coupon yields.\n')
        f.write(f'Data: {df["Date"].iloc[0].date()} to {df["Date"].iloc[-1].date()}, {len(df)} trading days.\n')
        f.write('Stage 1: OLS on daily dr (k, sigma). Stage 2: theta from mean yield curve.\n"""\n\n')
        f.write("CALIBRATED_VASICEK = dict(\n")
        f.write(f"    k={k:.8f},\n")
        f.write(f"    theta={theta:.8f},\n")
        f.write(f"    sigma={sigma:.8f},\n")
        f.write(f"    r0={r0:.8f},\n")
        f.write(")\n")

    print(f"\nSaved calibrated parameters to: {out_path}")


if __name__ == "__main__":
    main()

"""Calibrate Gauss+ parameters from NY Fed zero-coupon yield data.

Three-stage procedure (Appendix A9.2.2):
  Stage 1 -- alpha: match OLS yield-change betas at all maturities
  Stage 2 -- sigma: match realized yield-change covariance at benchmarks
  Stage 3 -- mu:   minimize weighted yield-fitting error across sample

Saves calibrated parameters to calibrated_params.py so gauss_plus_figures.py
can import them instead of TABLE_9_1.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from gauss_plus import GaussPlus, estimate_gauss_plus


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
    df = df.dropna(subset=["y1", "y2", "y3", "y5", "y7", "y10", "y12", "y15"]).reset_index(drop=True)
    return df


def main():
    df = load_yields()
    print(f"Loaded {len(df)} trading days: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")

    # Maturities present in the dataset (use a clean subset for calibration)
    taus = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0])
    cols = ["y1", "y2", "y3", "y5", "y7", "y10", "y12", "y15"]

    yields_arr  = df[cols].values           # T x N
    short_rate  = df["y1"].values           # T-vector, treat y1 as short rate

    print(f"Yield array shape: {yields_arr.shape}")
    print(f"Maturities: {taus}")
    print("\nRunning three-stage calibration (this may take 1-3 minutes)...")

    result = estimate_gauss_plus(
        yields     = yields_arr,
        taus       = taus,
        short_rate = short_rate,
        benchmark_taus = (2.0, 10.0),
        decay      = 0.8,
        mu_init    = 0.04,
    )

    print("\n=== CALIBRATED PARAMETERS ===")
    for k, v in result.items():
        if k != "model":
            print(f"  {k:12s} = {v:.6f}")

    # Write calibrated params to a Python module for import by the figures script
    out_path = os.path.join(os.path.dirname(__file__), "calibrated_params.py")
    with open(out_path, "w") as f:
        f.write('"""Gauss+ parameters calibrated from NY Fed zero-coupon yields.\n')
        f.write(f'Data: {df["Date"].iloc[0].date()} to {df["Date"].iloc[-1].date()}, {len(df)} trading days.\n')
        f.write('Three-stage procedure: Appendix A9.2.2 of Tuckman & Serrat (4e).\n"""\n\n')
        f.write("CALIBRATED = dict(\n")
        for k in ["alpha_r", "alpha_m", "alpha_l", "sigma_m", "sigma_l", "rho", "mu"]:
            f.write(f"    {k}={result[k]:.8f},\n")
        f.write(")\n")

    print(f"\nSaved calibrated parameters to: {out_path}")


if __name__ == "__main__":
    main()

"""Reproduce result charts for the Gauss+ section of Tuckman & Serrat Ch. 9.

Figures produced (saved to examples/ directory):
  gauss_plus_fig9_7.png  -- Factor loadings Upsilon(tau, alpha) vs maturity
  gauss_plus_fig9_8.png  -- Time-series: extracted Gauss+ factors + 2Y 1Y forward
  gauss_plus_fig9_9.png  -- One-year forward curve
  gauss_plus_fig9_10.png -- Time-series: 10Y risk premium (bps) vs 10Y 1Y forward (%)
  gauss_plus_fig9_11.png -- Shifted long factor L(l_t) from simulated paths

Parameters: Table 9.1 (alpha_r=1.0547, alpha_m=0.6358, alpha_l=0.0165,
            sigma_m=0.01092, sigma_l=0.00964, rho=0.212, mu=0.10555).
State (Jan-2022 representative): r=8 bps, m=1.2%, l=2.2%.

Figs 9.8 and 9.10 use daily NY Fed zero-coupon yield data from
zero_coupon_yields.xlsx (sheet "Table Data") in the project root.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from gauss_plus import GaussPlus, TABLE_9_1

try:
    from calibrated_params import CALIBRATED
    _PARAMS = CALIBRATED
    _PARAMS_LABEL = "calibrated"
except ImportError:
    _PARAMS = TABLE_9_1
    _PARAMS_LABEL = "TABLE_9_1"

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
})

OUT_DIR = os.path.dirname(__file__)

# ── model setup ───────────────────────────────────────────────────────────────
g = GaussPlus(**_PARAMS)
g.set_factors(r=0.0008, m=0.012, l=0.022)

TAUS = np.linspace(0.25, 30, 300)
TENOR = 1.0  # one-year forward tenor throughout


# ── Figure 9.7 — factor loadings ──────────────────────────────────────────────
def fig9_7():
    upsilons = np.array([g.upsilon(t) for t in TAUS])   # (N, 3)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(TAUS, upsilons[:, 0], label=r"$\Upsilon_r$ (short)")
    ax.plot(TAUS, upsilons[:, 1], label=r"$\Upsilon_m$ (medium)", linestyle="--")
    ax.plot(TAUS, upsilons[:, 2], label=r"$\Upsilon_l$ (long)",   linestyle=":")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("Maturity τ (years)")
    ax.set_ylabel(r"Loading $\Upsilon(\tau,\alpha)$")
    ax.set_title("Figure 9.7 — Gauss+ Factor Loadings on Zero Yields")
    ax.legend()
    path = os.path.join(OUT_DIR, "gauss_plus_fig9_7.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 9.8 — extracted factors + 2Y 1Y forward (time series) ─────────────
def _load_yields():
    """Load NY Fed zero-coupon yields from project-root Excel file."""
    xlsx = os.path.join(os.path.dirname(__file__), "..", "zero_coupon_yields.xlsx")
    df = pd.read_excel(xlsx, sheet_name="Table Data", header=0)
    df = df.iloc[1:].copy()
    df.columns = ["Date", "y10", "y1", "y2", "y3", "y5", "y6",
                  "y7", "y8", "y9", "y12", "y15", "y17", "y20", "y25", "y27", "y30"]
    df["Date"] = pd.to_datetime(df["Date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100
    df = df.dropna(subset=["y1", "y2", "y10", "y12"]).reset_index(drop=True)
    return df


def fig9_8():
    """Time-series of Gauss+ extracted factors (r, m, l) and the 2Y 1Y forward.

    For each trading day: set r = y1 (observed short rate), compute the 2Y and
    10Y one-year forwards from observed zero yields, then call fit_factors so the
    model matches those two forwards exactly.  Plot the resulting daily factor
    series alongside the 2Y 1Y forward — replicating Figure 9.8 of Tuckman &
    Serrat (4e), adapted to the available data window.
    """
    df = _load_yields()
    gm = GaussPlus(**_PARAMS)

    dates = []
    r_series, m_series, l_series, fwd2_series = [], [], [], []

    for _, row in df.iterrows():
        y1, y2, y10, y12 = row["y1"], row["y2"], row["y10"], row["y12"]

        # 2Y 1Y continuously-compounded forward: (2*y2 - 1*y1)
        f2 = 2.0 * y2 - 1.0 * y1

        # 10Y 1Y forward: interpolate y11 = midpoint(y10, y12), then 11*y11 - 10*y10
        y11 = (y10 + y12) / 2.0
        f10 = 11.0 * y11 - 10.0 * y10

        try:
            gm.fit_factors(observed_forwards={2: f2, 10: f10}, tenor=1.0, r_obs=y1)
        except Exception:
            continue

        dates.append(row["Date"])
        r_series.append(gm.r * 100)
        m_series.append(gm.m * 100)
        l_series.append(gm.l * 100)
        fwd2_series.append(f2 * 100)

    dates = pd.to_datetime(dates)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, fwd2_series, color="black",      linewidth=1.8, label="2yr Fwd")
    ax.plot(dates, l_series,    color="steelblue",  linewidth=1.4, label="Long ($l_t$)",   linestyle="-")
    ax.plot(dates, m_series,    color="darkorange",  linewidth=1.4, label="Medium ($m_t$)", linestyle="--")
    ax.plot(dates, r_series,    color="green",       linewidth=1.4, label="Short ($r_t$)",  linestyle=":")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Rate (% p.a.)")
    ax.set_title("Figure 9.8 — 2yr Forward Rate and Gauss+ Factors Extracted from Daily Market Data")
    ax.legend(fontsize=9)
    path = os.path.join(OUT_DIR, "gauss_plus_fig9_8.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 9.9 — one-year forward curve ───────────────────────────────────────
def fig9_9():
    # Forward taus run from 0 to 29 so the forward window [tau, tau+1] fits in [0,30].
    tau_fwd = np.linspace(0.0, 29.0, 300)
    fwds    = np.array([g.forward_rate(t, tenor=TENOR) for t in tau_fwd]) * 100
    yields  = np.array([g.yield_(t) for t in TAUS]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(TAUS, yields, color="steelblue", label="Spot yield", linestyle="--", linewidth=1.5)
    ax.plot(tau_fwd, fwds, color="darkorange", label="1Y forward")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    ax.set_xlabel("Start of forward period τ (years)")
    ax.set_ylabel("Rate (% p.a.)")
    ax.set_title("Figure 9.9 — Gauss+ One-Year Forward Curve vs Spot Yield")
    ax.legend()
    path = os.path.join(OUT_DIR, "gauss_plus_fig9_9.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 9.10 — risk premium on 10Y forward (time series, dual axis) ────────
def fig9_10():
    """Dual-axis time-series: 10Y risk premium in bps (left) vs 10Y 1Y forward % (right).

    For each trading day: extract Gauss+ factors via fit_factors, infer the
    price of risk lambda_t from the 14Y/15Y forward spread (Appendix A9.2.3,
    Eq. A9.28), then decompose the 10Y 1Y forward into expectation, risk premium,
    and convexity.  This replicates Figure 9.10 of Tuckman & Serrat (4e).
    Requires columns y14/y15 — approximated by linear interpolation from
    available maturities (y12, y15, y17) when exact columns are absent.
    """
    df = _load_yields()

    # Need y14 and y15 for the lambda extraction; interpolate linearly if absent.
    # Available: y12, y15 — so y14 = y12 + (2/3)*(y15 - y12)
    if "y14" not in df.columns:
        df["y14"] = df["y12"] + (2.0 / 3.0) * (df["y15"] - df["y12"])
    if "y15" not in df.columns:
        df["y15"] = df["y15"]   # already present

    gm = GaussPlus(**_PARAMS)

    dates, rp_series, fwd10_series = [], [], []

    for _, row in df.iterrows():
        y1, y2, y10, y12 = row["y1"], row["y2"], row["y10"], row["y12"]
        y14, y15 = row["y14"], row["y15"]
        if any(pd.isna(v) for v in [y1, y2, y10, y12, y14, y15]):
            continue

        y11 = (y10 + y12) / 2.0
        f2  = 2.0 * y2  - 1.0 * y1
        f10 = 11.0 * y11 - 10.0 * y10

        try:
            gm.fit_factors(observed_forwards={2: f2, 10: f10}, tenor=1.0, r_obs=y1)
        except Exception:
            continue

        # Interpolate y13 = midpoint(y12, y14) to get 13Y 1Y and 14Y 1Y forwards
        y13 = (y12 + y14) / 2.0
        f14_rate = 15.0 * y14 - 14.0 * y13    # 14Y 1Y forward
        f15_rate = 16.0 * y15 - 15.0 * y14    # 15Y 1Y forward (approx, no y16)

        try:
            lam = gm.implied_lambda(f14_rate, f15_rate, tau=14, tau_prime=15, tenor=1.0)
            d   = gm.decompose_forward(10, tenor=1.0, lambda_t=lam)
        except Exception:
            continue

        dates.append(row["Date"])
        rp_series.append(d["risk_premium"] * 10000)   # bps
        fwd10_series.append(f10 * 100)                 # percent

    dates = pd.to_datetime(dates)
    rp_arr  = np.array(rp_series)
    fwd_arr = np.array(fwd10_series)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    ax1.bar(dates, rp_arr, width=2, color="steelblue", alpha=0.7, label="Risk Premium (bps)")
    ax2.plot(dates, fwd_arr, color="darkorange", linewidth=1.8, label="10yr Forward (%)")

    ax1.set_ylabel("Risk Premium (basis points)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylabel("10yr 1Y Forward Rate (%)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax1.axhline(0, color="k", linewidth=0.5)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=30)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title("Figure 9.10 — Estimated Risk Premium on the 10-Year Forward Rate")
    path = os.path.join(OUT_DIR, "gauss_plus_fig9_10.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 9.11 — long-run expectations of the short-term rate ────────────────
def fig9_11():
    """Time-series: Gauss+ model long-run expectation vs Cleveland Fed measure.

    For each trading day in the market data, extract Gauss+ factors via
    fit_factors, infer lambda_t from the 14Y/15Y forward spread, then read off
    the P-measure expectation of the short rate at the 10-year horizon using
    decompose_forward(10).

    The Cleveland Fed long-run expectation is approximated as the sum of their
    published 10-year expected inflation (FRED: EXPINF10YR) and the 10-year
    real interest rate (FRED: FII10), both fetched live and resampled to daily.
    """
    import urllib.request
    import io

    # ── Cleveland Fed proxy (monthly -> daily forward-filled) ─────────────────
    def _fetch_fred(series):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        with urllib.request.urlopen(url, timeout=20) as r:
            raw = r.read().decode()
        df = pd.read_csv(io.StringIO(raw), parse_dates=[0])
        df.columns = ["Date", series]
        df[series] = pd.to_numeric(df[series], errors="coerce")
        return df.dropna()

    try:
        inf  = _fetch_fred("EXPINF10YR")   # 10Y expected inflation, % p.a.
        real = _fetch_fred("FII10")         # 10Y real rate, % p.a.
        clev = inf.merge(real, on="Date", how="inner")
        clev["ClevFed"] = clev["EXPINF10YR"] + clev["FII10"]
        clev = clev[["Date", "ClevFed"]].set_index("Date")
        clev_available = True
    except Exception:
        clev_available = False

    # ── Gauss+ model: daily factor extraction ─────────────────────────────────
    df = _load_yields()
    if "y14" not in df.columns:
        df["y14"] = df["y12"] + (2.0 / 3.0) * (df["y15"] - df["y12"])

    gm = GaussPlus(**_PARAMS)
    dates_m, expect_m = [], []

    for _, row in df.iterrows():
        y1, y2, y10, y12 = row["y1"], row["y2"], row["y10"], row["y12"]
        y14, y15 = row["y14"], row["y15"]
        if any(pd.isna(v) for v in [y1, y2, y10, y12, y14, y15]):
            continue

        y11 = (y10 + y12) / 2.0
        f2  = 2.0 * y2  - 1.0 * y1
        f10 = 11.0 * y11 - 10.0 * y10
        try:
            gm.fit_factors(observed_forwards={2: f2, 10: f10}, tenor=1.0, r_obs=y1)
        except Exception:
            continue

        y13 = (y12 + y14) / 2.0
        f14_rate = 15.0 * y14 - 14.0 * y13
        f15_rate = 16.0 * y15 - 15.0 * y14
        try:
            lam = gm.implied_lambda(f14_rate, f15_rate, tau=14, tau_prime=15, tenor=1.0)
            d   = gm.decompose_forward(10, tenor=1.0, lambda_t=lam)
        except Exception:
            continue

        dates_m.append(row["Date"])
        expect_m.append(d["expectation_real_world"] * 100)

    dates_m  = pd.to_datetime(dates_m)
    expect_m = np.array(expect_m)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(dates_m, expect_m, color="gray", linewidth=1.4, label="Model")

    if clev_available:
        # Reindex Cleveland Fed to trading days, forward-fill monthly values
        idx = pd.date_range(clev.index.min(), clev.index.max(), freq="D")
        clev_daily = clev.reindex(idx).ffill()
        # Restrict to dates covered by our yield data
        mask = (clev_daily.index >= dates_m.min()) & (clev_daily.index <= dates_m.max())
        clev_plot = clev_daily[mask]
        ax.plot(clev_plot.index, clev_plot["ClevFed"].values,
                color="black", linewidth=2.0, label="Cleveland Fed")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Rate")
    ax.set_title(
        "Figure 9.11 — Long-Run Expectations of the Short-Term Rate\n"
        "Gauss+ Model vs Cleveland Fed (Inflation Exp. + Real Rate)"
    )
    ax.legend(fontsize=10)
    path = os.path.join(OUT_DIR, "gauss_plus_fig9_11.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Gauss+ — reproducing Figures 9.7–9.11 (Tuckman & Serrat, 4e)")
    print(f"Parameters ({_PARAMS_LABEL}): {_PARAMS}")
    print(f"State: r={g.r:.4%}, m={g.m:.4%}, l={g.l:.4%}\n")
    fig9_7()
    fig9_8()
    fig9_9()
    fig9_10()
    fig9_11()
    print("\nDone — all figures saved to the examples/ directory.")


if __name__ == "__main__":
    main()

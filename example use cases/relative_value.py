"""Relative value: model vs market yield richness/cheapness.

Uses the calibrated Gauss+ model to fit factors daily from the 2Y and 10Y
yields, then computes the model-implied yield at all other maturities and
measures the spread (market - model) as a richness/cheapness signal.

Outputs:
  - Time series of richness/cheapness at the 5Y and 30Y
  - Current (end-of-sample) model curve vs observed curve
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example charts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gauss_plus import GaussPlus
from calibrated_params import CALIBRATED


def load_yields():
    xlsx = os.path.join(os.path.dirname(__file__), "..", "zero_coupon_yields.xlsx")
    df = pd.read_excel(xlsx, sheet_name="Table Data", header=0)
    df = df.iloc[1:].copy()
    df.columns = ["Date", "y10", "y1", "y2", "y3", "y5", "y6",
                  "y7", "y8", "y9", "y12", "y15", "y17", "y20", "y25", "y27", "y30"]
    df["Date"] = pd.to_datetime(df["Date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100
    df = df.dropna(subset=["y1", "y2", "y5", "y10", "y30"]).reset_index(drop=True)
    return df


df = load_yields()
g = GaussPlus(**CALIBRATED)

# Maturities to monitor for richness/cheapness (2Y and 10Y used for fitting)
monitor_taus = {"5Y": 5.0, "30Y": 30.0}
tenor = 1.0

spreads = {k: [] for k in monitor_taus}
dates = []

print("Fitting Gauss+ factors daily and computing richness/cheapness...")
for _, row in df.iterrows():
    r_obs = row["y1"]
    f2  = row["y2"]    # use spot yields as forward approximation for extraction
    f10 = row["y10"]

    try:
        g.fit_factors(observed_forwards={2.0: f2, 10.0: f10},
                      tenor=tenor, r_obs=r_obs)
    except Exception:
        for k in monitor_taus:
            spreads[k].append(np.nan)
        dates.append(row["Date"])
        continue

    dates.append(row["Date"])
    for label, tau in monitor_taus.items():
        col = {5.0: "y5", 30.0: "y30"}[tau]
        obs = row[col]
        model = g.yield_(tau)
        spreads[label].append((obs - model) * 1e4)   # bps

dates = pd.to_datetime(dates)

# Current snapshot: model vs market
last = df.iloc[-1]
r_obs = last["y1"]
g.fit_factors(observed_forwards={2.0: last["y2"], 10.0: last["y10"]},
              tenor=tenor, r_obs=r_obs)

obs_taus  = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])
obs_cols  = ["y1", "y2", "y3", "y5", "y7", "y10", "y15", "y20", "y30"]
obs_yields = np.array([last[c] for c in obs_cols])
model_yields = np.array([g.yield_(t) for t in obs_taus])
rich_bps = (obs_yields - model_yields) * 1e4

print(f"\nCurrent snapshot ({last['Date'].date()}) — market vs Gauss+ model:")
print(f"{'Maturity':>8}  {'Market':>9}  {'Model':>9}  {'Rich/Cheap (bps)':>17}")
for tau, obs, mod, rb in zip(obs_taus, obs_yields, model_yields, rich_bps):
    flag = "CHEAP" if rb > 5 else ("RICH" if rb < -5 else "")
    print(f"{tau:8.0f}Y  {obs*100:8.3f}%  {mod*100:8.3f}%  {rb:+16.1f}  {flag}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for label, color in zip(monitor_taus, ["steelblue", "tomato"]):
    s = pd.Series(spreads[label], index=dates).dropna()
    ax.plot(s.index, s.values, linewidth=1.2, label=label, color=color, alpha=0.85)
ax.axhline(0, color="k", linewidth=1)
ax.fill_between(dates, -10, 10, alpha=0.08, color="gray", label="+/- 10 bps")
ax.set_ylabel("Market minus model (bps)")
ax.set_title("Gauss+ Richness / Cheapness", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
tau_fine = np.linspace(0.5, 30, 100)
model_fine = np.array([g.yield_(t) for t in tau_fine])
ax.plot(tau_fine, model_fine * 100, "b-", linewidth=2, label="Gauss+ model")
ax.scatter(obs_taus, obs_yields * 100, color="red", s=80, zorder=5, label="Market")
for tau, obs, rb in zip(obs_taus, obs_yields, rich_bps):
    color = "green" if rb < -5 else ("red" if rb > 5 else "gray")
    ax.annotate(f"{rb:+.0f}", (tau, obs * 100), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=color)
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Yield (%)")
ax.set_title(f"Current Curve vs Model ({last['Date'].date()})", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "relative_value.png"), dpi=150, bbox_inches="tight")
print("\nSaved: relative_value.png")
plt.show()

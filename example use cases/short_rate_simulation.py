"""Short-rate simulation under Vasicek.

Simulates risk-neutral short-rate paths, shows the distribution of future
rates, and prices a zero-coupon bond by Monte Carlo vs. the analytic formula.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example charts"))

import numpy as np
import matplotlib.pyplot as plt
from vasicek import Vasicek
from calibrated_vasicek_params import CALIBRATED_VASICEK

v = Vasicek(**CALIBRATED_VASICEK)

T = 10
n_steps = 120       # monthly
n_paths = 10_000
seed = 42

paths = v.simulate(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
times = np.linspace(0, T, n_steps + 1)

# Monte Carlo bond price vs analytic
mc_price, mc_se = v.bond_price_mc(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
analytic_price = v.bond_price(T)

print(f"10-year zero-coupon bond pricing")
print(f"  Analytic:    {analytic_price:.6f}  (yield {v.spot_rate(T)*100:.3f}%)")
print(f"  Monte Carlo: {mc_price:.6f}  +/- {mc_se:.6f}  (2-sigma band: "
      f"[{mc_price - 2*mc_se:.6f}, {mc_price + 2*mc_se:.6f}])")

# Distribution of r_T
r_T = paths[:, -1]
print(f"\nDistribution of r_10:")
print(f"  E[r_10] model:    {v.expected_rate(T)*100:.3f}%")
print(f"  E[r_10] MC:       {r_T.mean()*100:.3f}%")
print(f"  Std[r_10] model:  {v.rate_std(T)*100:.3f}%")
print(f"  Std[r_10] MC:     {r_T.std()*100:.3f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sample paths
ax = axes[0]
for i in range(50):
    ax.plot(times, paths[i] * 100, alpha=0.3, linewidth=0.8, color="steelblue")
ax.plot(times, paths[:50].mean(axis=0) * 100, "r-", linewidth=2, label="Mean (50 paths)")
ax.axhline(v.theta * 100, color="k", linestyle="--", linewidth=1.5, label=f"theta={v.theta*100:.2f}%")
ax.set_xlabel("Years")
ax.set_ylabel("Short rate (%)")
ax.set_title("Short-Rate Paths (50 of 10,000)", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution of r_T
ax = axes[1]
ax.hist(r_T * 100, bins=60, density=True, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(v.expected_rate(T) * 100, color="r", linewidth=2, label=f"E[r_10]={v.expected_rate(T)*100:.2f}%")
ax.axvline(v.r0 * 100, color="k", linestyle="--", linewidth=1.5, label=f"r_0={v.r0*100:.2f}%")
ax.set_xlabel("r_10 (%)")
ax.set_ylabel("Density")
ax.set_title(f"Distribution of r at T={T}", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# Mean +/- 1 and 2 sigma bands over time
ax = axes[2]
mean_t = np.array([v.expected_rate(t) for t in times])
std_t = np.array([v.rate_std(t) for t in times])
ax.plot(times, mean_t * 100, "r-", linewidth=2, label="E[r_t]")
ax.fill_between(times, (mean_t - std_t) * 100, (mean_t + std_t) * 100, alpha=0.3, label="+-1 std")
ax.fill_between(times, (mean_t - 2*std_t) * 100, (mean_t + 2*std_t) * 100, alpha=0.15, label="+-2 std")
ax.axhline(v.theta * 100, color="k", linestyle="--", linewidth=1, label=f"theta")
ax.set_xlabel("Years")
ax.set_ylabel("Short rate (%)")
ax.set_title("Model Mean and Uncertainty Bands", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "short_rate_simulation.png"), dpi=150, bbox_inches="tight")
print("\nSaved: short_rate_simulation.png")
plt.show()

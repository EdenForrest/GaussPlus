"""Bond pricing under Vasicek and Gauss+.

Prices zero-coupon bonds across maturities using calibrated parameters,
compares the two models, and shows the bond price / yield relationship.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from vasicek import Vasicek
from gauss_plus import GaussPlus

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example charts"))
from calibrated_vasicek_params import CALIBRATED_VASICEK
from calibrated_params import CALIBRATED

taus = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])

# Vasicek model
v = Vasicek(**CALIBRATED_VASICEK)
v_prices = np.array([v.bond_price(t) for t in taus])
v_yields = np.array([v.spot_rate(t) for t in taus])

# Gauss+ model — extract factors at current short rate
g = GaussPlus(**CALIBRATED)
r0 = CALIBRATED_VASICEK["r0"]
g.set_factors(r=r0, m=r0, l=CALIBRATED["mu"])
g_prices = np.array([np.exp(-g.yield_(t) * t) for t in taus])
g_yields = np.array([g.yield_(t) for t in taus])

print("Zero-coupon bond prices and yields")
print(f"{'Maturity':>8}  {'Vasicek P':>10}  {'Vasicek y':>10}  {'Gauss+ P':>10}  {'Gauss+ y':>10}")
print("-" * 58)
for tau, vp, vy, gp, gy in zip(taus, v_prices, v_yields, g_prices, g_yields):
    print(f"{tau:8.2f}  {vp:10.6f}  {vy*100:9.3f}%  {gp:10.6f}  {gy*100:9.3f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(taus, v_prices, "b-o", markersize=5, label="Vasicek")
ax.plot(taus, g_prices, "r-s", markersize=5, label="Gauss+")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Bond price (per $1 face)")
ax.set_title("Zero-Coupon Bond Prices", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(taus, v_yields * 100, "b-o", markersize=5, label="Vasicek")
ax.plot(taus, g_yields * 100, "r-s", markersize=5, label="Gauss+")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Yield (%)")
ax.set_title("Implied Yield Curves", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "bond_pricing.png"), dpi=150, bbox_inches="tight")
print("\nSaved: bond_pricing.png")
plt.show()

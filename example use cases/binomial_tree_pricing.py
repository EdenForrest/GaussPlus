"""Vasicek binomial tree: short-rate tree and bond pricing.

Builds the Appendix A9.1 recombining binomial tree from calibrated parameters,
prices zero-coupon bonds by backward induction, and compares to the analytic
formula at each node horizon.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example charts"))

import numpy as np
import matplotlib.pyplot as plt
from vasicek import Vasicek
from calibrated_vasicek_params import CALIBRATED_VASICEK

v = Vasicek(**CALIBRATED_VASICEK)

n_steps = 10
dt = 1.0           # annual steps

tree = v.binomial_tree(n_steps=n_steps, dt=dt)
rates = tree["rates"]
probs = tree["probs"]

# ── Print short-rate tree ──────────────────────────────────────────────────
print("Short-rate tree (annual steps, calibrated Vasicek)")
print(f"  k={v.k:.4f}, theta={v.theta*100:.2f}%, sigma={v.sigma*100:.2f}%, r0={v.r0*100:.2f}%\n")
for d, nodes in enumerate(rates):
    node_str = "  ".join(f"{r*100:6.2f}%" for r in nodes)
    print(f"  t={d}: {node_str}")

# ── Price a zero-coupon bond by backward induction ────────────────────────
def tree_bond_price(maturity_steps):
    """Price a ZCB maturing at `maturity_steps` periods using backward induction."""
    # Terminal payoff = 1 at each node
    values = [np.ones(len(rates[maturity_steps]))]
    for step in range(maturity_steps - 1, -1, -1):
        child_vals = values[-1]
        node_vals = np.zeros(len(rates[step]))
        for i, r_i in enumerate(rates[step]):
            p_up, p_dn = probs[step][i]
            ev = p_up * child_vals[i] + p_dn * child_vals[i + 1]
            node_vals[i] = ev * np.exp(-r_i * dt)
        values.append(node_vals)
    return float(values[-1][0])

print("\nZero-coupon bond prices: tree vs analytic")
print(f"{'Maturity':>8}  {'Tree price':>10}  {'Analytic':>10}  {'Diff (bps yield)':>16}")
for mat in range(1, n_steps + 1):
    tp = tree_bond_price(mat)
    ap = v.bond_price(mat * dt)
    ty = -np.log(tp) / (mat * dt) if tp > 0 else np.nan
    ay = v.spot_rate(mat * dt)
    diff_bps = (ty - ay) * 1e4
    print(f"{mat*dt:8.0f}  {tp:10.6f}  {ap:10.6f}  {diff_bps:+15.2f}")

# ── Visualize tree ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for d, nodes in enumerate(rates[:8]):     # show first 7 steps
    for i, r in enumerate(nodes):
        ax.scatter(d, r * 100, color="steelblue", s=60, zorder=3)
        if d < len(probs) and i < len(probs[d]):
            if i < len(rates[d + 1]) - 1:
                ax.plot([d, d + 1], [r * 100, rates[d + 1][i] * 100], "b-", alpha=0.4, linewidth=0.8)
                ax.plot([d, d + 1], [r * 100, rates[d + 1][i + 1] * 100], "b-", alpha=0.4, linewidth=0.8)
ax.axhline(v.theta * 100, color="r", linestyle="--", linewidth=1.5, label=f"theta={v.theta*100:.2f}%")
ax.set_xlabel("Period")
ax.set_ylabel("Short rate (%)")
ax.set_title("Vasicek Binomial Tree (7 steps)", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
mats = list(range(1, n_steps + 1))
tree_prices = [tree_bond_price(m) for m in mats]
analytic_prices = [v.bond_price(m * dt) for m in mats]
ax.plot(mats, tree_prices, "bo-", markersize=6, label="Tree")
ax.plot(mats, analytic_prices, "r--", linewidth=2, label="Analytic")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Bond price")
ax.set_title("Tree vs Analytic Bond Prices", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "binomial_tree_pricing.png"), dpi=150, bbox_inches="tight")
print("\nSaved: binomial_tree_pricing.png")
plt.show()

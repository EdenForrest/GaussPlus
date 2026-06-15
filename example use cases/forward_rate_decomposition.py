"""Forward rate decomposition: expectations, risk premium, convexity.

Decomposes the model forward curve into its three components under both
Vasicek (constant lambda) and Gauss+ (time-varying implied lambda).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example charts"))

import numpy as np
import matplotlib.pyplot as plt
from vasicek import Vasicek
from gauss_plus import GaussPlus
from calibrated_vasicek_params import CALIBRATED_VASICEK
from calibrated_params import CALIBRATED

taus = np.linspace(0.5, 15, 60)

# ── Vasicek decomposition (requires a risk premium assumption) ─────────────
# Estimate lambda as the difference between theta and the long-run OLS mean
# implied by the time-series: use OLS theta_ols as the P-measure mean.
# Here we set lambda so that r_inf = 5% (plausible long-run real-world mean).
r_inf_assumed = 0.05
v = Vasicek(**CALIBRATED_VASICEK)
lam = v.k * (v.theta - r_inf_assumed)
v.risk_premium = lam

v_exp, v_rp, v_cvx, v_fwd = [], [], [], []
for t in taus:
    d = v.decompose_forward(t)
    v_exp.append(d["expectation"])
    v_rp.append(d["risk_premium"])
    v_cvx.append(d["convexity"])
    v_fwd.append(d["forward"])

print("Vasicek forward decomposition (selected maturities)")
print(f"  lambda = {lam*1e4:.1f} bps/yr,  r_inf = {r_inf_assumed*100:.1f}%")
print(f"{'tau':>5}  {'Expectation':>12}  {'Risk prem':>10}  {'Convexity':>10}  {'Forward':>9}")
for t in [1, 2, 5, 10, 15]:
    d = v.decompose_forward(t)
    print(f"{t:5.0f}  {d['expectation']*100:11.3f}%  {d['risk_premium']*100:9.3f}%  "
          f"{d['convexity']*100:9.3f}%  {d['forward']*100:8.3f}%")

# ── Gauss+ decomposition (implied lambda from successive forwards) ──────────
g = GaussPlus(**CALIBRATED)
r0 = CALIBRATED_VASICEK["r0"]
g.set_factors(r=r0, m=r0, l=CALIBRATED["mu"])
tenor = 1.0

g_exp, g_rp, g_cvx, g_fwd = [], [], [], []
taus_g = np.arange(1.0, 15.0, 1.0)   # integer maturities for implied lambda
for tau in taus_g:
    f_tau = g.forward_rate(tau, tenor)
    f_next = g.forward_rate(tau + 1, tenor)
    lam_t = g.implied_lambda(f_tau=f_tau, f_tau_prime=f_next,
                              tau=tau, tau_prime=tau + 1, tenor=tenor)
    d = g.decompose_forward(tau=tau, tenor=tenor, lambda_t=lam_t)
    g_exp.append(d["expectation_real_world"])
    g_rp.append(d["risk_premium"])
    g_cvx.append(d["convexity"])
    g_fwd.append(d["forward"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vasicek
ax = axes[0]
ax.stackplot(taus,
             np.array(v_exp) * 100,
             np.array(v_rp) * 100,
             np.maximum(np.array(v_cvx) * 100, 0),
             labels=["Expectation", "Risk premium", "Convexity"],
             alpha=0.75)
ax.plot(taus, np.array(v_fwd) * 100, "k-", linewidth=2, label="Forward rate")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Rate (%)")
ax.set_title("Vasicek: Forward Rate Decomposition", fontweight="bold")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Gauss+
ax = axes[1]
ax.bar(taus_g, np.array(g_exp) * 100, label="Expectation", alpha=0.8)
ax.bar(taus_g, np.array(g_rp) * 100, bottom=np.array(g_exp) * 100, label="Risk premium", alpha=0.8)
ax.bar(taus_g, np.array(g_cvx) * 100,
       bottom=(np.array(g_exp) + np.array(g_rp)) * 100, label="Convexity", alpha=0.8)
ax.plot(taus_g, np.array(g_fwd) * 100, "ko-", markersize=5, linewidth=1.5, label="Forward rate")
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Rate (%)")
ax.set_title("Gauss+: Forward Rate Decomposition", fontweight="bold")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "forward_rate_decomposition.png"), dpi=150, bbox_inches="tight")
print("\nSaved: forward_rate_decomposition.png")
plt.show()

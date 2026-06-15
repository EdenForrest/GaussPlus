"""Gauss+ parameters calibrated from NY Fed zero-coupon yields.
Data: 2019-10-25 to 2025-10-17, 1479 trading days.
Three-stage procedure: Appendix A9.2.2 of Tuckman & Serrat (4e).
"""

CALIBRATED = dict(
    alpha_r=1.26028779,
    alpha_m=0.34912873,
    alpha_l=0.09545467,
    sigma_m=0.03855550,
    sigma_l=0.00914858,
    rho=-0.28222252,
    mu=0.06006250,
)

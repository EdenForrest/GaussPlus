"""Vasicek parameters calibrated from NY Fed zero-coupon yields.
Data: 2019-10-25 to 2025-10-17, 1479 trading days.
Stage 1: OLS on daily dr (k, sigma). Stage 2: theta from mean yield curve.
"""

CALIBRATED_VASICEK = dict(
    k=0.13154932,
    theta=0.00944075,
    sigma=0.00911283,
    r0=0.03630000,
)

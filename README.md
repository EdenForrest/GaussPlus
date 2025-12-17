# Gauss+ Term Structure Model (Gaussian Cascade)

Python implementation of the **Gauss+ (Gaussian cascade) term structure model**, a 3-factor Gaussian affine model for interest rates.

## Features

* Exact fit to 2- and 10-year zero-coupon yields
* Factor extraction for short-, medium-, and long-term rates
* Yield curve and forward rate computation
* Parameter calibration (α, σ, μ) to historical yield data
* Interest rate forecasting and scenario simulation

## Installation

Clone the repository and import the module:

```bash
git clone https://github.com/EdenForrest/GaussPlus.git
```

```python
from GaussPlusModel import GaussPlusModel
```

## Usage Example

```python
# Initialize model
model = GaussPlusModel(alpha=[0.9,0.6,0.01], sigma=[0.1,0.09,0.2], mu=0.03)

# Calibrate parameters (example)
alpha_hat = model.calibrate_alpha(delta_Y, delta_Yb, tau_bench=[2,10], maturities=taus)
sigma_hat = model.calibrate_sigma(delta_Yb)

# Extract factors for a given date
x = model.extract_factors(r_t, y2, y10)

# Compute zero-coupon yield curve
yields = model.yield_curve([1,2,5,10,20], x)
```

## Applications

* Pricing bonds, swaps, and interest rate derivatives
* Yield curve modeling and forecasting
* Risk management and scenario analysis

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

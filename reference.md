# Derivations — Vasicek and Gauss+

Direct transcription and exposition of Appendix A9 of Tuckman & Serrat, *Fixed Income Securities, 4e*. Equation numbers in parentheses refer to the book.

---

## 1. Vasicek model

### 1.1 Risk-neutral dynamics

$$
\mathrm{d}r = k(\theta - r)\,\mathrm{d}t + \sigma\,\mathrm{d}W
\tag{9.1}
$$

The drift pulls `r` toward the long-run level `θ` at speed `k`; `σ dW` is the diffusion. Under the real-world measure with a constant risk premium `λ` (in absolute rate units per year),

$$
\mathrm{d}r = k\bigl(r_\infty - r\bigr)\,\mathrm{d}t + \lambda\,\mathrm{d}t + \sigma\,\mathrm{d}W
        = k\!\left(\!r_\infty + \tfrac{\lambda}{k} - r\!\right)\!\mathrm{d}t + \sigma\,\mathrm{d}W,
\qquad \theta = r_\infty + \tfrac{\lambda}{k}.
\tag{9.2}-(9.3)
$$

### 1.2 Closed forms

$$
\mathbb{E}[r_t] = r_0 e^{-kt} + \theta(1 - e^{-kt}) \tag{9.4}
$$

$$
\mathrm{Var}[r_t] = \frac{\sigma^2}{2k}\bigl(1 - e^{-2kt}\bigr) \tag{9.5}
$$

Instantaneous continuously compounded forward rate:

$$
f(t) = \theta + e^{-kt}(r_0 - \theta) - \frac{\sigma^2}{2k^2}\bigl(1 + e^{-2kt} - 2e^{-kt}\bigr) \tag{9.6}
$$

The third term, written as `-(σ²/2k²)(1 - e^{-kt})²`, is the (negative) convexity adjustment. Continuously compounded zero yield:

$$
\hat r(t) = \theta + \frac{1 - e^{-kt}}{kt}(r_0 - \theta)
         - \frac{\sigma^2}{2k^2}\!\left(1 + \frac{1 - e^{-2kt}}{2kt} - 2\frac{1 - e^{-kt}}{kt}\right) \tag{9.7}
$$

Half-life of a shock:

$$
h = \frac{\ln 2}{k} \tag{9.8}
$$

### 1.3 Decomposition of `f(t)`

Combining (9.2) and (9.6) gives

$$
f(t) = \underbrace{r_\infty + e^{-kt}(r_0 - r_\infty)}_{\text{expectation}}
     + \underbrace{\frac{\lambda}{k}\bigl(1 - e^{-kt}\bigr)}_{\text{risk premium}}
     - \underbrace{\frac{\sigma^2}{2k^2}\bigl(1 - e^{-kt}\bigr)^2}_{\text{convexity}}.
$$

This is the curve plotted as Figure 9.4 of the text. The forward volatility term structure is `σ exp(-kt)`.

### 1.4 Binomial tree (A9.1)

Set the date-0 short rate to `r₀`. The expected short rate after one period of length `Δt` is

$$
r_0 + k(\theta - r_0)\Delta t. \tag{A9.1}
$$

Place the date-1 up- and down-states symmetrically around this drift:

$$
r_{\text{up}}^{(1)} = r_0 + k(\theta - r_0)\Delta t + \sigma\sqrt{\Delta t},\quad
r_{\text{dn}}^{(1)} = r_0 + k(\theta - r_0)\Delta t - \sigma\sqrt{\Delta t}.
$$

Move from each date-1 node to date 2 by recomputing the drift at that node. Generally the tree will not recombine in the naive construction; a recombining version solves transition probabilities to match drift and volatility. For node `r_i` on date 1 with drift `r_i + k(θ - r_i)Δt`, set children `r_up` and `r_dn` from the same symmetric construction and probabilities from

$$
p\,r_{\text{up}} + (1-p)\,r_{\text{dn}} = r_i + k(\theta - r_i)\Delta t,
\qquad
\sqrt{p(r_{\text{up}} - \bar r)^2 + (1-p)(r_{\text{dn}} - \bar r)^2} = \sigma\sqrt{\Delta t}. \tag{A9.3-A9.4}
$$

For the textbook parameters `r₀=2%, θ=11%, k=0.0165, σ=0.95%`, this yields `p = 0.4917, r_uu = 4.1949%`.

---

## 2. Gauss+ model

### 2.1 Cascade dynamics (Eqs. 9.9-9.12)

$$
\mathrm{d}r_t = -\alpha_r (m_t - r_t)\,\mathrm{d}t
$$

$$
\mathrm{d}m_t = -\alpha_m (l_t - m_t)\,\mathrm{d}t + \sigma_m\!\left(\rho\,\mathrm{d}W^1_t + \sqrt{1-\rho^2}\,\mathrm{d}W^2_t\right)
$$

$$
\mathrm{d}l_t = -\alpha_l (\mu - l_t)\,\mathrm{d}t + \sigma_l\,\mathrm{d}W^1_t
$$

$\mathbb{E}[\mathrm{d}W^1 \mathrm{d}W^2] = 0$. Three state variables (`r, m, l`) but two sources of risk. `r` carries no diffusion — this captures the policy-rate behavior of central banks and yields the empirical "hump" in the term structure of volatility.

### 2.2 Reduced form

Define the reduced-form state `X_t` so each component mean-reverts to a constant: `x_t = A(α) X_t + μ` (Eq. A9.5) with

$$
A(\alpha) = \begin{pmatrix}
1 & \frac{\alpha_r}{\alpha_r-\alpha_m} & \frac{\alpha_r \alpha_m}{(\alpha_r-\alpha_m)(\alpha_r-\alpha_l)} \\
0 & \frac{\alpha_r}{\alpha_r-\alpha_m} & \frac{\alpha_r \alpha_m}{(\alpha_r-\alpha_m)(\alpha_m-\alpha_l)} \\
0 & 0 & \frac{\alpha_r \alpha_m}{(\alpha_r-\alpha_l)(\alpha_m-\alpha_l)}
\end{pmatrix} \tag{A9.6}
$$

The reduced-form dynamics are

$$
\mathrm{d}X_t = -\mathrm{diag}(\alpha) X_t\,\mathrm{d}t + A(\alpha)^{-1} \Omega\,\mathrm{d}W_t \tag{A9.7}
$$

with

$$
\Omega(\sigma) = \begin{pmatrix}
0 & 0 & 0 \\
\rho\sigma_m & \sqrt{1-\rho^2}\,\sigma_m & 0 \\
\sigma_l & 0 & 0
\end{pmatrix}. \tag{A9.8}
$$

### 2.3 Affine zero-coupon bond price

Taking the risk-neutral expectation of `exp(-∫₀^τ r_s ds)`,

$$
P_t(\tau) = \mathbb{E}^Q\!\left[ e^{-\int_0^\tau r_s \mathrm{d}s} \mid X_t \right] = \exp(-y_t(\tau)\,\tau) \tag{A9.9}
$$

with the yield given by

$$
y_t(\tau) = \mu - C(\tau,\alpha,\sigma) + B(\tau,\alpha)\,X_t. \tag{A9.10}
$$

`B(τ,α)` is the three-vector with entries `(1 - e^{-α_i τ}) / (α_i τ)`, and

$$
C(\tau,\alpha,\sigma) = \sum_{i,j=1}^3 \frac{\Sigma_{ij}}{2\alpha_i\alpha_j}
   \!\left[1 - B_i(\tau) - B_j(\tau) - \frac{1 - e^{-(\alpha_i+\alpha_j)\tau}}{(\alpha_i+\alpha_j)\tau}\right] \tag{A9.11}
$$

where `Σ = A^{-1} Ω Ω^T A^{-T}`. The cascade form expression is then

$$
y_t(\tau) = \mu\bigl(1 - \Upsilon(\tau,\alpha)\mathbf 1\bigr) - C(\tau,\alpha,\sigma) + \Upsilon(\tau,\alpha) x_t \tag{A9.13}
$$

with the loading vector

$$
\Upsilon(\tau,\alpha) = B(\tau,\alpha)\,A(\alpha)^{-1} = (\Upsilon_s, \Upsilon_m, \Upsilon_l). \tag{A9.14}
$$

`Υ_s, Υ_m, Υ_l` are the partial derivatives of the zero yield with respect to the short, medium, and long factors. Their term-structure shape (Figure 9.7) gives the economic interpretation of the factors.

### 2.4 Forward rate

For the continuously compounded forward of tenor τ' starting at τ:

$$
f_t(\tau) = \mu\bigl(1 - \Upsilon'(\tau,\alpha,\tau')\mathbf 1\bigr) + \Upsilon'(\tau,\alpha,\tau')\,x_t - C'(\tau,\alpha,\sigma,\tau') \tag{A9.15}
$$

with the term-increments

$$
\Upsilon'(\tau,\alpha,\tau') = \frac{(B(\tau+\tau',\alpha) - B(\tau,\alpha))\,A(\alpha)^{-1}}{\tau'} \tag{A9.16}
$$

$$
C'(\tau,\alpha,\sigma,\tau') = \frac{C(\tau+\tau',\alpha,\sigma) - C(\tau,\alpha,\sigma)}{\tau'} \tag{A9.17}
$$

The first two terms on the right of (A9.15) are the risk-neutral expectation of the τ'-maturity yield at time `t+τ`; the third is the (positive) convexity advantage that drives a wedge between forward and futures rates.

### 2.5 Staged estimation (A9.2.2)

Stage 0 — data preparation. Net out the observed short rate from each observed yield: subtract `Υ_s(τ,α) r_t` from both sides of (A9.13). Drop `r_t` from the state, leaving `x_t = (m_t, l_t)` for stages 1-3.

Stage 1 — `α`. Run OLS of yield changes on changes in the benchmark 2y and 10y yields:

$$
\hat\beta = (\Delta y_b^\prime \Delta y_b)^{-1} \Delta y_b^\prime \Delta y. \tag{A9.21}
$$

Stack benchmark loadings as `Υ_b(α)` (2×2 after dropping the short-rate column). The model implies

$$
\Delta y(\tau) = \Upsilon(\alpha) \Upsilon_b(\alpha)^{-1} \Delta y_b. \tag{A9.19}
$$

Choose `α` to minimize

$$
\min_\alpha \big\| \Upsilon(\alpha) \Upsilon_b(\alpha)^{-1} - \hat\beta \big\|_2. \tag{A9.20}
$$

Stage 2 — `σ = (σ_m, σ_l, ρ)`. Match model and realized yield-change covariances at the benchmark maturities:

$$
\min_\sigma \big\| \Upsilon_b(\hat\alpha)\,\Omega(\sigma)\Omega(\sigma)^\prime\,\Upsilon_b(\hat\alpha)^\prime - \mathrm{diag}(\Delta y^\prime \Delta y) \cdot 252 / T \big\|. \tag{A9.22}
$$

Stage 3 — `μ`. Minimize total yield-fitting errors over the sample:

$$
\min_\mu \sum_{t=1}^T \|Y_t - y_t\|. \tag{A9.23}
$$

Stage 4 — exact fit. Each day, set `r_t` to the observed short rate and solve `(m_t, l_t)` from the two-equation system that matches the two- and ten-year forwards exactly. A weighted L2 with decay `0.8^{years_ago}` and an 8-year sample is the recommended balance between robustness and relevance.

The shifted long factor that the text plots is

$$
L(l_t) = \mu(1 - e^{-10\alpha_l}) + l_t e^{-10\alpha_l}. \tag{A9.24}
$$

### 2.6 Implied risk premia (A9.2.3)

Assume only the long factor carries a risk price. Applying Itô to (A9.9), the true-measure dynamics of a τ-maturity bond are

$$
\frac{\mathrm{d}P(t,\tau)}{P(t,\tau)} = \bigl[r_t + \lambda_t(\tau-t)\Upsilon_3(\tau-t,\alpha)\sigma_l\bigr]\mathrm{d}t
     - (\tau-t)\Upsilon(\tau-t,\alpha)\Omega\,\mathrm{d}W^*_t. \tag{A9.25}
$$

Consider the strategy: long one (τ+Δτ)-maturity zero, short one τ-maturity zero. Its cumulative real-world expected return from `t` to `τ` is

$$
\mathbb{E}_t[R_t^\tau] = \int_0^{\Delta\tau} \!\!
   \lambda_t (\tau + \Delta\tau - t - s)\,\Upsilon_3(\tau+\Delta\tau-t-s,\alpha)\,\sigma_l
$$

$$
\quad - \tfrac{1}{2}(\tau+\Delta\tau-t-s)^2\,\Upsilon(\tau+\Delta\tau-t-s,\alpha)\,\Omega\Omega'\,\Upsilon(\tau+\Delta\tau-t-s,\alpha)^\prime\,\mathrm{d}s
   = \lambda_t \cdot RP(t,\tau,\Delta\tau). \tag{A9.27}
$$

Since `λ_t` factors out, comparing two long maturities `τ' > τ` lets you solve for the price of risk:

$$
\lambda_t = \frac{f_t(\tau') - f_t(\tau)}{RP(t,\tau',\Delta\tau)/\Delta\tau - RP(t,\tau,\Delta\tau)/\Delta\tau}. \tag{A9.28}
$$

For any maturity `τ`, the real-world expected short rate at `τ` is

$$
\mathbb{E}_t[r_\tau] = f_t(\tau) - \lambda_t\,RP(t,\tau,\Delta\tau) + \text{convexity}. \tag{A9.29}
$$

Textbook sanity check (Appendix p. 489): `λ ≈ 0.09`, ten-year loading on the long factor `≈ 0.7`, `σ_l ≈ 100 bps`, ten-year duration `≈ 10`, so the implied ten-year risk premium is `0.09 × 0.01 × 10 × 0.7 ≈ 63 bps`, with a convexity advantage of about 24 bps. If the ten-year forward rate is 3%, the implied ten-year real-world expectation of the short rate is `3% + 0.24% − 0.63% ≈ 2.61%`.

---

## 3. Symbol glossary

| Symbol | Meaning |
| --- | --- |
| `r, m, l` | Cascade-form short, medium, long factors |
| `μ` | Reversion target of the long factor (Eq. 9.11) |
| `α_r, α_m, α_l` | Mean-reversion speeds, in 1/year |
| `σ_m, σ_l` | Diffusion magnitudes of `m, l`, annualized |
| `ρ` | Correlation of the `m` and `l` shocks |
| `A(α)` | Reduced-form → cascade mapping matrix (A9.6) |
| `Ω(σ)` | Diffusion matrix (A9.8) |
| `B(τ,α)` | Vector with entries `(1 - e^{-α_i τ})/(α_i τ)` |
| `Υ(τ,α)` | Loadings of yield on factors (A9.14) |
| `C(τ,α,σ)` | Convexity adjustment to the yield (A9.11) |
| `λ_t` | Price of risk on the long factor (Eq. A9.28) |
| `RP(t,τ,Δτ)` | Amount-of-risk integral (Eq. A9.27) |

# Mathematical Models of Interest Rates: Vasicek and Gauss+ Term Structure Models

## Table of Contents

1. [Mathematical Preliminaries](#mathematical-preliminaries)
2. [Vasicek One-Factor Model](#vasicek-one-factor-model)
3. [Gauss+ Three-Factor Model](#gauss-three-factor-model)
4. [Affine Term Structure Framework](#affine-term-structure-framework)
5. [Numerical Methods](#numerical-methods)
6. [Calibration Theory](#calibration-theory)

---

## Mathematical Preliminaries

### Probability Spaces and Filtrations

Let $(\Omega, \mathcal{F}, P)$ be a complete probability space equipped with a right-continuous filtration $\{\mathcal{F}_t\}_{t \geq 0}$ satisfying the usual conditions. We work with two measures:

- **Physical measure** $P$: the real-world probability measure governing actual observed market dynamics
- **Risk-neutral measure** $Q$: the martingale measure under which discounted bond prices are martingales

The Radon-Nikodým derivative defines the market price of risk and relates the two measures via the Girsanov theorem.

### Stochastic Calculus

For an Itô process $X_t$ satisfying
$$dX_t = \mu_t \, dt + \sigma_t \, dW_t$$

where $W_t$ is a standard Brownian motion, Itô's lemma states that for a $C^2$ function $f(t, x)$:
$$df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma_t \frac{\partial f}{\partial x} dW_t$$

### Term Structure Notation

- $r_t$: instantaneous short rate (spot rate at time $t$)
- $P(t, T)$: price at time $t$ of a zero-coupon bond maturing at time $T$
- $\tau = T - t$: time to maturity
- $y(t, T) = -\frac{1}{\tau} \ln P(t, T)$: continuously compounded spot (zero-coupon) yield
- $f(t, T) = -\frac{\partial}{\partial T} \ln P(t, T)$: instantaneous forward rate
- Relationship: $r_t = f(t, t)$

---

## Vasicek One-Factor Model

### 1.1 Model Dynamics

#### Risk-Neutral Dynamics

Under the risk-neutral measure $Q$, the short rate follows a mean-reverting Ornstein-Uhlenbeck process:

$$dr_t = k(\theta - r_t) dt + \sigma dW_t^Q \tag{1.1}$$

where:
- $k > 0$: **mean reversion speed** (1/time units)
- $\theta > 0$: **long-run mean** under risk-neutral measure (annualized decimal)
- $\sigma > 0$: **volatility** (annualized decimal)
- $W_t^Q$: standard Brownian motion under measure $Q$

#### Real-World Dynamics

Under the physical measure $P$, introduce a constant **market price of risk** $\lambda \in \mathbb{R}$ (in absolute rate units):

$$dr_t = k(r_\infty - r_t) dt + \sigma dW_t^P \tag{1.2}$$

where:
- $r_\infty$: long-run mean under physical measure
- $W_t^P$: standard Brownian motion under measure $P$

**Relation between measures:** The Girsanov theorem gives us
$$dW_t^Q = dW_t^P + \frac{\lambda}{\sigma} dt$$

Comparing coefficients in the drift, we have:
$$\theta = r_\infty + \frac{\lambda}{k} \tag{1.3}$$

### 1.2 Solution of the SDE

The explicit solution to equation (1.1) is a Gaussian random variable. For $s \leq t$:

$$r_t = e^{-k(t-s)} r_s + \theta(1 - e^{-k(t-s)}) + \sigma \int_s^t e^{-k(t-u)} dW_u^Q \tag{1.4}$$

**Conditional distribution:**
$$r_t | \mathcal{F}_s \sim \mathcal{N}\left( \mu_t^s, \sigma_t^{2,s} \right)$$

where:
$$\mu_t^s = E^Q[r_t | \mathcal{F}_s] = e^{-k(t-s)} r_s + \theta(1 - e^{-k(t-s)}) \tag{1.5}$$

$$\sigma_t^{2,s} = \text{Var}^Q[r_t | \mathcal{F}_s] = \frac{\sigma^2}{2k} (1 - e^{-2k(t-s)}) \tag{1.6}$$

### 1.3 Affine Bond Pricing

The bond price is given by the affine formula:
$$P(t, T) = e^{A(T-t) - B(T-t) r_t} \tag{1.7}$$

where the functions $A(\tau)$ and $B(\tau)$ satisfy the Riccati ODEs derived from the PDE for bond pricing.

#### Derivation via PDE

Let $P(t, T) = P(t, r_t; T)$. By Itô's lemma:
$$\frac{\partial P}{\partial t} + k(\theta - r) \frac{\partial P}{\partial r} + \frac{1}{2}\sigma^2 \frac{\partial^2 P}{\partial r^2} = r P \tag{1.8}$$

Substituting the ansatz $P(t, r, T) = e^{A(\tau) - B(\tau)r}$ where $\tau = T - t$:

$$\frac{\partial P}{\partial t} = -\frac{dA}{d\tau} P + r \frac{dB}{d\tau} P$$

$$\frac{\partial P}{\partial r} = -B P, \quad \frac{\partial^2 P}{\partial r^2} = B^2 P$$

Substituting into (1.8) and dividing by $P$:
$$-\frac{dA}{d\tau} + r \frac{dB}{d\tau} - k(\theta - r) B + \frac{1}{2}\sigma^2 B^2 = r$$

Matching terms independent and dependent on $r$:

**Coefficient of $r$:**
$$\frac{dB}{d\tau} + kB - 1 = 0 \tag{1.9a}$$

**Constant term:**
$$-\frac{dA}{d\tau} - k\theta B + \frac{1}{2}\sigma^2 B^2 = 0 \tag{1.9b}$$

#### Solving for B(τ)

Equation (1.9a) is a first-order linear ODE with boundary condition $B(0) = 0$:
$$\frac{dB}{d\tau} = 1 - kB$$

Solution by separation of variables:
$$B(\tau) = \frac{1 - e^{-k\tau}}{k} \tag{1.10}$$

#### Solving for A(τ)

Substituting (1.10) into (1.9b):
$$\frac{dA}{d\tau} = k\theta B(\tau) - \frac{1}{2}\sigma^2 B^2(\tau)$$

$$= k\theta \cdot \frac{1 - e^{-k\tau}}{k} - \frac{1}{2}\sigma^2 \left( \frac{1 - e^{-k\tau}}{k} \right)^2$$

$$= \theta(1 - e^{-k\tau}) - \frac{\sigma^2}{2k^2}(1 - e^{-k\tau})^2$$

Integrating from 0 to $\tau$:
$$A(\tau) = \theta \left[ \tau - \frac{1 - e^{-k\tau}}{k} \right] - \frac{\sigma^2}{2k^2} \left[ \tau - 2 \cdot \frac{1 - e^{-k\tau}}{k} + \frac{1 - e^{-2k\tau}}{2k} \right]$$

$$= \left(\theta - \frac{\sigma^2}{2k^2}\right) \tau + \left(\frac{\sigma^2}{k^2} - \theta\right) \frac{1 - e^{-k\tau}}{k} - \frac{\sigma^2}{4k^3}(1 - e^{-2k\tau}) \tag{1.11}$$

### 1.4 Zero-Coupon Yield

From the bond price $P(t, T) = e^{A(\tau) - B(\tau)r_t}$, the continuously compounded yield is:
$$y(t, T) = -\frac{\ln P(t, T)}{\tau} = \frac{B(\tau) r_t - A(\tau)}{\tau} \tag{1.12}$$

Substituting $B(\tau)$ and $A(\tau)$:
$$y(t, T) = \theta + \frac{1 - e^{-k\tau}}{k\tau}(r_t - \theta) - \frac{\sigma^2}{2k^2}\left[1 + \frac{1 - e^{-2k\tau}}{2k\tau} - \frac{2(1 - e^{-k\tau})}{k\tau}\right] \tag{1.13}$$

The last term is the **convexity correction**.

### 1.5 Forward Rates

The instantaneous forward rate is:
$$f(t, T) = -\frac{\partial}{\partial T} \ln P(t, T) = \frac{\partial A}{\partial \tau} - r_t \frac{\partial B}{\partial \tau} \tag{1.14}$$

Computing derivatives:
$$\frac{\partial B}{\partial \tau} = e^{-k\tau}$$

$$\frac{\partial A}{\partial \tau} = \left(\theta - \frac{\sigma^2}{2k^2}\right) + \left(\frac{\sigma^2}{k^2} - \theta\right) e^{-k\tau} - \frac{\sigma^2}{2k^2} e^{-2k\tau}$$

Therefore:
$$f(t, T) = \theta + e^{-k\tau}(r_t - \theta) - \frac{\sigma^2}{2k^2}(1 + e^{-2k\tau} - 2e^{-k\tau}) \tag{1.15}$$

The third term is the **convexity adjustment** (always negative, reflecting Jensen's inequality).

### 1.6 Forward Rate Decomposition

Under the real-world measure with constant risk premium $\lambda$:

**Expected short rate (physical measure):**
$$E_P[r_t | \mathcal{F}_s] = r_\infty + e^{-k(t-s)}(r_s - r_\infty) \tag{1.16}$$

where $r_\infty = \theta - \lambda/k$ is the long-run mean under $P$.

**Risk premium:**
$$\text{Risk Premium} = \frac{\lambda}{k}(1 - e^{-k\tau}) \tag{1.17}$$

This represents compensation for bearing interest rate risk. When $\lambda > 0$, longer maturities have higher risk premium.

**Convexity correction:**
$$\text{Convexity} = -\frac{\sigma^2}{2k^2}(1 - e^{-k\tau})^2 \tag{1.18}$$

The forward rate decomposes as:
$$f(t, T) = \underbrace{E_P[r_T]}_{\text{expectation}} + \underbrace{\frac{\lambda}{k}(1 - e^{-k\tau})}_{\text{risk premium}} - \underbrace{\frac{\sigma^2}{2k^2}(1 - e^{-k\tau})^2}_{\text{convexity}} \tag{1.19}$$

### 1.7 Bond Price Distribution and Variance

The variance of the short rate is:
$$\text{Var}^Q[r_t | r_0] = \frac{\sigma^2}{2k}(1 - e^{-2kt}) \tag{1.20}$$

The standard deviation of bond log-prices grows with maturity. Specifically, the volatility of the $\tau$-maturity forward rate is:
$$\sigma_f(\tau) = \sigma e^{-k\tau} \tag{1.21}$$

### 1.8 Half-Life of a Shock

An important quantity for practitioners is the half-life of a shock to the short rate—the time it takes for a 1 basis point shock to dissipate to 0.5 basis points:
$$\text{Half-life} = \frac{\ln 2}{k} \tag{1.22}$$

### 1.9 Bond Pricing via Monte Carlo

For path-dependent derivatives or when closed-form solutions are unavailable, simulate the short rate:

$$r_{t+\Delta t} = r_t e^{-k\Delta t} + \theta(1 - e^{-k\Delta t}) + \sigma \sqrt{\frac{1 - e^{-2k\Delta t}}{2k}} \, Z$$

where $Z \sim \mathcal{N}(0,1)$.

Integrate the short rate along each path:
$$\int_0^T r_s \, ds$$

and compute the bond price as:
$$P(0, T) = E\left[ \exp\left( -\int_0^T r_s \, ds \right) \right]$$

**Standard error:** For $n_{\text{paths}}$ paths, the standard error is $\sigma / \sqrt{n_{\text{paths}}}$.

### 1.10 Binomial Tree Construction

For the Vasicek model, construct a recombining binomial tree as follows:

At each node, given a rate $r$ and expected next rate $\mu = r + k(\theta - r) \Delta t$, branch into up and down states:
- $r_{\text{up}} = \mu + \sigma\sqrt{\Delta t}$
- $r_{\text{down}} = \mu - \sigma\sqrt{\Delta t}$

The probability $p$ of the up move is determined by matching the conditional expectation:
$$p \cdot r_{\text{up}} + (1-p) \cdot r_{\text{down}} = \mu$$

Solving:
$$p = \frac{\mu - r_{\text{down}}}{r_{\text{up}} - r_{\text{down}}} = \frac{\sigma\sqrt{\Delta t} + k(\theta - r)\Delta t}{2\sigma\sqrt{\Delta t}}$$

Ensure $0 \leq p \leq 1$ by construction choice.

---

## Gauss+ Three-Factor Model

### 2.1 Model Dynamics

#### Cascade Form

The Gauss+ model represents the term structure using three factors in a **cascade form**:
$$x_t = (r_t, m_t, l_t)^T$$

where:
- $r_t$: **short-rate factor** (policy rate, directly observable)
- $m_t$: **medium-term factor** (monetary policy expectations)
- $l_t$: **long-term factor** (long-run equilibrium rate expectations)

The dynamics under the risk-neutral measure are:
$$dr_t = \alpha_r (\mu_r - r_t) dt + 0 \, dW_t \tag{2.1a}$$

$$dm_t = \alpha_m (m_t^* - m_t) dt + \sigma_m \, dW_t^m \tag{2.1b}$$

$$dl_t = \alpha_l (l_t^* - l_t) dt + \sigma_l \, dW_t^l \tag{2.1c}$$

where $\mu_r$, $m_t^*$, $l_t^*$ are long-run means, and the Brownian shocks satisfy:
$$dW_t^m dW_t^l = \rho \, dt \tag{2.1d}$$

with correlation $\rho \in (-1, 1)$.

**Key feature:** The short rate $r_t$ has zero diffusion—it's directly controlled by policy. The other factors have correlated diffusions.

#### Reduced Form Transformation

The cascade form is related to the **reduced form** $y_t = (y_t^{(1)}, y_t^{(2)}, y_t^{(3)})$ via the invertible transformation:
$$y_t = A^{-1} x_t \tag{2.2}$$

where $A$ is the cascade-to-reduced-form matrix:
$$A = \begin{pmatrix}
1 & -\frac{\alpha_r}{\alpha_r - \alpha_m} & \frac{\alpha_r \alpha_m}{(\alpha_r - \alpha_m)(\alpha_r - \alpha_l)} \\
0 & \frac{\alpha_r}{\alpha_r - \alpha_m} & \frac{\alpha_r \alpha_m}{(\alpha_r - \alpha_m)(\alpha_m - \alpha_l)} \\
0 & 0 & \frac{\alpha_r \alpha_m}{(\alpha_r - \alpha_l)(\alpha_m - \alpha_l)}
\end{pmatrix}$$

The reduced form factors satisfy:
$$dy_t = -\text{diag}(\alpha_r, \alpha_m, \alpha_l) \, y_t \, dt + \text{d}\mathcal{M}_t$$

where $\text{d}\mathcal{M}_t$ are the transformed martingale increments.

### 2.2 Bond Pricing under Affine Term Structure

The short rate is the first component of the cascade form:
$$r_t = r_t$$

By the general theory of affine models, the bond price has the form:
$$P(t, T) = e^{A(\tau) - B_r(\tau) r_t - B_m(\tau) m_t - B_l(\tau) l_t} \tag{2.3}$$

where $\tau = T - t$ and the functions $A(\tau)$, $B_r(\tau)$, $B_m(\tau)$, $B_l(\tau)$ satisfy coupled Riccati ODEs.

#### Riccati Equations

For the cascade form (2.1), the Riccati equations are:

$$\frac{dB_r}{d\tau} + \alpha_r B_r - 1 = 0, \quad B_r(0) = 0 \tag{2.4a}$$

$$\frac{dB_m}{d\tau} + \alpha_m B_m = 0, \quad B_m(0) = 0 \tag{2.4b}$$

$$\frac{dB_l}{d\tau} + \alpha_l B_l = 0, \quad B_l(0) = 0 \tag{2.4c}$$

#### Solutions

$$B_r(\tau) = \frac{1 - e^{-\alpha_r \tau}}{\alpha_r} \tag{2.5a}$$

$$B_m(\tau) = -e^{-\alpha_m \tau} = \frac{e^{-\alpha_m \tau} - 1}{\alpha_m} \cdot \alpha_m = 0 \text{ (with appropriate scaling)} \tag{2.5b}$$

More carefully: $B_m(\tau) = 0$ (the medium term doesn't directly enter the yield via this form), but enters through the correlation structure.

Actually, the correct formulation requires solving the full system. Let me reconsider.

For a three-factor model with state $x = (r, m, l)$ and short rate $r_t = c_0 + c^T x_t$ with $c = (1, 0, 0)^T$, the bond price is:
$$P(t, T) = \exp(A(\tau) + B(\tau)^T x_t) \tag{2.6}$$

where $B(\tau) = (B_r(\tau), B_m(\tau), B_l(\tau))^T$ satisfies:
$$\frac{dB}{d\tau} + A_0^T B - c = 0, \quad B(0) = 0$$

where $A_0$ is the drift matrix under the risk-neutral measure.

### 2.3 Yield and Forward Rate Loadings

Define the **yield loadings** (also called factor loadings or upsilon):
$$\Upsilon(\tau) = (1 - e^{-\alpha_r \tau})/(\alpha_r \tau), (1 - e^{-\alpha_m \tau})/(\alpha_m \tau), (1 - e^{-\alpha_l \tau})/(\alpha_l \tau)$$

Wait, this applies to each factor separately. In the cascade form, we need to account for the transformation $A^{-1}$.

The yield in the cascade form is:
$$y(t, T) = \mu(1 - \Upsilon(\tau) \cdot \mathbf{1}) - C(\tau) + \Upsilon(\tau)^T A^{-1} x_t \tag{2.7}$$

where:
- $\mu$ is the long-run mean under the risk-neutral measure
- $\Upsilon(\tau) = (B_r(\tau)/\tau, B_m(\tau)/\tau, B_l(\tau)/\tau)$
- $C(\tau)$ is the convexity correction
- $\mathbf{1} = (1, 1, 1)^T$

### 2.4 Convexity Correction

The convexity correction for zero-coupon yields is:
$$C(\tau) = -\frac{1}{2\tau} \sum_{i,j} \Sigma_{ij} b(\tau, \alpha_i) b(\tau, \alpha_j) \tag{2.8}$$

where:
- $b(\tau, \alpha) = (1 - e^{-\alpha \tau})/\alpha$ is the price loading
- $\Sigma = A^{-1} \Omega \Omega^T (A^{-1})^T$ is the covariance matrix of reduced-form innovations

The instantaneous volatility matrix is:
$$\Omega = \begin{pmatrix}
0 & 0 & 0 \\
\rho \sigma_m & \sqrt{1 - \rho^2} \sigma_m & 0 \\
\sigma_l & 0 & 0
\end{pmatrix} \tag{2.9}$$

reflecting that $r$ has no volatility, $m$ and $l$ are correlated with correlation $\rho$.

### 2.5 Forward Rates

The forward rate from time $T$ to $T + \tau'$ as seen from time $t$ is:
$$f(t, T; \tau') = \mu(1 - \Upsilon_f(\tau, \tau')^T \mathbf{1}) + \Upsilon_f(\tau, \tau')^T A^{-1} x_t - C_f(\tau, \tau') \tag{2.10}$$

where $\tau = T - t$ is the start date, and:
$$\Upsilon_f(\tau, \tau') = \frac{1}{\tau'} [B(\tau + \tau') - B(\tau)] \tag{2.11}$$

is the forward-rate loading, and:
$$C_f(\tau, \tau') = \frac{C(\tau + \tau') - C(\tau)}{\tau'} \tag{2.12}$$

### 2.6 Factor Extraction from Yields

Given observed yields at benchmarks (e.g., 2Y and 10Y), extract the unobserved factors $m$ and $l$ by exact fit at these maturities while treating $r$ as observable.

From the yield equation (2.7), at benchmark maturities $\tau_1$ and $\tau_2$:
$$y_i = \mu(1 - \Upsilon_i \cdot \mathbf{1}) - C_i + \Upsilon_i^T A^{-1} \begin{pmatrix} r \\ m \\ l \end{pmatrix}$$

where subscript $i$ denotes evaluation at $\tau_i$.

In matrix form:
$$\begin{pmatrix} y_1 - y_1^{(r)} \\ y_2 - y_2^{(r)} \end{pmatrix} = \begin{pmatrix} \Upsilon_1^{(m)} & \Upsilon_1^{(l)} \\ \Upsilon_2^{(m)} & \Upsilon_2^{(l)} \end{pmatrix} A^{-1}_{1:2, 1:2} \begin{pmatrix} m \\ l \end{pmatrix}$$

where $y_i^{(r)}$ is the contribution from the short rate. Solve this $2 \times 2$ system:
$$\begin{pmatrix} m \\ l \end{pmatrix} = U^{-1} (y - y^{(r)})$$

where $U$ is the $2 \times 2$ loading matrix.

### 2.7 Three-Stage Calibration

#### Stage 1: Calibrate Mean-Reversion Speeds ($\alpha_r, \alpha_m, \alpha_l$)

**Principle:** Yield changes across maturities are driven by shocks to the factors. The empirical yield-change regression slopes reveal the factor loading structure.

For each yield maturity $\tau_i$ (not at benchmarks), regress changes in yield on changes in benchmark yields:
$$\Delta y_i = \beta_1 \Delta y_1 + \beta_2 \Delta y_2 + \epsilon_i$$

where $\beta = (B_1^T \Sigma B_1)^{-1} B_1^T \Sigma B_i$, and $B_j$ are the benchmark loadings.

The model-implied slope is:
$$\beta_{\text{model}} = \Upsilon_i A^{-1} (A^{-1T} \Upsilon_{1:2}^T)^{-1}$$

which depends on the cascade parameters. Optimize $\alpha = (\alpha_r, \alpha_m, \alpha_l)$ to match empirical and model slopes:
$$\min_{\alpha} \|\beta_{\text{data}} - \beta_{\text{model}}(\alpha)\|_F^2 \tag{2.13}$$

#### Stage 2: Calibrate Volatilities and Correlation ($\sigma_m, \sigma_l, \rho$)

**Principle:** The covariance matrix of yield changes encodes the volatility parameters.

Compute the realized covariance of benchmark yield changes from data:
$$\hat{\Sigma}_y = \frac{1}{T} \sum_{t=1}^T \Delta y_t \Delta y_t^T$$

(annualized by multiplying by 252 for daily data).

The model-implied covariance is:
$$\Sigma_y = \Upsilon_{1:2} A^{-1} \Omega \Omega^T (A^{-1})^T \Upsilon_{1:2}^T \times 252$$

Optimize $\sigma = (\sigma_m, \sigma_l, \rho)$ to minimize:
$$\min_{\sigma} \|\Sigma_y^{\text{data}} - \Sigma_y^{\text{model}}(\sigma)\|_F^2 \tag{2.14}$$

#### Stage 3: Calibrate Long-Run Mean ($\mu$)

**Principle:** The long-run mean is the level around which yields revert. Fit it to match average yield levels.

For observed yields $y_t^{\tau_i}$ at maturity $\tau_i$:
$$\min_{\mu} \sum_t \left( y_t^{\tau_i} - \hat{y}_t(\mu, \alpha, \sigma) \right)^2$$

where $\hat{y}_t$ is the model yield. This is typically done at the benchmark maturities or across all maturities with weights:
$$\min_{\mu} \sum_i w_i \sum_t \left( y_t^{\tau_i} - \hat{y}_t(\mu) \right)^2$$

Often use exact fit at benchmarks with exponential decay weighting to prefer long-maturity fit.

---

## Affine Term Structure Framework

### 3.1 Generalized Affine Model

An **affine model** is a term structure model where the short rate and bond prices have affine (linear plus constant) dependence on a state vector $x_t \in \mathbb{R}^n$.

**Definition:** The short rate is:
$$r_t = \rho_0 + \rho^T x_t \tag{3.1}$$

The bond price is:
$$P(t, T) = \exp(A(\tau) + B(\tau)^T x_t) \tag{3.2}$$

where $A: \mathbb{R}_+ \to \mathbb{R}$ and $B: \mathbb{R}_+ \to \mathbb{R}^n$.

### 3.2 Fundamental PDE

Let the state vector follow:
$$dx_t = \mu(x_t) dt + \sigma(x_t) dW_t \tag{3.3}$$

The bond price satisfies the PDE:
$$\frac{\partial P}{\partial t} + \mu^T \nabla_x P + \frac{1}{2} \text{Tr}(\sigma \sigma^T \nabla_x^2 P) = r P \tag{3.4}$$

with boundary condition $P(T, T) = 1$.

### 3.3 Riccati Equations

For an affine drift:
$$\mu(x) = K_0 + K_1 x$$

and a separable (piece-wise affine) diffusion coefficient satisfying certain growth conditions, substituting the ansatz (3.2) into (3.4) yields:

$$\frac{dA}{d\tau} + K_0^T B - \frac{1}{2} B^T H_0 B = 0, \quad A(0) = 0 \tag{3.5a}$$

$$\frac{dB}{d\tau} + K_1^T B + H_1^T B - \rho = 0, \quad B(0) = 0 \tag{3.5b}$$

where the diffusion matrix determines $H_0$ and $H_1$ in an affine-compatible way.

### 3.4 Classification of Affine Models

| Model | State Space | Drift | Diffusion | Affinity |
|-------|------------|-------|-----------|----------|
| **Vasicek** | $r \in \mathbb{R}$ | $k(\theta - r)$ | $\sigma$ (const) | Fully affine |
| **CIR** | $r \in \mathbb{R}_+$ | $k(\theta - r)$ | $\sigma \sqrt{r}$ | Square-root affine |
| **Gauss+** | $(r, m, l) \in \mathbb{R}^3$ | Cascade linear | Piece-wise const | Fully affine |
| **Multifactor CIR** | $x \in \mathbb{R}^n_+$ | Linear | Diagonal square-root | Square-root affine |

---

## Numerical Methods

### 4.1 Optimization: Nelder-Mead Simplex

The **Nelder-Mead method** is a derivative-free optimization algorithm suitable for non-linear least-squares problems.

**Algorithm:**
1. Initialize a simplex with $n+1$ vertices (for $n$-dimensional problem)
2. Evaluate the objective at each vertex
3. Rank vertices by objective value: best, good, ..., worst
4. Attempt **reflection** across the centroid of the $n$ best vertices:
   $$x_r = \bar{x} + \alpha(x_{\text{worst}} - \bar{x})$$
5. Based on objective improvement, perform **expansion, contraction, or shrinkage**
6. Repeat until convergence (simplex volume $< \epsilon$)

**Pros:** Robust, no derivatives needed  
**Cons:** Slow convergence (superlinear but not quadratic), unreliable in high dimensions ($n > 5$)

### 4.2 Optimization: L-BFGS-B

The **L-BFGS-B method** is a limited-memory quasi-Newton algorithm for bounded optimization.

Uses a low-rank approximation to the Hessian:
$$H_k \approx (I - \rho s_k^T) H_{k-1} (I - \rho y_k s_k^T) + \rho s_k s_k^T$$

where $s_k = x_{k+1} - x_k$ and $y_k = \nabla f_{k+1} - \nabla f_k$.

**Pros:** Superlinear convergence, handles bounds via active-set method, efficient  
**Cons:** Requires gradients (or numerical approximation), more complex to implement

### 4.3 OLS Regression

For factor loading extraction:
$$\min_{\beta} \sum_t (y_t - X_t \beta)^2$$

**Normal equations:**
$$\beta = (X^T X)^{-1} X^T y$$

**Numerical stability:** Use QR decomposition or SVD instead of inverting $X^T X$ directly.

SVD: $X = U \Sigma V^T$, then $\beta = V \Sigma^{-1} U^T y$.

### 4.4 Monte Carlo Simulation

**Exact simulation for Vasicek:**
$$r_{t+\Delta t} = r_t e^{-k\Delta t} + \theta (1 - e^{-k\Delta t}) + \sigma \sqrt{\frac{1 - e^{-2k\Delta t}}{2k}} Z$$

where $Z \sim \mathcal{N}(0, 1)$. This has **zero discretization error** for the marginal distribution.

**Convergence:** With $m$ Monte Carlo paths, the standard error is $O(m^{-1/2})$. For $95\%$ confidence, need $m \sim 40000$ for 1 basis point accuracy.

---

## Calibration Theory

### 5.1 Objective Functions

#### Least-Squares Fit
$$\text{RSS} = \sum_i w_i (y_i^{\text{model}} - y_i^{\text{obs}})^2 \tag{5.1}$$

Common choices for weights:
- Uniform: $w_i = 1$
- Dollar-duration weighted: $w_i = \text{DV01}_i^2$
- Inverse variance: $w_i = 1/\text{Var}[y_i]$
- Maturity-declining: $w_i = e^{-\lambda \tau_i}$ for $\lambda > 0$

#### Matching Statistics
**Volatility matching:**
$$\text{Objective} = \|\text{Cov}[\Delta y]^{\text{data}} - \text{Cov}[\Delta y]^{\text{model}}\|_F^2$$

**Slope matching (OLS):**
$$\text{Objective} = \|\beta^{\text{data}} - \beta^{\text{model}}\|^2$$

### 5.2 Identifiability Issues

**Problem:** Multiple parameter combinations may yield similar fit.

**Example (Vasicek):** For a flat yield curve, only $\theta$ is identified; $k$ and $\sigma$ are not.

**Solution:** 
- Fix some parameters (e.g., $k$ from historical data) and calibrate others
- Use additional information (e.g., swaptions) to pin down $\sigma$
- Impose cross-sectional/time-series constraints

### 5.3 Calibration to Derivative Prices

For more accurate parameter estimation, calibrate to **options on bonds** (swaptions, cap/floor, etc.).

**Swaption pricing:** A European swaption exercisable at $T$ into a swap paying fixed $K$ has value:
$$V = P(0, T) \mathbb{E}^T \left[ \left( A_T - \sum_{j=1}^n P(T, T_j) \right)^+ \right]$$

where $A_T$ is the annuity and the expectation is under the $T$-forward measure.

This depends on the **volatility of forward rates**, which is highly sensitive to $\sigma$ (and to $k$ through the decay rate).

### 5.4 Time-Varying Calibration

In practice, recalibrate parameters at regular intervals (daily, weekly, monthly) to track regime changes.

**Exponential moving average of volatility:**
$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) (\Delta y_t)^2$$

with decay $\lambda \approx 0.94$ (half-life $\sim 11$ days).

---

## References and Further Reading

1. **Tuckman, B., & Serrat, A.** (2011). *Fixed Income Securities: Tools for Today's Markets* (4th ed.). Hoboken, NJ: Wiley.
   - Core reference for Vasicek and Gauss+ models

2. **Vasicek, O.** (1977). An equilibrium characterization of the term structure. *Journal of Financial Economics*, 5(2), 177-188.
   - Original Vasicek model paper

3. **Duffie, D.** (2001). *Dynamic Asset Pricing Theory* (3rd ed.). Princeton, NJ: Princeton University Press.
   - Affine term structure theory and martingale pricing

4. **Cairns, A. J. G.** (2004). *Interest Rate Models: An Introduction*. Princeton, NJ: Princeton University Press.
   - Practical guide to term structure models

5. **James, J., & Webber, N.** (2000). *Interest Rate Modelling*. Chichester, UK: Wiley.
   - Comprehensive treatment of one- and multi-factor models

6. **Brigo, D., & Mercurio, F.** (2006). *Interest Rate Models - Theory and Practice* (2nd ed.). Berlin: Springer.
   - Advanced topics, smile and skew modeling

---

## Appendix: Key Formulas Summary

### Vasicek

| Quantity | Formula |
|----------|---------|
| **Bond Price** | $P(t,T) = \exp(A(\tau) - B(\tau) r_t)$ where $\tau = T-t$ |
| **B function** | $B(\tau) = \frac{1 - e^{-k\tau}}{k}$ |
| **Spot Yield** | $y(t,T) = \theta + \frac{1-e^{-k\tau}}{k\tau}(r_t - \theta) - C(\tau)$ |
| **Forward Rate** | $f(t,T) = \theta + e^{-k\tau}(r_t - \theta) - C_f(\tau)$ |
| **Half-life** | $T_{1/2} = \frac{\ln 2}{k}$ |
| **Short Rate Variance** | $\text{Var}[r_t] = \frac{\sigma^2}{2k}(1 - e^{-2kt})$ |

### Gauss+

| Quantity | Formula |
|----------|---------|
| **Bond Price** | $P(t,T) = \exp(A(\tau) - B_r(\tau) r_t - B_m(\tau) m_t - B_l(\tau) l_t)$ |
| **B functions** | $B_i(\tau) = \frac{1 - e^{-\alpha_i \tau}}{\alpha_i}$ |
| **Yield Loadings** | $\Upsilon(\tau) = A^{-1} \cdot (B_r/\tau, B_m/\tau, B_l/\tau)^T$ |
| **Zero Yield** | $y(t,T) = \mu(1 - \Upsilon \cdot \mathbf{1}) - C(\tau) + \Upsilon^T x_t$ |
| **Convexity** | $C(\tau) = -\frac{1}{2\tau} B^T \Sigma B$ where $\Sigma = A^{-1} \Omega \Omega^T (A^{-1})^T$ |

---

*Document generated for graduate-level study of fixed-income term structure models.*

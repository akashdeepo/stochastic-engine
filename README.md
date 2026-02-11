# Stochastic Engine

A comprehensive, Pythonic quantitative finance library for stochastic processes, option pricing, and risk metrics.

**No more Googling formulas. No more reimplementing Black-Scholes. Just clean, intuitive APIs.**

## Installation

```bash
pip install stochastic-engine
```

With real market data support:

```bash
pip install stochastic-engine[data]
```

Or install from source:

```bash
git clone https://github.com/akashdeepo/stochastic-engine.git
cd stochastic-engine
pip install -e ".[all]"
```

## Quick Start

### Simulate Stock Prices (GBM)

```python
from stochastic_engine import GBM

# Simulate 1000 paths of stock prices over 1 year
gbm = GBM(S0=100, mu=0.05, sigma=0.2)
paths = gbm.simulate(T=1, steps=252, n_paths=1000)

# paths.shape = (1000, 253)  # 1000 paths, 253 time points
```

### Price Options with Black-Scholes

```python
from stochastic_engine import BlackScholes

# Price a European call option
option = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)

print(f"Price: ${option.price:.2f}")      # $8.02
print(f"Delta: {option.delta:.4f}")       # 0.5462
print(f"Gamma: {option.gamma:.4f}")       # 0.0188
print(f"Vega:  {option.vega:.4f}")        # 0.3745
print(f"Theta: {option.theta:.4f}")       # -0.0180

# All Greeks at once
print(option.greeks)
```

### Correlated Multi-Asset Simulation

```python
import numpy as np
from stochastic_engine import CorrelatedGBM

corr = np.array([[1.0, 0.6, 0.3],
                 [0.6, 1.0, 0.5],
                 [0.3, 0.5, 1.0]])

portfolio = CorrelatedGBM(
    S0=np.array([100, 50, 200]),
    mu=np.array([0.05, 0.08, 0.03]),
    sigma=np.array([0.2, 0.3, 0.15]),
    correlation=corr,
)
paths = portfolio.simulate(T=1, steps=252, n_paths=10000)
# paths.shape = (3, 10000, 253)  # 3 assets, 10000 paths
```

### Price Barrier and Digital Options

```python
from stochastic_engine import BarrierOption, DigitalOption

# Down-and-out barrier call
barrier = BarrierOption(S=100, K=105, B=80, T=1, r=0.05, sigma=0.2,
                        barrier_type="down-and-out")
result = barrier.price()  # Closed-form (Reiner-Rubinstein)
print(f"Barrier call: ${result.price:.2f}")

# Cash-or-nothing digital call
digital = DigitalOption(S=100, K=105, T=1, r=0.05, sigma=0.2,
                        Q=100, digital_type="cash-or-nothing")
print(f"Digital call: ${digital.price:.2f}")
print(f"Delta: {digital.delta:.4f}")
```

### Calibrate SABR and Heston Models

```python
from stochastic_engine import SABR, HestonCalibrator

# Calibrate SABR to market smile
strikes = np.linspace(80, 120, 21)
market_vols = [...]  # Your market implied vols
result = SABR.fit(f=100, strikes=strikes, T=1.0, market_vols=market_vols, beta=0.5)
print(f"SABR: alpha={result.alpha:.4f}, rho={result.rho:.4f}, nu={result.nu:.4f}")

# Calibrate Heston to IV surface
calibrator = HestonCalibrator(S=100, r=0.05)
heston_result = calibrator.calibrate(strikes, maturities, market_ivs)
heston = heston_result.to_heston(S0=100, mu=0.05)  # Ready to simulate
```

### Work with Real Market Data

```python
from stochastic_engine.data import fetch_prices, returns_from_prices, estimate_gbm_params

# Fetch and estimate parameters
prices = fetch_prices("AAPL", period="1y")  # Requires: pip install stochastic-engine[data]
returns = returns_from_prices(prices)
params = estimate_gbm_params(prices)

# Use estimated parameters directly
gbm = GBM(S0=params["S0"], mu=params["mu"], sigma=params["sigma"])
```

### Calculate Risk Metrics

```python
from stochastic_engine import VaR, CVaR
from stochastic_engine.risk import sharpe_ratio, max_drawdown

# Historical VaR
returns = [...]  # Your return data
var = VaR(returns, confidence=0.95)
print(f"95% VaR: {var.historical():.2%}")
print(f"95% CVaR: {CVaR(returns, confidence=0.95).historical():.2%}")

# Performance metrics
print(f"Sharpe Ratio: {sharpe_ratio(returns):.2f}")
print(f"Max Drawdown: {max_drawdown(prices):.2%}")
```

## Features

### Stochastic Processes
- **GBM** - Geometric Brownian Motion (stock prices)
- **OrnsteinUhlenbeck** - Mean-reverting process (interest rates, pairs trading)
- **Heston** - Stochastic volatility model with analytical European pricing
- **Vasicek** - Interest rate model with bond pricing and yield curves
- **CIR** - Cox-Ingersoll-Ross rate model (non-negative rates, Feller condition)
- **MertonJumpDiffusion** - GBM + Poisson jumps with closed-form option pricing
- **CorrelatedGBM** - Multi-asset simulation with Cholesky-decomposed correlations

### Option Pricing
- **BlackScholes** - Closed-form European options with all Greeks (vectorized)
- **MonteCarloPricer** - Flexible MC engine with variance reduction
  - European, Asian, lookback, and custom payoff functions
- **BinomialTree** - CRR binomial tree for American/European options
- **BarrierOption** - All 8 barrier types (up/down, in/out, call/put)
  - Reiner-Rubinstein closed-form + Monte Carlo pricing
  - In/out parity cross-validation
- **DigitalOption** - Cash-or-nothing and asset-or-nothing options
  - Full Greeks (delta, gamma, vega), vectorized inputs

### Risk Metrics
- **VaR** - Historical, Parametric, Monte Carlo, Cornish-Fisher
- **CVaR** - Expected Shortfall
- **Sharpe Ratio, Sortino Ratio, Calmar Ratio**
- **Max Drawdown, Beta, Alpha, Information Ratio**

### Volatility
- **Implied Volatility Solver** - Newton-Raphson, Bisection, Brent's method
- **GARCH(1,1)** - Volatility forecasting with MLE fitting
- **SABR** - Hagan et al. (2002) smile model with calibration
- **HestonCalibrator** - Calibrate Heston parameters to IV surfaces

### Data Utilities
- **returns_from_prices** - Log or simple returns from price series
- **correlation_from_returns** - Correlation matrix estimation
- **estimate_gbm_params** - Estimate drift and volatility from prices
- **estimate_ou_params** - Estimate mean-reversion parameters via OLS
- **fetch_prices** - Historical prices via yfinance (optional dependency)
- **fetch_options_chain** - Options data for calibration workflows

## Design Philosophy

1. **Simple API**: One-liners for common tasks
2. **NumPy Native**: Vectorized operations, array inputs
3. **Stateless**: Thread-safe, no global state
4. **Type-Hinted**: Full IDE/mypy support
5. **Minimal Dependencies**: Only numpy + scipy required

## API Reference

### Processes

```python
from stochastic_engine import GBM, OrnsteinUhlenbeck, Heston, Vasicek, CIR
from stochastic_engine import MertonJumpDiffusion, CorrelatedGBM

# Geometric Brownian Motion
gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
paths = gbm.simulate(T=1, steps=252, n_paths=1000, method="exact")

# Ornstein-Uhlenbeck
ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02)
print(ou.half_life)  # Time to revert halfway to mean

# Heston stochastic volatility
heston = Heston(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04, xi=0.3, rho=-0.7)
price = heston.price_european_call(K=100, T=1, r=0.05)

# Interest rate models
vasicek = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
print(vasicek.bond_price(T=5))
print(vasicek.yield_curve(np.array([1, 2, 5, 10, 30])))

cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
print(cir.feller_satisfied)  # True if 2*kappa*theta > sigma^2

# Jump-diffusion
mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=3, mu_j=-0.02, sigma_j=0.1)
call_price = mjd.price_european_call(K=100, T=1, r=0.05)  # Merton series formula

# Multi-asset with correlations
portfolio = CorrelatedGBM(
    S0=np.array([100, 50]), mu=np.array([0.05, 0.08]),
    sigma=np.array([0.2, 0.3]),
    correlation=np.array([[1, 0.6], [0.6, 1]])
)
paths = portfolio.simulate(T=1, steps=252, n_paths=5000)  # shape: (2, 5000, 253)
```

### Pricing

```python
from stochastic_engine import BlackScholes, MonteCarloPricer, BinomialTree
from stochastic_engine import BarrierOption, DigitalOption

# Black-Scholes (vectorized)
strikes = np.array([95, 100, 105, 110])
bs = BlackScholes(S=100, K=strikes, T=1, r=0.05, sigma=0.2)
print(bs.price)   # Array of 4 prices
print(bs.delta)   # Array of 4 deltas

# Monte Carlo with custom payoff
mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1)
result = mc.price_european_call(K=100, n_paths=100000)
print(f"${result.price:.2f} ± ${result.std_error:.4f}")

# American options via binomial tree
tree = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=500, exercise="american")

# Barrier options - all 8 types
barrier = BarrierOption(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                        barrier_type="down-and-out")
cf_result = barrier.price(method="closed_form")
mc_result = barrier.price(method="monte_carlo", n_paths=100000)

# Digital options with Greeks
digital = DigitalOption(S=100, K=105, T=1, r=0.05, sigma=0.2,
                        digital_type="asset-or-nothing")
print(digital.price, digital.delta, digital.gamma, digital.vega)
```

### Volatility

```python
from stochastic_engine import implied_volatility, GARCH, SABR, HestonCalibrator

# Implied volatility
iv = implied_volatility(market_price=8.02, S=100, K=105, T=1, r=0.05)

# GARCH forecasting
garch = GARCH()
garch.fit(returns)
forecast = garch.forecast(horizon=10)

# SABR smile calibration
sabr = SABR(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
vol = sabr.implied_vol(f=100, K=105, T=1)
strikes, vols = sabr.smile(f=100, T=1)
result = SABR.fit(f=100, strikes=strikes, T=1, market_vols=market_vols, beta=0.5)

# Heston calibration
calibrator = HestonCalibrator(S=100, r=0.05)
result = calibrator.calibrate(strikes, maturities, market_ivs)
print(result.rmse, result.feller_satisfied)
heston = result.to_heston(S0=100, mu=0.05)
```

### Risk

```python
from stochastic_engine import VaR, CVaR

var = VaR(returns, confidence=0.95)
var.historical()          # Historical simulation
var.parametric()          # Variance-covariance (normal)
var.parametric("t")       # Student's t-distribution
var.monte_carlo()         # Monte Carlo
var.cornish_fisher()      # Adjusted for skewness/kurtosis

# Backtest VaR
results = var.backtest(window=250)
print(f"Violation rate: {results['violation_rate']:.2%}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

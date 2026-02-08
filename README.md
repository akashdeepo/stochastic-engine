# Stochastic Engine

A comprehensive, Pythonic quantitative finance library for stochastic processes, option pricing, and risk metrics.

**No more Googling formulas. No more reimplementing Black-Scholes. Just clean, intuitive APIs.**

## Installation

```bash
pip install stochastic-engine
```

Or install from source:

```bash
git clone https://github.com/akashdeepo/stochastic-engine.git
cd stochastic-engine
pip install -e ".[dev]"
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

### Monte Carlo Pricing

```python
from stochastic_engine import MonteCarloPricer

mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1)

# European call
result = mc.price_european_call(K=105, n_paths=100000)
print(f"Price: ${result.price:.2f} ± {result.std_error:.4f}")

# Asian call (average price option)
asian = mc.price_asian_call(K=100, n_paths=50000, steps=252)
print(f"Asian call: ${asian.price:.2f}")
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

### Implied Volatility

```python
from stochastic_engine import implied_volatility

# Find IV from market price
iv = implied_volatility(
    market_price=8.02,
    S=100, K=105, T=1, r=0.05
)
print(f"Implied Vol: {iv:.2%}")  # ~20%
```

## Features

### Stochastic Processes
- **GBM** - Geometric Brownian Motion (stock prices)
- **OrnsteinUhlenbeck** - Mean-reverting process (interest rates, pairs trading)
- **Heston** - Stochastic volatility model (volatility smile)

### Option Pricing
- **BlackScholes** - Closed-form European options with all Greeks
- **MonteCarloPricer** - Flexible MC engine with variance reduction
  - European calls/puts
  - Asian options (arithmetic & geometric)
  - Custom payoff functions
- **BinomialTree** - CRR binomial tree for American/European options

### Risk Metrics
- **VaR** - Historical, Parametric, Monte Carlo, Cornish-Fisher
- **CVaR** - Expected Shortfall
- **Sharpe Ratio, Sortino Ratio, Calmar Ratio**
- **Max Drawdown**
- **Beta, Alpha, Information Ratio**

### Volatility
- **Implied Volatility Solver** - Newton-Raphson, Bisection, Brent's method
- **GARCH(1,1)** - Volatility forecasting with MLE fitting

## Design Philosophy

1. **Simple API**: One-liners for common tasks
2. **NumPy Native**: Vectorized operations, array inputs
3. **Stateless**: Thread-safe, no global state
4. **Type-Hinted**: Full IDE/mypy support
5. **Well-Documented**: NumPy-style docstrings, tutorials, examples

## API Reference

### Processes

```python
# Geometric Brownian Motion
gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
paths = gbm.simulate(T=1, steps=252, n_paths=1000, method="exact")
samples = gbm.sample(t=1, n_samples=10000)
print(gbm.mean(1), gbm.variance(1))

# Ornstein-Uhlenbeck
ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02)
paths = ou.simulate(T=10, steps=2520, n_paths=100)
print(ou.half_life)  # Time to revert halfway to mean
```

### Pricing

```python
# Black-Scholes (vectorized)
strikes = np.array([95, 100, 105, 110])
bs = BlackScholes(S=100, K=strikes, T=1, r=0.05, sigma=0.2)
print(bs.price)   # Array of 4 prices
print(bs.delta)   # Array of 4 deltas

# Monte Carlo with custom payoff
def barrier_call(paths, K=100, barrier=120):
    """Knock-out call: worthless if price ever exceeds barrier."""
    knocked_out = (paths.max(axis=1) >= barrier)
    payoff = np.maximum(paths[:, -1] - K, 0)
    payoff[knocked_out] = 0
    return payoff

mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1)
result = mc.price_custom(barrier_call, n_paths=100000, steps=252)
```

### Risk

```python
# VaR methods
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

## Examples

### Monte Carlo Option Pricing vs Black-Scholes

```python
from stochastic_engine import BlackScholes, MonteCarloPricer

# Analytical price
bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"BS Price: ${bs.price:.4f}")

# Monte Carlo price
mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
result = mc.price_european_call(K=100, n_paths=100000)
print(f"MC Price: ${result.price:.4f} ± ${result.std_error:.4f}")
```

### Portfolio VaR

```python
import numpy as np
from stochastic_engine import VaR, CVaR

# Simulate portfolio returns
np.random.seed(42)
returns = np.random.normal(0.0005, 0.02, 252)  # 1 year of daily returns

var = VaR(returns, confidence=0.99)
cvar = CVaR(returns, confidence=0.99)

print(f"99% 1-day VaR:  {var.historical():.2%}")
print(f"99% 1-day CVaR: {cvar.historical():.2%}")
print(f"99% 10-day VaR: {var.scale_to_horizon(var.historical(), 10):.2%}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

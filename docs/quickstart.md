# Quickstart

Get up and running with Stochastic Engine in 5 minutes.

## Installation

```bash
pip install stochastic-engine
```

For development:

```bash
pip install stochastic-engine[dev]
```

## Your First Simulation

### Simulate Stock Prices with GBM

```python
from stochastic_engine import GBM

# Create a GBM process
# S0 = initial price, mu = drift, sigma = volatility
gbm = GBM(S0=100, mu=0.05, sigma=0.2)

# Simulate 1000 paths over 1 year (252 trading days)
paths = gbm.simulate(T=1, steps=252, n_paths=1000)

print(paths.shape)  # (1000, 253) - 1000 paths, 253 time points
print(paths[:, -1].mean())  # Average terminal price
```

### Price a European Option

```python
from stochastic_engine import BlackScholes

# Create an option
option = BlackScholes(
    S=100,      # Current stock price
    K=105,      # Strike price
    T=1,        # Time to expiration (years)
    r=0.05,     # Risk-free rate
    sigma=0.2,  # Volatility
    option_type='call'
)

# Get price and Greeks
print(f"Price: ${option.price:.2f}")
print(f"Delta: {option.delta:.4f}")
print(f"Gamma: {option.gamma:.4f}")
print(f"Vega: {option.vega:.4f}")
print(f"Theta: {option.theta:.4f}")
```

### Calculate Risk Metrics

```python
import numpy as np
from stochastic_engine import VaR, CVaR
from stochastic_engine.risk import sharpe_ratio

# Sample returns
returns = np.random.normal(0.0005, 0.02, 252)

# Value at Risk
var = VaR(returns, confidence=0.95)
print(f"95% VaR: {var.historical():.2%}")

# Expected Shortfall
cvar = CVaR(returns, confidence=0.95)
print(f"95% CVaR: {cvar.historical():.2%}")

# Sharpe Ratio
sr = sharpe_ratio(returns, risk_free_rate=0)
print(f"Sharpe Ratio: {sr:.2f}")
```

### Find Implied Volatility

```python
from stochastic_engine import implied_volatility

# Market price of a call option
market_price = 8.02

# Find the implied volatility
iv = implied_volatility(
    market_price=market_price,
    S=100, K=105, T=1, r=0.05
)

print(f"Implied Volatility: {iv:.2%}")  # ~20%
```

## Next Steps

- Explore the [API Reference](api/processes.md) for detailed documentation
- Check out the [Jupyter notebooks](https://github.com/yourusername/stochastic-engine/tree/main/notebooks) for more examples

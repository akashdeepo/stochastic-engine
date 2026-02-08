# Stochastic Engine

**A Pythonic Quantitative Finance Library**

No more Googling formulas. No more reimplementing Black-Scholes. Just clean, intuitive APIs.

## Why Stochastic Engine?

Every quant has been there:

- Need to price an option? Google "Black-Scholes formula Python"
- Need Monte Carlo simulation? Copy-paste from an old notebook
- Need VaR calculation? Write it from scratch (again)

**Stochastic Engine** packages everything you need into a single, well-documented library.

## Quick Example

```python
from stochastic_engine import GBM, BlackScholes, VaR

# Simulate stock prices
paths = GBM(S0=100, mu=0.05, sigma=0.2).simulate(T=1, steps=252, n_paths=1000)

# Price options with all Greeks
option = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)
print(option.price, option.delta, option.gamma)

# Calculate risk metrics
var = VaR(returns, confidence=0.95).historical()
```

## Features

| Module | What's Included |
|--------|-----------------|
| **Processes** | GBM, Ornstein-Uhlenbeck |
| **Pricing** | Black-Scholes (with Greeks), Monte Carlo |
| **Risk** | VaR, CVaR, Sharpe, Sortino, Max Drawdown |
| **Volatility** | Implied Vol Solver, IV Surface |

## Design Philosophy

1. **Simple API**: One-liners for common tasks
2. **NumPy Native**: Vectorized operations
3. **Type-Hinted**: Full IDE support
4. **Well-Documented**: Every function has examples

## Installation

```bash
pip install stochastic-engine
```

## Next Steps

- [Quickstart Guide](quickstart.md) - Get up and running in 5 minutes
- [API Reference](api/processes.md) - Detailed documentation

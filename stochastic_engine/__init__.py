"""
Stochastic Engine - A Pythonic Quantitative Finance Library.

A comprehensive library for stochastic processes, option pricing,
and risk metrics with clean, intuitive APIs.

Example
-------
>>> from stochastic_engine import GBM, BlackScholes, VaR
>>>
>>> # Simulate stock price paths
>>> paths = GBM(S0=100, mu=0.05, sigma=0.2).simulate(T=1, steps=252, n_paths=1000)
>>>
>>> # Price an option with Greeks
>>> option = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)
>>> print(option.price, option.delta)
"""

from stochastic_engine.processes.gbm import GBM
from stochastic_engine.processes.ornstein_uhlenbeck import OrnsteinUhlenbeck
from stochastic_engine.pricing.black_scholes import BlackScholes
from stochastic_engine.pricing.monte_carlo import MonteCarloPricer
from stochastic_engine.volatility.implied import implied_volatility
from stochastic_engine.risk.var import VaR
from stochastic_engine.risk.cvar import CVaR
from stochastic_engine.risk.metrics import sharpe_ratio, sortino_ratio, max_drawdown

__version__ = "0.1.0"

__all__ = [
    # Processes
    "GBM",
    "OrnsteinUhlenbeck",
    # Pricing
    "BlackScholes",
    "MonteCarloPricer",
    # Volatility
    "implied_volatility",
    # Risk
    "VaR",
    "CVaR",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
]

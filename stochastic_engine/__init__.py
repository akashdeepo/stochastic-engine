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

# Processes
from stochastic_engine.processes.gbm import GBM
from stochastic_engine.processes.heston import Heston
from stochastic_engine.processes.interest_rate import CIR, Vasicek
from stochastic_engine.processes.merton_jump_diffusion import MertonJumpDiffusion
from stochastic_engine.processes.multi_asset import CorrelatedGBM
from stochastic_engine.processes.ornstein_uhlenbeck import OrnsteinUhlenbeck

# Pricing
from stochastic_engine.pricing.barrier import BarrierOption, barrier_call, barrier_put
from stochastic_engine.pricing.binomial import BinomialTree, american_call, american_put
from stochastic_engine.pricing.black_scholes import BlackScholes
from stochastic_engine.pricing.digital import (
    DigitalOption,
    asset_or_nothing_call,
    asset_or_nothing_put,
    cash_or_nothing_call,
    cash_or_nothing_put,
)
from stochastic_engine.pricing.monte_carlo import MonteCarloPricer

# Volatility
from stochastic_engine.volatility.garch import GARCH
from stochastic_engine.volatility.heston_calibration import HestonCalibrator
from stochastic_engine.volatility.implied import implied_volatility
from stochastic_engine.volatility.sabr import SABR

# Risk
from stochastic_engine.risk.cvar import CVaR
from stochastic_engine.risk.metrics import max_drawdown, sharpe_ratio, sortino_ratio
from stochastic_engine.risk.var import VaR

# Data
from stochastic_engine.data.market import (
    correlation_from_returns,
    estimate_gbm_params,
    estimate_ou_params,
    returns_from_prices,
)

__version__ = "0.3.0"

__all__ = [
    # Processes
    "GBM",
    "OrnsteinUhlenbeck",
    "Heston",
    "CorrelatedGBM",
    "MertonJumpDiffusion",
    "Vasicek",
    "CIR",
    # Pricing
    "BlackScholes",
    "MonteCarloPricer",
    "BinomialTree",
    "american_put",
    "american_call",
    "BarrierOption",
    "barrier_call",
    "barrier_put",
    "DigitalOption",
    "cash_or_nothing_call",
    "cash_or_nothing_put",
    "asset_or_nothing_call",
    "asset_or_nothing_put",
    # Volatility
    "implied_volatility",
    "GARCH",
    "SABR",
    "HestonCalibrator",
    # Risk
    "VaR",
    "CVaR",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    # Data
    "returns_from_prices",
    "correlation_from_returns",
    "estimate_gbm_params",
    "estimate_ou_params",
]

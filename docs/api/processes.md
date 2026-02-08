# Stochastic Processes

This module provides implementations of common stochastic processes used in quantitative finance.

## Geometric Brownian Motion (GBM)

The standard model for stock prices. Assumes constant drift and volatility.

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

::: stochastic_engine.processes.gbm.GBM

## Ornstein-Uhlenbeck Process

A mean-reverting process used for interest rates, volatility, and pairs trading.

$$
dX_t = \theta (\mu - X_t) \, dt + \sigma \, dW_t
$$

::: stochastic_engine.processes.ornstein_uhlenbeck.OrnsteinUhlenbeck

## Base Class

All processes inherit from this abstract base class.

::: stochastic_engine.processes.base.StochasticProcess

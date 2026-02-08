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

## Heston Stochastic Volatility

A two-factor model where both price and volatility are stochastic. Captures the volatility smile observed in option markets.

$$
dS_t = \mu S_t \, dt + \sqrt{v_t} S_t \, dW^1_t
$$

$$
dv_t = \kappa (\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW^2_t
$$

where $dW^1_t dW^2_t = \rho \, dt$.

::: stochastic_engine.processes.heston.Heston

## Base Class

All processes inherit from this abstract base class.

::: stochastic_engine.processes.base.StochasticProcess

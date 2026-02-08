# Volatility

This module provides tools for volatility calculations including implied volatility solvers.

## Implied Volatility

Find the volatility that makes the Black-Scholes price match the market price.

::: stochastic_engine.volatility.implied.implied_volatility

## ImpliedVolSolver Class

For more control over the solving process.

::: stochastic_engine.volatility.implied.ImpliedVolSolver

## GARCH(1,1) Model

Generalized Autoregressive Conditional Heteroskedasticity model for volatility forecasting.

$$
\sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}
$$

::: stochastic_engine.volatility.garch.GARCH

::: stochastic_engine.volatility.garch.GARCHResult

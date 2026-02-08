# Option Pricing

This module provides option pricing models including closed-form solutions and Monte Carlo methods.

## Black-Scholes Model

Closed-form European option pricing with all Greeks.

$$
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

where:

$$
d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

::: stochastic_engine.pricing.black_scholes.BlackScholes

### Convenience Functions

::: stochastic_engine.pricing.black_scholes.bs_call

::: stochastic_engine.pricing.black_scholes.bs_put

## Monte Carlo Pricer

Flexible Monte Carlo engine for pricing options with simulation.

::: stochastic_engine.pricing.monte_carlo.MonteCarloPricer

::: stochastic_engine.pricing.monte_carlo.MCResult

## Binomial Tree

Cox-Ross-Rubinstein (CRR) binomial tree for American and European option pricing.

::: stochastic_engine.pricing.binomial.BinomialTree

### Convenience Functions

::: stochastic_engine.pricing.binomial.american_put

::: stochastic_engine.pricing.binomial.american_call

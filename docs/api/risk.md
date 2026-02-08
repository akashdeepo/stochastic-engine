# Risk Metrics

This module provides risk measurement and performance analytics tools.

## Value at Risk (VaR)

Estimates the maximum potential loss at a given confidence level.

::: stochastic_engine.risk.var.VaR

## Conditional Value at Risk (CVaR)

Also known as Expected Shortfall. Measures expected loss beyond VaR.

$$
\text{CVaR}_\alpha = E[L \mid L > \text{VaR}_\alpha]
$$

::: stochastic_engine.risk.cvar.CVaR

## Performance Metrics

### Sharpe Ratio

Risk-adjusted return measure.

$$
\text{SR} = \frac{E[R_p - R_f]}{\sigma_p} \times \sqrt{N}
$$

::: stochastic_engine.risk.metrics.sharpe_ratio

### Sortino Ratio

Like Sharpe, but only penalizes downside volatility.

::: stochastic_engine.risk.metrics.sortino_ratio

### Maximum Drawdown

Largest peak-to-trough decline.

::: stochastic_engine.risk.metrics.max_drawdown

### Other Metrics

::: stochastic_engine.risk.metrics.calmar_ratio

::: stochastic_engine.risk.metrics.information_ratio

::: stochastic_engine.risk.metrics.beta

::: stochastic_engine.risk.metrics.alpha

::: stochastic_engine.risk.metrics.volatility

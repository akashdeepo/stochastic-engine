"""Risk-adjusted performance metrics."""

import numpy as np


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Sharpe Ratio.

    Measures risk-adjusted return as excess return per unit of total risk.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns (as decimals).
    risk_free_rate : float, optional
        Risk-free rate for the same period as returns. Default is 0.
    periods_per_year : int, optional
        Number of periods per year for annualization.
        252 for daily, 52 for weekly, 12 for monthly. Default is 252.

    Returns
    -------
    float
        Annualized Sharpe Ratio.

    Examples
    --------
    >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
    >>> sharpe_ratio(returns, risk_free_rate=0.0001)
    1.89...

    Notes
    -----
    The Sharpe Ratio is calculated as:

    .. math::
        SR = \\frac{E[R_p - R_f]}{\\sigma_p} \\times \\sqrt{N}

    where N is periods_per_year.

    Interpretation:
    - SR > 1: Good
    - SR > 2: Very good
    - SR > 3: Excellent
    """
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate

    mean_excess = excess_returns.mean()
    std = excess_returns.std()

    if std == 0:
        return np.inf if mean_excess > 0 else (-np.inf if mean_excess < 0 else 0.0)

    return float(mean_excess / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Sortino Ratio.

    Measures risk-adjusted return using only downside deviation,
    penalizing only harmful volatility.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns (as decimals).
    risk_free_rate : float, optional
        Risk-free rate for the same period as returns. Default is 0.
    target_return : float, optional
        Minimum acceptable return (MAR). Default is 0.
    periods_per_year : int, optional
        Number of periods per year for annualization. Default is 252.

    Returns
    -------
    float
        Annualized Sortino Ratio.

    Examples
    --------
    >>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
    >>> sortino_ratio(returns)
    2.76...

    Notes
    -----
    The Sortino Ratio uses downside deviation instead of standard deviation:

    .. math::
        Sortino = \\frac{E[R_p - R_f]}{\\sigma_d} \\times \\sqrt{N}

    where σ_d is the downside deviation (std of returns below target).
    """
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate

    mean_excess = excess_returns.mean()

    # Downside deviation: std of returns below target
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0

    downside_std = np.sqrt(np.mean((downside_returns - target_return) ** 2))

    if downside_std == 0:
        return np.inf if mean_excess > 0 else 0.0

    return float(mean_excess / downside_std * np.sqrt(periods_per_year))


def max_drawdown(prices: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown.

    The largest peak-to-trough decline in portfolio value.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices or portfolio values over time.

    Returns
    -------
    float
        Maximum drawdown as a positive decimal (e.g., 0.2 for 20%).

    Examples
    --------
    >>> prices = np.array([100, 105, 95, 90, 100, 85, 95])
    >>> max_drawdown(prices)
    0.190...  # From 105 to 85 = 19.05%

    Notes
    -----
    Maximum drawdown is calculated as:

    .. math::
        MDD = \\max_t \\left( \\frac{\\text{Peak}_t - \\text{Price}_t}{\\text{Peak}_t} \\right)
    """
    prices = np.asarray(prices)

    # Running maximum
    running_max = np.maximum.accumulate(prices)

    # Drawdown at each point
    drawdowns = (running_max - prices) / running_max

    return float(np.max(drawdowns))


def max_drawdown_duration(prices: np.ndarray) -> int:
    """
    Calculate Maximum Drawdown Duration.

    The longest time between a peak and subsequent recovery.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices or portfolio values over time.

    Returns
    -------
    int
        Maximum drawdown duration in periods.

    Examples
    --------
    >>> prices = np.array([100, 105, 95, 90, 100, 85, 95, 100, 105])
    >>> max_drawdown_duration(prices)
    5  # From index 1 to 8 (peak to recovery)
    """
    prices = np.asarray(prices)
    running_max = np.maximum.accumulate(prices)

    max_duration = 0
    current_duration = 0
    in_drawdown = False

    for i in range(len(prices)):
        if prices[i] < running_max[i]:
            if not in_drawdown:
                in_drawdown = True
                current_duration = 1
            else:
                current_duration += 1
        else:
            if in_drawdown:
                max_duration = max(max_duration, current_duration)
                in_drawdown = False
                current_duration = 0

    # Check if still in drawdown at end
    if in_drawdown:
        max_duration = max(max_duration, current_duration)

    return max_duration


def calmar_ratio(
    returns: np.ndarray,
    prices: np.ndarray | None = None,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Calmar Ratio.

    Ratio of annualized return to maximum drawdown.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    prices : np.ndarray | None, optional
        Array of prices for drawdown calculation.
        If None, prices are computed from returns assuming initial value of 1.
    periods_per_year : int, optional
        Number of periods per year. Default is 252.

    Returns
    -------
    float
        Calmar Ratio.

    Examples
    --------
    >>> returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02, -0.005])
    >>> calmar_ratio(returns)
    2.18...

    Notes
    -----
    .. math::
        Calmar = \\frac{\\text{Annualized Return}}{\\text{Maximum Drawdown}}
    """
    returns = np.asarray(returns)

    if prices is None:
        # Compute prices from returns
        prices = np.cumprod(1 + returns)

    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    mdd = max_drawdown(prices)

    if mdd == 0:
        return np.inf if ann_return > 0 else 0.0

    return float(ann_return / mdd)


def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate the Information Ratio.

    Measures active return (alpha) per unit of tracking error.

    Parameters
    ----------
    returns : np.ndarray
        Array of portfolio returns.
    benchmark_returns : np.ndarray
        Array of benchmark returns (same length as returns).
    periods_per_year : int, optional
        Number of periods per year. Default is 252.

    Returns
    -------
    float
        Annualized Information Ratio.

    Examples
    --------
    >>> portfolio = np.array([0.01, 0.02, -0.01, 0.015, 0.008])
    >>> benchmark = np.array([0.008, 0.015, -0.005, 0.01, 0.012])
    >>> information_ratio(portfolio, benchmark)
    0.61...

    Notes
    -----
    .. math::
        IR = \\frac{E[R_p - R_b]}{\\sigma(R_p - R_b)} \\times \\sqrt{N}

    where σ(R_p - R_b) is the tracking error.
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()

    if tracking_error == 0:
        mean_active = active_returns.mean()
        return np.inf if mean_active > 0 else (-np.inf if mean_active < 0 else 0.0)

    return float(active_returns.mean() / tracking_error * np.sqrt(periods_per_year))


def beta(
    returns: np.ndarray,
    market_returns: np.ndarray,
) -> float:
    """
    Calculate Beta (systematic risk).

    Measures sensitivity of returns to market movements.

    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns.
    market_returns : np.ndarray
        Array of market returns.

    Returns
    -------
    float
        Beta coefficient.

    Examples
    --------
    >>> asset = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    >>> market = np.array([0.01, -0.005, 0.02, -0.01, 0.008])
    >>> beta(asset, market)
    1.89...

    Notes
    -----
    .. math::
        \\beta = \\frac{Cov(R_a, R_m)}{Var(R_m)}

    Interpretation:
    - β = 1: Moves with market
    - β > 1: More volatile than market
    - β < 1: Less volatile than market
    - β < 0: Inversely correlated with market
    """
    returns = np.asarray(returns)
    market_returns = np.asarray(market_returns)

    covariance = np.cov(returns, market_returns)[0, 1]
    market_variance = market_returns.var()

    if market_variance == 0:
        return 0.0

    return float(covariance / market_variance)


def alpha(
    returns: np.ndarray,
    market_returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Jensen's Alpha.

    Measures excess return above CAPM expectation.

    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns.
    market_returns : np.ndarray
        Array of market returns.
    risk_free_rate : float, optional
        Risk-free rate per period. Default is 0.
    periods_per_year : int, optional
        Number of periods per year. Default is 252.

    Returns
    -------
    float
        Annualized alpha.

    Examples
    --------
    >>> asset = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    >>> market = np.array([0.01, -0.005, 0.02, -0.01, 0.008])
    >>> alpha(asset, market)
    0.42...

    Notes
    -----
    .. math::
        \\alpha = E[R_a] - (R_f + \\beta (E[R_m] - R_f))
    """
    returns = np.asarray(returns)
    market_returns = np.asarray(market_returns)

    b = beta(returns, market_returns)

    expected_return = returns.mean()
    expected_market = market_returns.mean()

    # CAPM expected return
    capm_return = risk_free_rate + b * (expected_market - risk_free_rate)

    # Alpha (annualized)
    periodic_alpha = expected_return - capm_return
    return float(periodic_alpha * periods_per_year)


def volatility(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized volatility.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    periods_per_year : int, optional
        Number of periods per year. Default is 252.

    Returns
    -------
    float
        Annualized volatility.

    Examples
    --------
    >>> returns = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
    >>> volatility(returns)
    0.25...
    """
    returns = np.asarray(returns)
    return float(returns.std() * np.sqrt(periods_per_year))

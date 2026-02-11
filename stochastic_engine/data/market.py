"""Market data utilities for fetching prices and estimating model parameters."""

from typing import Literal

import numpy as np


def _get_yfinance():
    """Lazy import yfinance."""
    try:
        import yfinance as yf

        return yf
    except ImportError:
        raise ImportError(
            "yfinance is required for fetching market data. "
            "Install with: pip install stochastic-engine[data]"
        )


def fetch_prices(
    tickers: str | list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "1y",
    interval: str = "1d",
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Fetch historical close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : str or list[str]
        Ticker symbol(s) to fetch.
    start : str or None, optional
        Start date (e.g., "2023-01-01"). If None, uses period.
    end : str or None, optional
        End date. If None, uses today.
    period : str, optional
        Period to fetch (e.g., "1y", "6mo", "5d"). Default is "1y".
        Ignored if start is provided.
    interval : str, optional
        Data interval ("1d", "1wk", "1mo"). Default is "1d".

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        Close prices as numpy array (single ticker) or dict of arrays.
        Requires yfinance: ``pip install stochastic-engine[data]``

    Examples
    --------
    >>> prices = fetch_prices("AAPL", period="6mo")
    >>> prices.shape
    (126,)

    >>> multi = fetch_prices(["AAPL", "MSFT"], period="1y")
    >>> multi["AAPL"].shape
    (252,)
    """
    yf = _get_yfinance()

    if isinstance(tickers, str):
        ticker_obj = yf.Ticker(tickers)
        kwargs = {"interval": interval}
        if start is not None:
            kwargs["start"] = start
            if end is not None:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        hist = ticker_obj.history(**kwargs)
        return hist["Close"].to_numpy()

    # Multiple tickers
    result = {}
    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)
        kwargs = {"interval": interval}
        if start is not None:
            kwargs["start"] = start
            if end is not None:
                kwargs["end"] = end
        else:
            kwargs["period"] = period

        hist = ticker_obj.history(**kwargs)
        result[ticker] = hist["Close"].to_numpy()

    return result


def fetch_options_chain(
    ticker: str,
    expiration: str | None = None,
) -> dict:
    """
    Fetch options chain data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    expiration : str or None, optional
        Expiration date string. If None, uses the nearest expiration.

    Returns
    -------
    dict
        Dictionary with keys:
        - "calls": dict with "strikes", "last_prices", "implied_vols", "volume"
        - "puts": dict with "strikes", "last_prices", "implied_vols", "volume"
        - "expiration": the expiration date used
        - "spot": current spot price

        Requires yfinance: ``pip install stochastic-engine[data]``

    Examples
    --------
    >>> chain = fetch_options_chain("AAPL")
    >>> chain["calls"]["strikes"][:5]
    array([100., 105., 110., ...])
    """
    yf = _get_yfinance()

    ticker_obj = yf.Ticker(ticker)
    spot = ticker_obj.history(period="1d")["Close"].iloc[-1]

    if expiration is None:
        expirations = ticker_obj.options
        if len(expirations) == 0:
            raise ValueError(f"No options available for {ticker}")
        expiration = expirations[0]

    chain = ticker_obj.option_chain(expiration)

    def _extract(df):
        return {
            "strikes": df["strike"].to_numpy(),
            "last_prices": df["lastPrice"].to_numpy(),
            "implied_vols": df["impliedVolatility"].to_numpy(),
            "volume": df["volume"].to_numpy(),
        }

    return {
        "calls": _extract(chain.calls),
        "puts": _extract(chain.puts),
        "expiration": expiration,
        "spot": float(spot),
    }


def returns_from_prices(
    prices: np.ndarray,
    method: Literal["log", "simple"] = "log",
) -> np.ndarray:
    """
    Convert price series to returns.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices over time.
    method : {"log", "simple"}, optional
        Return calculation method. Default is "log".
        - "log": ln(P_t / P_{t-1})
        - "simple": (P_t - P_{t-1}) / P_{t-1}

    Returns
    -------
    np.ndarray
        Array of returns (length = len(prices) - 1).

    Examples
    --------
    >>> prices = np.array([100, 105, 103, 108])
    >>> returns_from_prices(prices, method="log")
    array([ 0.04879..., -0.01923...,  0.04738...])

    >>> returns_from_prices(prices, method="simple")
    array([ 0.05, -0.01904...,  0.04854...])
    """
    prices = np.asarray(prices, dtype=float)

    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to compute returns")

    if method == "log":
        return np.log(prices[1:] / prices[:-1])
    elif method == "simple":
        return (prices[1:] - prices[:-1]) / prices[:-1]
    else:
        raise ValueError(f"method must be 'log' or 'simple', got {method}")


def correlation_from_returns(
    returns: dict[str, np.ndarray] | np.ndarray,
) -> np.ndarray:
    """
    Compute correlation matrix from returns.

    Parameters
    ----------
    returns : dict[str, np.ndarray] or np.ndarray
        Either a dict mapping asset names to return arrays,
        or a 2D array of shape (n_assets, n_observations).

    Returns
    -------
    np.ndarray
        Correlation matrix of shape (N, N).

    Examples
    --------
    >>> ret_a = np.array([0.01, -0.02, 0.015, -0.01])
    >>> ret_b = np.array([0.008, -0.015, 0.012, -0.008])
    >>> corr = correlation_from_returns({"A": ret_a, "B": ret_b})
    >>> corr.shape
    (2, 2)
    """
    if isinstance(returns, dict):
        arrays = list(returns.values())
        # Ensure equal lengths by trimming to shortest
        min_len = min(len(a) for a in arrays)
        data = np.array([a[:min_len] for a in arrays])
    else:
        data = np.asarray(returns)
        if data.ndim == 1:
            return np.array([[1.0]])

    return np.corrcoef(data)


def estimate_gbm_params(
    prices: np.ndarray,
    dt: float = 1 / 252,
) -> dict:
    """
    Estimate GBM parameters from historical prices.

    Uses log-return statistics to estimate drift and volatility.

    Parameters
    ----------
    prices : np.ndarray
        Array of historical prices.
    dt : float, optional
        Time step between observations (in years).
        Default is 1/252 (daily data).

    Returns
    -------
    dict
        Dictionary with keys "S0", "mu", "sigma".

    Examples
    --------
    >>> from stochastic_engine import GBM
    >>> gbm = GBM(S0=100, mu=0.08, sigma=0.2, seed=42)
    >>> prices = gbm.simulate(T=1, steps=252, n_paths=1)[:, 0]
    >>> params = estimate_gbm_params(prices)  # Should recover mu~0.08, sigma~0.2
    """
    prices = np.asarray(prices, dtype=float)
    log_returns = np.log(prices[1:] / prices[:-1])

    sigma = float(log_returns.std() / np.sqrt(dt))
    mu = float(log_returns.mean() / dt + 0.5 * sigma**2)

    return {
        "S0": float(prices[0]),
        "mu": mu,
        "sigma": sigma,
    }


def estimate_ou_params(
    data: np.ndarray,
    dt: float = 1 / 252,
) -> dict:
    """
    Estimate Ornstein-Uhlenbeck parameters from time series.

    Uses OLS regression: X_{t+1} = a + b * X_t + epsilon.

    Parameters
    ----------
    data : np.ndarray
        Array of time series observations.
    dt : float, optional
        Time step between observations (in years).
        Default is 1/252 (daily data).

    Returns
    -------
    dict
        Dictionary with keys "X0", "mu", "theta", "sigma".

    Examples
    --------
    >>> from stochastic_engine import OrnsteinUhlenbeck
    >>> ou = OrnsteinUhlenbeck(X0=0.05, mu=0.03, theta=5.0, sigma=0.01, seed=42)
    >>> path = ou.simulate(T=5, steps=1260, n_paths=1)[0]
    >>> params = estimate_ou_params(path)  # Should recover theta~5, mu~0.03
    """
    data = np.asarray(data, dtype=float)

    if len(data) < 3:
        raise ValueError("Need at least 3 data points for OU estimation")

    # OLS: X_{t+1} = a + b * X_t
    X = data[:-1]
    Y = data[1:]

    n = len(X)
    sum_x = X.sum()
    sum_y = Y.sum()
    sum_xy = (X * Y).sum()
    sum_x2 = (X**2).sum()

    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    a = (sum_y - b * sum_x) / n

    # Convert to continuous-time parameters
    theta = -np.log(max(b, 1e-10)) / dt
    mu = a / (1 - b) if abs(1 - b) > 1e-10 else float(data.mean())

    # Estimate sigma from residuals
    residuals = Y - (a + b * X)
    sigma_discrete = residuals.std()
    sigma = sigma_discrete * np.sqrt(2 * theta / (1 - np.exp(-2 * theta * dt)))

    return {
        "X0": float(data[0]),
        "mu": float(mu),
        "theta": float(theta),
        "sigma": float(sigma),
    }

"""Risk visualization tools."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def _get_pyplot():
    """Lazy import matplotlib."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install stochastic-engine[notebooks]"
        )


def plot_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    bins: int = 50,
    ax: "Axes | None" = None,
) -> "Figure":
    """
    Plot return distribution with VaR and CVaR highlighted.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns.
    confidence : float, optional
        Confidence level. Default is 0.95.
    title : str | None, optional
        Plot title.
    figsize : tuple[int, int], optional
        Figure size.
    bins : int, optional
        Number of histogram bins.
    ax : Axes | None, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> returns = np.random.normal(0, 0.02, 1000)
    >>> fig = plot_var(returns, confidence=0.95)
    """
    plt = _get_pyplot()

    from stochastic_engine.risk import VaR, CVaR

    returns = np.asarray(returns)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Calculate VaR and CVaR
    var = VaR(returns, confidence=confidence)
    cvar = CVaR(returns, confidence=confidence)

    var_val = var.historical()
    cvar_val = cvar.historical()

    # Plot histogram
    n, bins_edges, patches = ax.hist(
        returns, bins=bins, density=True,
        alpha=0.7, edgecolor='black'
    )

    # Color the tail
    var_threshold = -var_val
    for i, (patch, left_edge) in enumerate(zip(patches, bins_edges[:-1])):
        if left_edge < var_threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)

    # VaR line
    ax.axvline(var_threshold, color='red', linestyle='--', linewidth=2,
               label=f'{confidence:.0%} VaR = {var_val:.2%}')

    # CVaR line
    ax.axvline(-cvar_val, color='darkred', linestyle=':', linewidth=2,
               label=f'{confidence:.0%} CVaR = {cvar_val:.2%}')

    # Mean line
    ax.axvline(returns.mean(), color='green', linestyle='-', linewidth=1,
               label=f'Mean = {returns.mean():.2%}')

    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.set_title(title or f'Return Distribution with VaR ({confidence:.0%})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_drawdown(
    prices: np.ndarray,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> "Figure":
    """
    Plot price series with drawdown visualization.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices or portfolio values.
    title : str | None, optional
        Plot title.
    figsize : tuple[int, int], optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> prices = np.cumprod(1 + np.random.normal(0.0005, 0.02, 252))
    >>> fig = plot_drawdown(prices)
    """
    plt = _get_pyplot()

    from stochastic_engine.risk.metrics import max_drawdown

    prices = np.asarray(prices)
    n = len(prices)
    time = np.arange(n)

    # Calculate running max and drawdown
    running_max = np.maximum.accumulate(prices)
    drawdown = (running_max - prices) / running_max

    mdd = max_drawdown(prices)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})

    # Price plot
    axes[0].plot(time, prices, 'b-', linewidth=1, label='Price')
    axes[0].plot(time, running_max, 'g--', linewidth=1, alpha=0.7, label='Running Max')
    axes[0].fill_between(time, prices, running_max, alpha=0.3, color='red')
    axes[0].set_ylabel('Price')
    axes[0].set_title(title or 'Price and Drawdown')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Drawdown plot
    axes[1].fill_between(time, 0, -drawdown * 100, color='red', alpha=0.7)
    axes[1].axhline(-mdd * 100, color='darkred', linestyle='--',
                    label=f'Max Drawdown = {mdd:.1%}')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Time')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=-mdd * 100 * 1.2, top=5)

    plt.tight_layout()
    return fig


def plot_rolling_metrics(
    returns: np.ndarray,
    window: int = 21,
    metrics: list[str] | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> "Figure":
    """
    Plot rolling risk metrics.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns.
    window : int, optional
        Rolling window size. Default is 21 (monthly).
    metrics : list[str] | None, optional
        Metrics to plot. Default is ["volatility", "sharpe", "var"].
    figsize : tuple[int, int], optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    plt = _get_pyplot()

    from stochastic_engine.risk.metrics import sharpe_ratio, volatility

    returns = np.asarray(returns)
    n = len(returns)

    if metrics is None:
        metrics = ["volatility", "sharpe", "var"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    time = np.arange(window, n)

    for i, metric in enumerate(metrics):
        values = []

        for j in range(window, n):
            window_returns = returns[j - window:j]

            if metric == "volatility":
                val = volatility(window_returns, periods_per_year=252)
            elif metric == "sharpe":
                val = sharpe_ratio(window_returns, periods_per_year=252)
            elif metric == "var":
                from stochastic_engine.risk import VaR
                val = VaR(window_returns, confidence=0.95).historical()
            else:
                continue

            values.append(val)

        axes[i].plot(time, values, linewidth=1)
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(np.mean(values), color='red', linestyle='--',
                        alpha=0.7, label=f'Mean: {np.mean(values):.4f}')
        axes[i].legend(loc='upper right')

    axes[0].set_title(f'Rolling Risk Metrics (window={window})')
    axes[-1].set_xlabel('Time')

    plt.tight_layout()
    return fig

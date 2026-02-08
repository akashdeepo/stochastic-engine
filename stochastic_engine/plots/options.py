"""Option payoff and Greeks visualization."""

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

from stochastic_engine.pricing.black_scholes import BlackScholes


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


def plot_payoff(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    S_range: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (10, 6),
    ax: "Axes | None" = None,
) -> "Figure":
    """
    Plot option payoff diagram with current value.

    Shows the option payoff at expiration and current option value
    as a function of the underlying price.

    Parameters
    ----------
    S : float
        Current underlying price (for reference line).
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    option_type : {"call", "put"}, optional
        Option type. Default is "call".
    q : float, optional
        Dividend yield. Default is 0.
    S_range : tuple[float, float] | None, optional
        Range of underlying prices to plot. Default is (0.5*K, 1.5*K).
    figsize : tuple[int, int], optional
        Figure size. Default is (10, 6).
    ax : Axes | None, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> from stochastic_engine.plots import plot_payoff
    >>> fig = plot_payoff(S=100, K=105, T=0.5, r=0.05, sigma=0.2)
    """
    plt = _get_pyplot()

    if S_range is None:
        S_range = (0.5 * K, 1.5 * K)

    S_vals = np.linspace(S_range[0], S_range[1], 200)

    # Calculate payoffs at expiration
    if option_type == "call":
        payoffs = np.maximum(S_vals - K, 0)
    else:
        payoffs = np.maximum(K - S_vals, 0)

    # Calculate current option values
    option_values = BlackScholes(
        S=S_vals, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type
    ).price

    # Current option price
    current_price = BlackScholes(
        S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type
    ).price

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot payoff at expiration
    ax.plot(S_vals, payoffs, 'b--', linewidth=2, label='Payoff at Expiration')

    # Plot current option value
    ax.plot(S_vals, option_values, 'g-', linewidth=2, label=f'Option Value (T={T}y)')

    # Reference lines
    ax.axvline(K, color='gray', linestyle=':', alpha=0.7, label=f'Strike K={K}')
    ax.axvline(S, color='red', linestyle='--', alpha=0.7, label=f'Current S={S}')
    ax.axhline(0, color='black', linewidth=0.5)

    # Mark current price
    ax.plot(S, current_price, 'ro', markersize=10)
    ax.annotate(f'${current_price:.2f}', (S, current_price),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold')

    # Labels
    ax.set_xlabel('Underlying Price ($)')
    ax.set_ylabel('Option Value ($)')
    ax.set_title(f'{option_type.capitalize()} Option Payoff Diagram')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S_range)

    plt.tight_layout()
    return fig


def plot_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
    q: float = 0.0,
    S_range: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> "Figure":
    """
    Plot all Greeks as a function of underlying price.

    Parameters
    ----------
    S : float
        Current underlying price (for reference).
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    option_type : {"call", "put"}, optional
        Option type. Default is "call".
    q : float, optional
        Dividend yield. Default is 0.
    S_range : tuple[float, float] | None, optional
        Range of underlying prices. Default is (0.5*K, 1.5*K).
    figsize : tuple[int, int], optional
        Figure size. Default is (12, 8).

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> from stochastic_engine.plots import plot_greeks
    >>> fig = plot_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.2)
    """
    plt = _get_pyplot()

    if S_range is None:
        S_range = (0.5 * K, 1.5 * K)

    S_vals = np.linspace(S_range[0], S_range[1], 200)

    # Calculate Greeks
    bs = BlackScholes(S=S_vals, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Price
    axes[0, 0].plot(S_vals, bs.price, 'b-', linewidth=2)
    axes[0, 0].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_title('Price')
    axes[0, 0].set_xlabel('Underlying ($)')
    axes[0, 0].set_ylabel('Option Price ($)')
    axes[0, 0].grid(True, alpha=0.3)

    # Delta
    axes[0, 1].plot(S_vals, bs.delta, 'g-', linewidth=2)
    axes[0, 1].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].axhline(0.5 if option_type == 'call' else -0.5, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Delta (Δ)')
    axes[0, 1].set_xlabel('Underlying ($)')
    axes[0, 1].set_ylabel('Delta')
    axes[0, 1].grid(True, alpha=0.3)

    # Gamma
    axes[0, 2].plot(S_vals, bs.gamma, 'r-', linewidth=2)
    axes[0, 2].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_title('Gamma (Γ)')
    axes[0, 2].set_xlabel('Underlying ($)')
    axes[0, 2].set_ylabel('Gamma')
    axes[0, 2].grid(True, alpha=0.3)

    # Vega
    axes[1, 0].plot(S_vals, bs.vega, 'm-', linewidth=2)
    axes[1, 0].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_title('Vega (ν) per 1% vol')
    axes[1, 0].set_xlabel('Underlying ($)')
    axes[1, 0].set_ylabel('Vega')
    axes[1, 0].grid(True, alpha=0.3)

    # Theta
    axes[1, 1].plot(S_vals, bs.theta, 'c-', linewidth=2)
    axes[1, 1].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Theta (Θ) daily')
    axes[1, 1].set_xlabel('Underlying ($)')
    axes[1, 1].set_ylabel('Theta')
    axes[1, 1].grid(True, alpha=0.3)

    # Rho
    axes[1, 2].plot(S_vals, bs.rho, 'y-', linewidth=2)
    axes[1, 2].axvline(K, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_title('Rho (ρ) per 1% rate')
    axes[1, 2].set_xlabel('Underlying ($)')
    axes[1, 2].set_ylabel('Rho')
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(f'{option_type.capitalize()} Option Greeks (K={K}, T={T}y, σ={sigma:.0%})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_greeks_surface(
    S: float,
    K_range: tuple[float, float],
    T_range: tuple[float, float],
    r: float,
    sigma: float,
    greek: Literal["delta", "gamma", "vega", "theta"] = "delta",
    option_type: Literal["call", "put"] = "call",
    n_points: int = 50,
    figsize: tuple[int, int] = (10, 8),
) -> "Figure":
    """
    Plot a Greek as a 3D surface over strike and time.

    Parameters
    ----------
    S : float
        Current underlying price.
    K_range : tuple[float, float]
        Range of strike prices (min, max).
    T_range : tuple[float, float]
        Range of times to expiration (min, max).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    greek : {"delta", "gamma", "vega", "theta"}, optional
        Which Greek to plot. Default is "delta".
    option_type : {"call", "put"}, optional
        Option type. Default is "call".
    n_points : int, optional
        Number of points per axis. Default is 50.
    figsize : tuple[int, int], optional
        Figure size. Default is (10, 8).

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> from stochastic_engine.plots import plot_greeks_surface
    >>> fig = plot_greeks_surface(S=100, K_range=(80, 120), T_range=(0.1, 2),
    ...                           r=0.05, sigma=0.2, greek="gamma")
    """
    plt = _get_pyplot()
    from mpl_toolkits.mplot3d import Axes3D

    K_vals = np.linspace(K_range[0], K_range[1], n_points)
    T_vals = np.linspace(T_range[0], T_range[1], n_points)
    K_grid, T_grid = np.meshgrid(K_vals, T_vals)

    # Calculate Greek values
    greek_vals = np.zeros_like(K_grid)
    for i in range(n_points):
        for j in range(n_points):
            bs = BlackScholes(
                S=S, K=K_grid[i, j], T=T_grid[i, j],
                r=r, sigma=sigma, option_type=option_type
            )
            greek_vals[i, j] = getattr(bs, greek)

    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(K_grid, T_grid, greek_vals, cmap='viridis',
                           edgecolor='none', alpha=0.8)

    ax.set_xlabel('Strike ($)')
    ax.set_ylabel('Time to Expiry (years)')
    ax.set_zlabel(greek.capitalize())
    ax.set_title(f'{option_type.capitalize()} {greek.capitalize()} Surface (S={S}, σ={sigma:.0%})')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    return fig

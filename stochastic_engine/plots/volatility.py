"""Volatility visualization tools."""

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


def plot_vol_smile(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    S: float | None = None,
    T: float | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    ax: "Axes | None" = None,
) -> "Figure":
    """
    Plot the volatility smile for a single expiration.

    Parameters
    ----------
    strikes : np.ndarray
        Array of strike prices.
    implied_vols : np.ndarray
        Array of implied volatilities (as decimals, e.g., 0.2 for 20%).
    S : float | None, optional
        Current underlying price (for ATM reference line).
    T : float | None, optional
        Time to expiration (for title).
    title : str | None, optional
        Plot title.
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
    >>> strikes = np.array([90, 95, 100, 105, 110])
    >>> ivs = np.array([0.22, 0.20, 0.19, 0.20, 0.23])
    >>> fig = plot_vol_smile(strikes, ivs, S=100)
    """
    plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot smile
    ax.plot(strikes, implied_vols * 100, 'bo-', linewidth=2, markersize=8)

    # ATM reference
    if S is not None:
        ax.axvline(S, color='red', linestyle='--', alpha=0.7, label=f'ATM (S={S})')
        ax.legend()

    ax.set_xlabel('Strike ($)')
    ax.set_ylabel('Implied Volatility (%)')

    if title:
        ax.set_title(title)
    elif T is not None:
        ax.set_title(f'Volatility Smile (T = {T:.2f}y)')
    else:
        ax.set_title('Volatility Smile')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_vol_surface(
    strikes: np.ndarray,
    expirations: np.ndarray,
    iv_surface: np.ndarray,
    S: float | None = None,
    figsize: tuple[int, int] = (12, 8),
    view_angle: tuple[int, int] = (30, 45),
) -> "Figure":
    """
    Plot the implied volatility surface as a 3D plot.

    Parameters
    ----------
    strikes : np.ndarray
        Array of strike prices.
    expirations : np.ndarray
        Array of expiration times in years.
    iv_surface : np.ndarray
        2D array of implied volatilities, shape (n_strikes, n_expirations).
    S : float | None, optional
        Current underlying price (for reference).
    figsize : tuple[int, int], optional
        Figure size. Default is (12, 8).
    view_angle : tuple[int, int], optional
        3D view angle (elevation, azimuth). Default is (30, 45).

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> strikes = np.array([90, 95, 100, 105, 110])
    >>> expirations = np.array([0.25, 0.5, 1.0])
    >>> iv_surface = np.random.uniform(0.15, 0.25, (5, 3))
    >>> fig = plot_vol_surface(strikes, expirations, iv_surface, S=100)
    """
    plt = _get_pyplot()
    from mpl_toolkits.mplot3d import Axes3D

    K_grid, T_grid = np.meshgrid(strikes, expirations, indexing='ij')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(
        K_grid, T_grid, iv_surface * 100,
        cmap='RdYlGn_r',
        edgecolor='none',
        alpha=0.8
    )

    ax.set_xlabel('Strike ($)')
    ax.set_ylabel('Time to Expiry (years)')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title('Implied Volatility Surface')
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='IV (%)')
    plt.tight_layout()
    return fig


def plot_vol_term_structure(
    expirations: np.ndarray,
    atm_vols: np.ndarray,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    ax: "Axes | None" = None,
) -> "Figure":
    """
    Plot the ATM volatility term structure.

    Parameters
    ----------
    expirations : np.ndarray
        Array of expiration times in years.
    atm_vols : np.ndarray
        Array of ATM implied volatilities.
    title : str | None, optional
        Plot title.
    figsize : tuple[int, int], optional
        Figure size.
    ax : Axes | None, optional
        Matplotlib axes.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    plt = _get_pyplot()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(expirations, atm_vols * 100, 'go-', linewidth=2, markersize=8)

    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('ATM Implied Volatility (%)')
    ax.set_title(title or 'Volatility Term Structure')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

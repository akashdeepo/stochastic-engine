"""Path visualization for stochastic processes."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

from stochastic_engine.processes.base import StochasticProcess


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


def plot_paths(
    process: StochasticProcess,
    T: float = 1.0,
    steps: int = 252,
    n_paths: int = 10,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    alpha: float = 0.7,
    show_mean: bool = True,
    ax: "Axes | None" = None,
) -> "Figure":
    """
    Plot simulated paths of a stochastic process.

    Parameters
    ----------
    process : StochasticProcess
        The stochastic process to simulate (GBM, OrnsteinUhlenbeck, etc.)
    T : float, optional
        Time horizon in years. Default is 1.0.
    steps : int, optional
        Number of time steps. Default is 252.
    n_paths : int, optional
        Number of paths to plot. Default is 10.
    title : str | None, optional
        Plot title. If None, uses process name.
    figsize : tuple[int, int], optional
        Figure size. Default is (10, 6).
    alpha : float, optional
        Path transparency. Default is 0.7.
    show_mean : bool, optional
        Whether to show theoretical mean. Default is True.
    ax : Axes | None, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> from stochastic_engine import GBM
    >>> from stochastic_engine.plots import plot_paths
    >>> gbm = GBM(S0=100, mu=0.05, sigma=0.2)
    >>> fig = plot_paths(gbm, T=1, n_paths=20)
    >>> fig.savefig("gbm_paths.png")
    """
    plt = _get_pyplot()

    # Simulate paths
    paths = process.simulate(T=T, steps=steps, n_paths=n_paths)
    t = np.linspace(0, T, steps + 1)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot paths
    for i in range(n_paths):
        ax.plot(t, paths[i], alpha=alpha, linewidth=0.8)

    # Plot theoretical mean
    if show_mean:
        mean_path = [process.mean(ti) for ti in t]
        ax.plot(t, mean_path, 'k--', linewidth=2, label='E[X(t)]')
        ax.legend()

    # Labels
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Value')
    ax.set_title(title or f'{process.__class__.__name__} Simulated Paths')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_distribution(
    process: StochasticProcess,
    t: float = 1.0,
    n_samples: int = 10000,
    bins: int = 50,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show_stats: bool = True,
    ax: "Axes | None" = None,
) -> "Figure":
    """
    Plot the distribution of a process at a specific time.

    Parameters
    ----------
    process : StochasticProcess
        The stochastic process to sample.
    t : float, optional
        Time point to sample at. Default is 1.0.
    n_samples : int, optional
        Number of samples. Default is 10000.
    bins : int, optional
        Number of histogram bins. Default is 50.
    title : str | None, optional
        Plot title.
    figsize : tuple[int, int], optional
        Figure size. Default is (10, 6).
    show_stats : bool, optional
        Whether to show mean/std annotations. Default is True.
    ax : Axes | None, optional
        Matplotlib axes to plot on.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> from stochastic_engine import GBM
    >>> from stochastic_engine.plots import plot_distribution
    >>> gbm = GBM(S0=100, mu=0.05, sigma=0.2)
    >>> fig = plot_distribution(gbm, t=1)
    """
    plt = _get_pyplot()

    # Sample the process
    samples = process.sample(t=t, n_samples=n_samples)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot histogram
    ax.hist(samples, bins=bins, density=True, alpha=0.7, edgecolor='black')

    # Theoretical mean line
    theoretical_mean = process.mean(t)
    ax.axvline(theoretical_mean, color='red', linestyle='--', linewidth=2,
               label=f'E[X({t})] = {theoretical_mean:.2f}')

    # Sample mean
    sample_mean = samples.mean()
    ax.axvline(sample_mean, color='green', linestyle=':', linewidth=2,
               label=f'Sample mean = {sample_mean:.2f}')

    # Stats annotation
    if show_stats:
        stats_text = (
            f'n = {n_samples:,}\n'
            f'Mean: {sample_mean:.4f}\n'
            f'Std: {samples.std():.4f}\n'
            f'Min: {samples.min():.4f}\n'
            f'Max: {samples.max():.4f}'
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, family='monospace')

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title or f'{process.__class__.__name__} Distribution at t={t}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

"""Visualization tools for quantitative finance."""

from stochastic_engine.plots.paths import plot_paths, plot_distribution
from stochastic_engine.plots.options import plot_payoff, plot_greeks, plot_greeks_surface
from stochastic_engine.plots.volatility import plot_vol_smile, plot_vol_surface
from stochastic_engine.plots.risk import plot_var, plot_drawdown

__all__ = [
    "plot_paths",
    "plot_distribution",
    "plot_payoff",
    "plot_greeks",
    "plot_greeks_surface",
    "plot_vol_smile",
    "plot_vol_surface",
    "plot_var",
    "plot_drawdown",
]

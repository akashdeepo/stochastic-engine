"""Geometric Brownian Motion process implementation."""

from typing import Literal

import numpy as np

from stochastic_engine.processes.base import StochasticProcess


class GBM(StochasticProcess):
    """
    Geometric Brownian Motion (GBM) process.

    Models asset prices with constant drift and volatility. The standard model
    for stock prices in Black-Scholes framework.

    The process follows the SDE:

    .. math::
        dS_t = \\mu S_t dt + \\sigma S_t dW_t

    with solution:

    .. math::
        S_t = S_0 \\exp\\left((\\mu - \\frac{\\sigma^2}{2})t + \\sigma W_t\\right)

    Parameters
    ----------
    S0 : float
        Initial asset price.
    mu : float
        Drift coefficient (expected return, annualized).
    sigma : float
        Volatility coefficient (annualized).
    seed : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> gbm = GBM(S0=100, mu=0.05, sigma=0.2)
    >>> paths = gbm.simulate(T=1, steps=252, n_paths=1000)
    >>> paths.shape
    (1000, 253)

    >>> # Sample terminal values
    >>> terminal = gbm.sample(t=1, n_samples=10000)
    >>> np.abs(terminal.mean() - 100 * np.exp(0.05)) < 1  # approximately
    True

    Notes
    -----
    - The exact simulation method uses the closed-form solution.
    - The Euler method uses Euler-Maruyama discretization.
    - For most applications, "exact" is preferred as it's both faster and exact.
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Geometric Brownian Motion.

        Parameters
        ----------
        S0 : float
            Initial asset price. Must be positive.
        mu : float
            Drift coefficient (expected return, annualized).
        sigma : float
            Volatility coefficient (annualized). Must be non-negative.
        seed : int | None, optional
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If S0 <= 0 or sigma < 0.
        """
        super().__init__(seed=seed)

        if S0 <= 0:
            raise ValueError(f"Initial price S0 must be positive, got {S0}")
        if sigma < 0:
            raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")

        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["euler", "exact"] = "exact",
    ) -> np.ndarray:
        """
        Generate sample paths of Geometric Brownian Motion.

        Parameters
        ----------
        T : float
            Time horizon (in years).
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths to simulate. Default is 1.
        method : {"euler", "exact"}, optional
            Simulation method:
            - "exact": Uses closed-form solution (recommended).
            - "euler": Uses Euler-Maruyama discretization.
            Default is "exact".

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, steps + 1) containing simulated paths.
            paths[i, j] is the price of path i at time step j.

        Examples
        --------
        >>> gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
        >>> paths = gbm.simulate(T=1, steps=252, n_paths=5)
        >>> paths.shape
        (5, 253)
        >>> paths[:, 0]  # All paths start at S0
        array([100., 100., 100., 100., 100.])
        """
        dt = T / steps
        rng = self._get_rng()

        # Generate random increments
        dW = rng.standard_normal((n_paths, steps)) * np.sqrt(dt)

        # Initialize paths array
        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.S0

        if method == "exact":
            # Exact simulation using closed-form solution
            # S_{t+dt} = S_t * exp((mu - sigma^2/2)*dt + sigma*dW)
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            for i in range(steps):
                paths[:, i + 1] = paths[:, i] * np.exp(drift + self.sigma * dW[:, i])

        elif method == "euler":
            # Euler-Maruyama discretization
            # S_{t+dt} = S_t + mu*S_t*dt + sigma*S_t*dW
            for i in range(steps):
                paths[:, i + 1] = (
                    paths[:, i]
                    + self.mu * paths[:, i] * dt
                    + self.sigma * paths[:, i] * dW[:, i]
                )
            # Ensure non-negative prices
            paths = np.maximum(paths, 0)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'exact' or 'euler'.")

        return paths

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample the process at a specific time point.

        Uses the exact distribution: S_t ~ S_0 * exp(Normal).

        Parameters
        ----------
        t : float
            Time point to sample at.
        n_samples : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing samples at time t.

        Examples
        --------
        >>> gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
        >>> samples = gbm.sample(t=1, n_samples=10000)
        >>> samples.mean()  # Should be close to 100 * exp(0.05) = 105.13
        """
        rng = self._get_rng()

        # S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*sqrt(t)*Z)
        drift = (self.mu - 0.5 * self.sigma**2) * t
        diffusion = self.sigma * np.sqrt(t)

        Z = rng.standard_normal(n_samples)
        return self.S0 * np.exp(drift + diffusion * Z)

    def mean(self, t: float) -> float:
        """
        Expected value of GBM at time t.

        For GBM: E[S_t] = S_0 * exp(mu * t)

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Expected value E[S_t].

        Examples
        --------
        >>> gbm = GBM(S0=100, mu=0.05, sigma=0.2)
        >>> gbm.mean(1)  # E[S_1] = 100 * exp(0.05)
        105.12710963760242
        """
        return self.S0 * np.exp(self.mu * t)

    def variance(self, t: float) -> float:
        """
        Variance of GBM at time t.

        For GBM: Var[S_t] = S_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Variance Var[S_t].

        Examples
        --------
        >>> gbm = GBM(S0=100, mu=0.05, sigma=0.2)
        >>> gbm.variance(1)
        453.20949287498744
        """
        return (
            self.S0**2
            * np.exp(2 * self.mu * t)
            * (np.exp(self.sigma**2 * t) - 1)
        )

    def std(self, t: float) -> float:
        """
        Standard deviation of GBM at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Standard deviation.
        """
        return np.sqrt(self.variance(t))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GBM(S0={self.S0}, mu={self.mu}, sigma={self.sigma})"

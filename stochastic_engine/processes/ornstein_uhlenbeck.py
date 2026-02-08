"""Ornstein-Uhlenbeck process implementation."""

from typing import Literal

import numpy as np

from stochastic_engine.processes.base import StochasticProcess


class OrnsteinUhlenbeck(StochasticProcess):
    """
    Ornstein-Uhlenbeck (OU) mean-reverting process.

    A continuous-time stochastic process that tends to drift towards its
    long-term mean. Widely used for modeling interest rates, volatility,
    and pairs trading strategies.

    The process follows the SDE:

    .. math::
        dX_t = \\theta (\\mu - X_t) dt + \\sigma dW_t

    with solution:

    .. math::
        X_t = \\mu + (X_0 - \\mu)e^{-\\theta t} + \\sigma \\int_0^t e^{-\\theta(t-s)} dW_s

    Parameters
    ----------
    X0 : float
        Initial value of the process.
    mu : float
        Long-term mean level.
    theta : float
        Speed of mean reversion. Higher values = faster reversion.
    sigma : float
        Volatility coefficient.
    seed : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> ou = OrnsteinUhlenbeck(X0=0.05, mu=0.03, theta=0.5, sigma=0.01)
    >>> paths = ou.simulate(T=10, steps=2520, n_paths=1000)
    >>> paths.shape
    (1000, 2521)

    >>> # Process mean-reverts to mu
    >>> ou.mean(t=100)  # Should be very close to mu
    0.03

    Notes
    -----
    - Unlike GBM, the OU process can take negative values.
    - For interest rate modeling, consider the CIR process (future version)
      which ensures non-negativity.
    - Half-life of mean reversion: t_half = ln(2) / theta
    """

    def __init__(
        self,
        X0: float,
        mu: float,
        theta: float,
        sigma: float,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Ornstein-Uhlenbeck process.

        Parameters
        ----------
        X0 : float
            Initial value of the process.
        mu : float
            Long-term mean level.
        theta : float
            Speed of mean reversion. Must be positive.
        sigma : float
            Volatility coefficient. Must be non-negative.
        seed : int | None, optional
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If theta <= 0 or sigma < 0.
        """
        super().__init__(seed=seed)

        if theta <= 0:
            raise ValueError(f"Mean reversion speed theta must be positive, got {theta}")
        if sigma < 0:
            raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")

        self.X0 = X0
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

    @property
    def half_life(self) -> float:
        """
        Half-life of mean reversion.

        The time it takes for the expected deviation from the mean
        to decrease by half.

        Returns
        -------
        float
            Half-life in the same units as the time parameter.
        """
        return np.log(2) / self.theta

    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["euler", "exact"] = "exact",
    ) -> np.ndarray:
        """
        Generate sample paths of the Ornstein-Uhlenbeck process.

        Parameters
        ----------
        T : float
            Time horizon.
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths to simulate. Default is 1.
        method : {"euler", "exact"}, optional
            Simulation method:
            - "exact": Uses exact transition density (recommended).
            - "euler": Uses Euler-Maruyama discretization.
            Default is "exact".

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, steps + 1) containing simulated paths.

        Examples
        --------
        >>> ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=1.0, sigma=0.02, seed=42)
        >>> paths = ou.simulate(T=5, steps=1000, n_paths=3)
        >>> paths[:, 0]  # All start at X0
        array([0.1, 0.1, 0.1])
        """
        dt = T / steps
        rng = self._get_rng()

        # Generate random increments
        dW = rng.standard_normal((n_paths, steps))

        # Initialize paths
        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.X0

        if method == "exact":
            # Exact simulation using transition density
            # X_{t+dt} | X_t ~ N(mu + (X_t - mu)*exp(-theta*dt), sigma_cond^2)
            exp_decay = np.exp(-self.theta * dt)
            sigma_cond = self.sigma * np.sqrt((1 - exp_decay**2) / (2 * self.theta))

            for i in range(steps):
                paths[:, i + 1] = (
                    self.mu
                    + (paths[:, i] - self.mu) * exp_decay
                    + sigma_cond * dW[:, i]
                )

        elif method == "euler":
            # Euler-Maruyama discretization
            # X_{t+dt} = X_t + theta*(mu - X_t)*dt + sigma*sqrt(dt)*Z
            sqrt_dt = np.sqrt(dt)
            for i in range(steps):
                paths[:, i + 1] = (
                    paths[:, i]
                    + self.theta * (self.mu - paths[:, i]) * dt
                    + self.sigma * sqrt_dt * dW[:, i]
                )

        else:
            raise ValueError(f"Unknown method: {method}. Use 'exact' or 'euler'.")

        return paths

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample the process at a specific time point.

        Uses the exact marginal distribution.

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
        >>> ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=1.0, sigma=0.02, seed=42)
        >>> samples = ou.sample(t=10, n_samples=10000)
        >>> np.abs(samples.mean() - 0.05) < 0.001  # Close to mu
        True
        """
        rng = self._get_rng()

        mean = self.mean(t)
        std = self.std(t)

        Z = rng.standard_normal(n_samples)
        return mean + std * Z

    def mean(self, t: float) -> float:
        """
        Expected value of OU process at time t.

        E[X_t] = mu + (X_0 - mu) * exp(-theta * t)

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Expected value E[X_t].

        Examples
        --------
        >>> ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=1.0, sigma=0.02)
        >>> ou.mean(0)  # At t=0, mean is X0
        0.1
        >>> ou.mean(100)  # At t->inf, mean approaches mu
        0.05
        """
        return self.mu + (self.X0 - self.mu) * np.exp(-self.theta * t)

    def variance(self, t: float) -> float:
        """
        Variance of OU process at time t.

        Var[X_t] = (sigma^2 / (2*theta)) * (1 - exp(-2*theta*t))

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Variance Var[X_t].

        Examples
        --------
        >>> ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=1.0, sigma=0.02)
        >>> ou.variance(0)  # At t=0, variance is 0
        0.0
        >>> ou.variance(100)  # Long-term variance
        0.0002
        """
        return (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * t))

    def std(self, t: float) -> float:
        """
        Standard deviation of OU process at time t.

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

    @property
    def long_term_variance(self) -> float:
        """
        Long-term (stationary) variance.

        As t -> infinity, Var[X_t] -> sigma^2 / (2*theta)

        Returns
        -------
        float
            Long-term variance.
        """
        return self.sigma**2 / (2 * self.theta)

    @property
    def long_term_std(self) -> float:
        """Long-term (stationary) standard deviation."""
        return np.sqrt(self.long_term_variance)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OrnsteinUhlenbeck(X0={self.X0}, mu={self.mu}, "
            f"theta={self.theta}, sigma={self.sigma})"
        )

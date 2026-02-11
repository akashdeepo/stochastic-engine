"""Correlated multi-asset simulation using Cholesky decomposition."""

from typing import Literal

import numpy as np

from stochastic_engine.processes.base import StochasticProcess


class CorrelatedGBM(StochasticProcess):
    """
    Correlated multi-asset Geometric Brownian Motion.

    Simulates N assets with individual drift/volatility and a correlation
    matrix, using Cholesky decomposition for correlated Brownian motions.

    .. math::
        dS_i = \\mu_i S_i \\, dt + \\sigma_i S_i \\, dW_i

    where :math:`dW_i dW_j = \\rho_{ij} \\, dt`.

    Parameters
    ----------
    S0 : np.ndarray
        Initial prices, shape (N,).
    mu : np.ndarray
        Drifts, shape (N,).
    sigma : np.ndarray
        Volatilities, shape (N,). Must be non-negative.
    correlation : np.ndarray
        Correlation matrix, shape (N, N). Must be symmetric,
        positive semi-definite, with unit diagonal.
    seed : int or None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    >>> cgbm = CorrelatedGBM(
    ...     S0=np.array([100, 50]),
    ...     mu=np.array([0.05, 0.08]),
    ...     sigma=np.array([0.2, 0.3]),
    ...     correlation=corr,
    ... )
    >>> paths = cgbm.simulate(T=1, steps=252, n_paths=1000)
    >>> paths.shape
    (2, 1000, 253)
    """

    def __init__(
        self,
        S0: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        correlation: np.ndarray,
        seed: int | None = None,
    ) -> None:
        """Initialize CorrelatedGBM."""
        super().__init__(seed=seed)

        self.S0 = np.asarray(S0, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.correlation = np.asarray(correlation, dtype=float)

        self.n_assets = len(self.S0)

        # Validate
        if self.S0.ndim != 1:
            raise ValueError(f"S0 must be 1D, got shape {self.S0.shape}")
        if np.any(self.S0 <= 0):
            raise ValueError("All initial prices S0 must be positive")
        if np.any(self.sigma < 0):
            raise ValueError("All volatilities sigma must be non-negative")
        if self.mu.shape != (self.n_assets,):
            raise ValueError(f"mu shape {self.mu.shape} doesn't match S0 shape {self.S0.shape}")
        if self.sigma.shape != (self.n_assets,):
            raise ValueError(
                f"sigma shape {self.sigma.shape} doesn't match S0 shape {self.S0.shape}"
            )

        self._validate_correlation(self.correlation)

        # Compute and cache Cholesky factor
        try:
            self._cholesky = np.linalg.cholesky(self.correlation)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Correlation matrix is not positive definite. "
                "Check that it is a valid correlation matrix."
            )

    def _validate_correlation(self, corr: np.ndarray) -> None:
        """Validate correlation matrix."""
        n = self.n_assets
        if corr.shape != (n, n):
            raise ValueError(
                f"Correlation matrix shape {corr.shape} doesn't match "
                f"number of assets ({n})"
            )
        if not np.allclose(corr, corr.T):
            raise ValueError("Correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0):
            raise ValueError("Correlation matrix diagonal must be all 1s")
        if np.any(corr > 1) or np.any(corr < -1):
            raise ValueError("Correlation values must be in [-1, 1]")

    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["exact"] = "exact",
    ) -> np.ndarray:
        """
        Simulate correlated multi-asset paths.

        Parameters
        ----------
        T : float
            Time horizon in years.
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths per asset. Default is 1.
        method : {"exact"}, optional
            Discretization scheme. Default is "exact".

        Returns
        -------
        np.ndarray
            Price paths of shape (n_assets, n_paths, steps + 1).
        """
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        rng = self._get_rng()
        n = self.n_assets

        # Initialize paths
        paths = np.zeros((n, n_paths, steps + 1))
        for a in range(n):
            paths[a, :, 0] = self.S0[a]

        # Precompute drifts
        drifts = (self.mu - 0.5 * self.sigma**2) * dt  # shape (n_assets,)

        for i in range(steps):
            # Independent normals: shape (n_assets, n_paths)
            Z_indep = rng.standard_normal((n, n_paths))

            # Correlated normals via Cholesky: L @ Z
            Z_corr = self._cholesky @ Z_indep  # shape (n_assets, n_paths)

            for a in range(n):
                paths[a, :, i + 1] = paths[a, :, i] * np.exp(
                    drifts[a] + self.sigma[a] * sqrt_dt * Z_corr[a]
                )

        return paths

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample prices at a specific time.

        Parameters
        ----------
        t : float
            Time point.
        n_samples : int, optional
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples of shape (n_assets, n_samples).
        """
        rng = self._get_rng()
        n = self.n_assets

        # Generate correlated log-returns
        Z_indep = rng.standard_normal((n, n_samples))
        Z_corr = self._cholesky @ Z_indep

        samples = np.zeros((n, n_samples))
        for a in range(n):
            drift = (self.mu[a] - 0.5 * self.sigma[a] ** 2) * t
            diffusion = self.sigma[a] * np.sqrt(t) * Z_corr[a]
            samples[a] = self.S0[a] * np.exp(drift + diffusion)

        return samples

    def mean(self, t: float) -> np.ndarray:
        """
        Expected value of prices at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        np.ndarray
            Expected prices, shape (n_assets,).
        """
        return self.S0 * np.exp(self.mu * t)

    def variance(self, t: float) -> np.ndarray:
        """
        Marginal variance of each asset at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        np.ndarray
            Variances, shape (n_assets,).
        """
        return self.S0**2 * np.exp(2 * self.mu * t) * (np.exp(self.sigma**2 * t) - 1)

    def covariance_matrix(self, t: float) -> np.ndarray:
        """
        Cross-asset covariance matrix of log-returns at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        np.ndarray
            Covariance matrix of shape (n_assets, n_assets).
        """
        # Cov(ln S_i(t), ln S_j(t)) = sigma_i * sigma_j * rho_ij * t
        sigma_outer = np.outer(self.sigma, self.sigma)
        return sigma_outer * self.correlation * t

    def __repr__(self) -> str:
        return (
            f"CorrelatedGBM(n_assets={self.n_assets}, "
            f"S0={self.S0}, mu={self.mu}, sigma={self.sigma})"
        )

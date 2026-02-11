"""Merton Jump-Diffusion model."""

from math import factorial
from typing import Literal

import numpy as np

from stochastic_engine.processes.base import StochasticProcess


class MertonJumpDiffusion(StochasticProcess):
    """
    Merton Jump-Diffusion model.

    GBM with compound Poisson jumps of log-normally distributed size.

    .. math::
        \\frac{dS}{S} = (\\mu - \\lambda k) \\, dt + \\sigma \\, dW + J \\, dN

    where :math:`dN \\sim \\text{Poisson}(\\lambda \\, dt)`,
    :math:`\\ln(1+J) \\sim N(\\mu_j, \\sigma_j^2)`, and
    :math:`k = e^{\\mu_j + \\sigma_j^2/2} - 1` is the expected jump size.

    Parameters
    ----------
    S0 : float
        Initial asset price (must be positive).
    mu : float
        Drift coefficient.
    sigma : float
        Diffusion volatility (must be non-negative).
    lam : float
        Jump intensity - expected number of jumps per year (must be non-negative).
    mu_j : float
        Mean of log-jump size.
    sigma_j : float
        Standard deviation of log-jump size (must be non-negative).
    seed : int or None, optional
        Random seed for reproducibility.

    Notes
    -----
    When ``lam=0``, the model reduces to standard GBM.

    The jump compensator ``k`` ensures the drift remains ``mu``
    after accounting for the expected jump contribution.

    Examples
    --------
    >>> mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2,
    ...                            lam=3, mu_j=-0.02, sigma_j=0.1)
    >>> paths = mjd.simulate(T=1, steps=252, n_paths=100)
    >>> paths.shape
    (100, 253)
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        lam: float,
        mu_j: float,
        sigma_j: float,
        seed: int | None = None,
    ) -> None:
        """Initialize Merton Jump-Diffusion model."""
        super().__init__(seed=seed)

        if S0 <= 0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if lam < 0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if sigma_j < 0:
            raise ValueError(f"sigma_j must be non-negative, got {sigma_j}")

        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j

        # Expected relative jump size
        self.k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["euler"] = "euler",
    ) -> np.ndarray:
        """
        Simulate paths of the Merton Jump-Diffusion model.

        Parameters
        ----------
        T : float
            Time horizon in years.
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths. Default is 1.
        method : {"euler"}, optional
            Discretization scheme. Default is "euler".

        Returns
        -------
        np.ndarray
            Price paths of shape (n_paths, steps + 1).
        """
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        rng = self._get_rng()

        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.S0

        # Compensated drift
        drift = (self.mu - self.lam * self.k - 0.5 * self.sigma**2) * dt

        for i in range(steps):
            # Diffusion component
            dW = sqrt_dt * rng.standard_normal(n_paths)

            # Jump component: number of jumps in this step
            n_jumps = rng.poisson(self.lam * dt, n_paths)

            # Total jump size (log-space)
            jump_log = np.zeros(n_paths)
            has_jumps = n_jumps > 0
            if has_jumps.any():
                for idx in np.where(has_jumps)[0]:
                    jump_sizes = rng.normal(self.mu_j, self.sigma_j, n_jumps[idx])
                    jump_log[idx] = jump_sizes.sum()

            # Update prices (log-space for numerical stability)
            paths[:, i + 1] = paths[:, i] * np.exp(
                drift + self.sigma * dW + jump_log
            )

        return paths

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample the price at a specific time.

        Uses simulation with fine time steps.

        Parameters
        ----------
        t : float
            Time point.
        n_samples : int, optional
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples of shape (n_samples,).
        """
        steps = max(int(t * 252), 100)
        paths = self.simulate(T=t, steps=steps, n_paths=n_samples)
        return paths[:, -1]

    def mean(self, t: float) -> float:
        """
        Expected value of price at time t.

        .. math::
            E[S_t] = S_0 e^{\\mu t}

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Expected price.
        """
        return self.S0 * np.exp(self.mu * t)

    def variance(self, t: float) -> float:
        """
        Variance of price at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Variance.
        """
        # Total variance includes diffusion and jump components
        jump_var_component = self.lam * (
            np.exp(2 * self.mu_j + 2 * self.sigma_j**2)
            - 2 * np.exp(self.mu_j + 0.5 * self.sigma_j**2)
            + 1
        )
        total_var_rate = self.sigma**2 + jump_var_component
        return self.S0**2 * np.exp(2 * self.mu * t) * (np.exp(total_var_rate * t) - 1)

    def price_european_call(
        self,
        K: float,
        T: float,
        r: float,
        n_terms: int = 50,
    ) -> float:
        """
        Price a European call using Merton's series formula.

        The price is a weighted sum of Black-Scholes prices:

        .. math::
            C = \\sum_{n=0}^{\\infty} \\frac{e^{-\\lambda' T} (\\lambda' T)^n}{n!}
            \\, BS(S, K, T, r_n, \\sigma_n)

        Parameters
        ----------
        K : float
            Strike price.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.
        n_terms : int, optional
            Number of terms in series. Default is 50.

        Returns
        -------
        float
            European call price.

        Examples
        --------
        >>> mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2,
        ...                            lam=3, mu_j=-0.02, sigma_j=0.1)
        >>> mjd.price_european_call(K=100, T=1, r=0.05)
        10.8...
        """
        from stochastic_engine.pricing.black_scholes import bs_call

        lam_prime = self.lam * (1 + self.k)
        price = 0.0

        for n in range(n_terms):
            # Adjusted parameters for n-th term
            sigma_n = np.sqrt(self.sigma**2 + n * self.sigma_j**2 / T)
            r_n = r - self.lam * self.k + n * np.log(1 + self.k) / T if self.k != 0 else r

            # Poisson weight
            weight = np.exp(-lam_prime * T) * (lam_prime * T) ** n / factorial(n)

            price += weight * bs_call(self.S0, K, T, r_n, sigma_n)

        return float(price)

    def price_european_put(
        self,
        K: float,
        T: float,
        r: float,
        n_terms: int = 50,
    ) -> float:
        """
        Price a European put using put-call parity.

        Parameters
        ----------
        K : float
            Strike price.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.
        n_terms : int, optional
            Number of terms in series. Default is 50.

        Returns
        -------
        float
            European put price.
        """
        call = self.price_european_call(K, T, r, n_terms)
        # Put-call parity: P = C - S + K*exp(-rT)
        put = call - self.S0 + K * np.exp(-r * T)
        return float(max(put, 0))

    def __repr__(self) -> str:
        return (
            f"MertonJumpDiffusion(S0={self.S0}, mu={self.mu}, sigma={self.sigma}, "
            f"lam={self.lam}, mu_j={self.mu_j}, sigma_j={self.sigma_j})"
        )

"""Monte Carlo option pricing engine."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from stochastic_engine.processes.gbm import GBM


@dataclass
class MCResult:
    """
    Monte Carlo pricing result.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_error : float
        Standard error of the estimate.
    confidence_interval : tuple[float, float]
        95% confidence interval for the price.
    n_paths : int
        Number of paths used in simulation.
    """

    price: float
    std_error: float
    confidence_interval: tuple[float, float]
    n_paths: int

    def __repr__(self) -> str:
        return (
            f"MCResult(price={self.price:.4f}, std_error={self.std_error:.4f}, "
            f"CI=({self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}))"
        )


class MonteCarloPricer:
    """
    Monte Carlo option pricing engine.

    Prices options by simulating paths of the underlying asset and
    computing the discounted expected payoff.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time to expiration in years.
    q : float, optional
        Continuous dividend yield. Default is 0.
    seed : int | None, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1)
    >>> result = mc.price_european_call(K=105, n_paths=100000)
    >>> print(f"Price: {result.price:.2f} +/- {result.std_error:.4f}")
    Price: 8.02 +/- 0.03...

    >>> # Custom payoff function
    >>> def asian_call_payoff(paths, K):
    ...     avg_price = paths.mean(axis=1)
    ...     return np.maximum(avg_price - K, 0)
    >>> result = mc.price_custom(asian_call_payoff, K=100, n_paths=50000, steps=252)

    Notes
    -----
    The Monte Carlo method estimates the option price as:

    .. math::
        V_0 = e^{-rT} \\mathbb{E}^Q[\\text{payoff}(S_T)]

    where the expectation is taken under the risk-neutral measure Q.

    Variance reduction techniques:
    - Antithetic variates: Uses pairs of negatively correlated paths
    - Control variates: Uses known expected values to reduce variance
    """

    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        q: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """Initialize Monte Carlo pricer."""
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.seed = seed

        # Risk-neutral drift
        self._mu = r - q

        # Create GBM process with risk-neutral drift
        self._process = GBM(S0=S0, mu=self._mu, sigma=sigma, seed=seed)

    def _simulate_paths(
        self,
        n_paths: int,
        steps: int,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulate paths with optional antithetic variates.

        Parameters
        ----------
        n_paths : int
            Number of paths to simulate.
        steps : int
            Number of time steps.
        antithetic : bool
            Whether to use antithetic variates.

        Returns
        -------
        np.ndarray
            Simulated paths of shape (n_paths, steps + 1).
        """
        if antithetic:
            # Simulate half the paths
            half_paths = n_paths // 2
            self._process.reset_rng(self.seed)

            # Generate random increments
            dt = self.T / steps
            rng = self._process._get_rng()
            dW = rng.standard_normal((half_paths, steps)) * np.sqrt(dt)

            # Create paths and antithetic paths
            paths = np.zeros((half_paths, steps + 1))
            anti_paths = np.zeros((half_paths, steps + 1))
            paths[:, 0] = self.S0
            anti_paths[:, 0] = self.S0

            drift = (self._mu - 0.5 * self.sigma**2) * dt

            for i in range(steps):
                paths[:, i + 1] = paths[:, i] * np.exp(drift + self.sigma * dW[:, i])
                anti_paths[:, i + 1] = anti_paths[:, i] * np.exp(
                    drift - self.sigma * dW[:, i]
                )

            # Combine paths
            all_paths = np.vstack([paths, anti_paths])

            # If n_paths was odd, add one more path
            if n_paths % 2 == 1:
                extra_path = self._process.simulate(
                    T=self.T, steps=steps, n_paths=1
                )
                all_paths = np.vstack([all_paths, extra_path])

            return all_paths
        else:
            self._process.reset_rng(self.seed)
            return self._process.simulate(T=self.T, steps=steps, n_paths=n_paths)

    def _compute_result(
        self,
        payoffs: np.ndarray,
        discount_factor: float,
    ) -> MCResult:
        """Compute MC result from payoffs."""
        discounted = payoffs * discount_factor
        price = discounted.mean()
        std_error = discounted.std() / np.sqrt(len(payoffs))

        # 95% confidence interval
        z = 1.96
        ci = (price - z * std_error, price + z * std_error)

        return MCResult(
            price=float(price),
            std_error=float(std_error),
            confidence_interval=ci,
            n_paths=len(payoffs),
        )

    def price_european_call(
        self,
        K: float,
        n_paths: int = 100000,
        steps: int = 1,
        antithetic: bool = True,
    ) -> MCResult:
        """
        Price a European call option.

        Parameters
        ----------
        K : float
            Strike price.
        n_paths : int, optional
            Number of simulation paths. Default is 100000.
        steps : int, optional
            Number of time steps. Default is 1 (terminal value only).
        antithetic : bool, optional
            Use antithetic variates. Default is True.

        Returns
        -------
        MCResult
            Pricing result with price and confidence interval.

        Examples
        --------
        >>> mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        >>> result = mc.price_european_call(K=105, n_paths=100000)
        >>> 7.5 < result.price < 8.5  # Should be close to BS price of ~8.02
        True
        """
        paths = self._simulate_paths(n_paths, steps, antithetic)
        terminal = paths[:, -1]
        payoffs = np.maximum(terminal - K, 0)
        discount_factor = np.exp(-self.r * self.T)
        return self._compute_result(payoffs, discount_factor)

    def price_european_put(
        self,
        K: float,
        n_paths: int = 100000,
        steps: int = 1,
        antithetic: bool = True,
    ) -> MCResult:
        """
        Price a European put option.

        Parameters
        ----------
        K : float
            Strike price.
        n_paths : int, optional
            Number of simulation paths. Default is 100000.
        steps : int, optional
            Number of time steps. Default is 1 (terminal value only).
        antithetic : bool, optional
            Use antithetic variates. Default is True.

        Returns
        -------
        MCResult
            Pricing result with price and confidence interval.

        Examples
        --------
        >>> mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        >>> result = mc.price_european_put(K=105, n_paths=100000)
        >>> 7.4 < result.price < 8.4  # Should be close to BS price of ~7.90
        True
        """
        paths = self._simulate_paths(n_paths, steps, antithetic)
        terminal = paths[:, -1]
        payoffs = np.maximum(K - terminal, 0)
        discount_factor = np.exp(-self.r * self.T)
        return self._compute_result(payoffs, discount_factor)

    def price_asian_call(
        self,
        K: float,
        n_paths: int = 100000,
        steps: int = 252,
        averaging: Literal["arithmetic", "geometric"] = "arithmetic",
        antithetic: bool = True,
    ) -> MCResult:
        """
        Price an Asian (average price) call option.

        Parameters
        ----------
        K : float
            Strike price.
        n_paths : int, optional
            Number of simulation paths. Default is 100000.
        steps : int, optional
            Number of averaging points. Default is 252 (daily for 1 year).
        averaging : {"arithmetic", "geometric"}, optional
            Type of averaging. Default is "arithmetic".
        antithetic : bool, optional
            Use antithetic variates. Default is True.

        Returns
        -------
        MCResult
            Pricing result with price and confidence interval.

        Examples
        --------
        >>> mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        >>> result = mc.price_asian_call(K=100, n_paths=50000)
        >>> result.price > 0
        True
        """
        paths = self._simulate_paths(n_paths, steps, antithetic)

        if averaging == "arithmetic":
            avg_price = paths[:, 1:].mean(axis=1)  # Exclude initial value
        elif averaging == "geometric":
            avg_price = np.exp(np.log(paths[:, 1:]).mean(axis=1))
        else:
            raise ValueError(f"averaging must be 'arithmetic' or 'geometric', got {averaging}")

        payoffs = np.maximum(avg_price - K, 0)
        discount_factor = np.exp(-self.r * self.T)
        return self._compute_result(payoffs, discount_factor)

    def price_asian_put(
        self,
        K: float,
        n_paths: int = 100000,
        steps: int = 252,
        averaging: Literal["arithmetic", "geometric"] = "arithmetic",
        antithetic: bool = True,
    ) -> MCResult:
        """
        Price an Asian (average price) put option.

        Parameters
        ----------
        K : float
            Strike price.
        n_paths : int, optional
            Number of simulation paths. Default is 100000.
        steps : int, optional
            Number of averaging points. Default is 252.
        averaging : {"arithmetic", "geometric"}, optional
            Type of averaging. Default is "arithmetic".
        antithetic : bool, optional
            Use antithetic variates. Default is True.

        Returns
        -------
        MCResult
            Pricing result with price and confidence interval.
        """
        paths = self._simulate_paths(n_paths, steps, antithetic)

        if averaging == "arithmetic":
            avg_price = paths[:, 1:].mean(axis=1)
        elif averaging == "geometric":
            avg_price = np.exp(np.log(paths[:, 1:]).mean(axis=1))
        else:
            raise ValueError(f"averaging must be 'arithmetic' or 'geometric', got {averaging}")

        payoffs = np.maximum(K - avg_price, 0)
        discount_factor = np.exp(-self.r * self.T)
        return self._compute_result(payoffs, discount_factor)

    def price_custom(
        self,
        payoff_func: Callable[[np.ndarray], np.ndarray],
        n_paths: int = 100000,
        steps: int = 252,
        antithetic: bool = True,
        **payoff_kwargs,
    ) -> MCResult:
        """
        Price an option with a custom payoff function.

        Parameters
        ----------
        payoff_func : Callable[[np.ndarray], np.ndarray]
            Function that takes paths array of shape (n_paths, steps + 1)
            and returns payoffs array of shape (n_paths,).
            Additional kwargs are passed to this function.
        n_paths : int, optional
            Number of simulation paths. Default is 100000.
        steps : int, optional
            Number of time steps. Default is 252.
        antithetic : bool, optional
            Use antithetic variates. Default is True.
        **payoff_kwargs
            Additional keyword arguments passed to payoff_func.

        Returns
        -------
        MCResult
            Pricing result with price and confidence interval.

        Examples
        --------
        >>> mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        >>> def lookback_call(paths):
        ...     return paths[:, -1] - paths.min(axis=1)
        >>> result = mc.price_custom(lookback_call, n_paths=50000)
        >>> result.price > 0
        True
        """
        paths = self._simulate_paths(n_paths, steps, antithetic)
        payoffs = payoff_func(paths, **payoff_kwargs)
        discount_factor = np.exp(-self.r * self.T)
        return self._compute_result(payoffs, discount_factor)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MonteCarloPricer(S0={self.S0}, r={self.r}, "
            f"sigma={self.sigma}, T={self.T})"
        )

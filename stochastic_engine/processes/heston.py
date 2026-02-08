"""Heston stochastic volatility model."""

from typing import Literal

import numpy as np
from scipy import integrate

from stochastic_engine.processes.base import StochasticProcess


class Heston(StochasticProcess):
    """
    Heston stochastic volatility model.

    A two-factor model where both price and volatility are stochastic.
    Captures the volatility smile observed in option markets.

    The model follows the coupled SDEs:

    .. math::
        dS_t = \\mu S_t \\, dt + \\sqrt{v_t} S_t \\, dW^1_t

        dv_t = \\kappa (\\theta - v_t) \\, dt + \\xi \\sqrt{v_t} \\, dW^2_t

    where :math:`dW^1_t dW^2_t = \\rho \\, dt`.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    v0 : float
        Initial variance (not volatility).
    mu : float
        Drift of the asset price.
    kappa : float
        Speed of mean reversion of variance.
    theta : float
        Long-term variance.
    xi : float
        Volatility of variance (vol of vol).
    rho : float
        Correlation between price and variance Brownian motions.
        Typically negative (leverage effect).
    seed : int | None, optional
        Random seed for reproducibility.

    Attributes
    ----------
    feller_satisfied : bool
        Whether the Feller condition (2*kappa*theta > xi^2) is satisfied.
        If not, variance can hit zero.

    Examples
    --------
    >>> heston = Heston(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04,
    ...                 xi=0.3, rho=-0.7)
    >>> paths, variances = heston.simulate(T=1, steps=252, n_paths=1000)
    >>> paths.shape
    (1000, 253)

    Notes
    -----
    The Heston model is widely used because:
    - It captures volatility smile/skew
    - It has a semi-closed form solution for European options
    - The parameters have intuitive interpretations

    Typical parameter values:
    - kappa: 1-5 (mean reversion speed)
    - theta: 0.01-0.09 (long-term variance, i.e., 10%-30% vol)
    - xi: 0.1-0.5 (vol of vol)
    - rho: -0.9 to -0.3 (leverage effect)
    """

    def __init__(
        self,
        S0: float,
        v0: float,
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        seed: int | None = None,
    ) -> None:
        """Initialize Heston model."""
        super().__init__(seed=seed)

        if S0 <= 0:
            raise ValueError(f"Initial price S0 must be positive, got {S0}")
        if v0 < 0:
            raise ValueError(f"Initial variance v0 must be non-negative, got {v0}")
        if theta < 0:
            raise ValueError(f"Long-term variance theta must be non-negative, got {theta}")
        if kappa <= 0:
            raise ValueError(f"Mean reversion kappa must be positive, got {kappa}")
        if xi < 0:
            raise ValueError(f"Vol of vol xi must be non-negative, got {xi}")
        if not -1 <= rho <= 1:
            raise ValueError(f"Correlation rho must be in [-1, 1], got {rho}")

        self.S0 = S0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    @property
    def feller_satisfied(self) -> bool:
        """
        Check if Feller condition is satisfied.

        The Feller condition 2*kappa*theta > xi^2 ensures that
        the variance process stays strictly positive.
        """
        return 2 * self.kappa * self.theta > self.xi**2

    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["euler", "milstein", "exact"] = "euler",
        return_variance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths of the Heston model.

        Parameters
        ----------
        T : float
            Time horizon in years.
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths. Default is 1.
        method : {"euler", "milstein"}, optional
            Discretization scheme. Default is "euler".
        return_variance : bool, optional
            If True, also return variance paths. Default is False.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            If return_variance is False: price paths of shape (n_paths, steps+1).
            If return_variance is True: tuple of (price_paths, variance_paths).

        Examples
        --------
        >>> heston = Heston(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04,
        ...                 xi=0.3, rho=-0.7, seed=42)
        >>> prices, variances = heston.simulate(T=1, steps=252, n_paths=100,
        ...                                      return_variance=True)
        >>> prices.shape, variances.shape
        ((100, 253), (100, 253))
        """
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        rng = self._get_rng()

        # Initialize arrays
        S = np.zeros((n_paths, steps + 1))
        v = np.zeros((n_paths, steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # Generate correlated Brownian motions
        for i in range(steps):
            # Independent normals
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)

            # Correlated normals
            dW1 = Z1 * sqrt_dt
            dW2 = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * sqrt_dt

            # Current variance (ensure non-negative)
            v_curr = np.maximum(v[:, i], 0)
            sqrt_v = np.sqrt(v_curr)

            if method == "euler":
                # Euler-Maruyama discretization
                S[:, i + 1] = S[:, i] * (1 + self.mu * dt + sqrt_v * dW1)
                v[:, i + 1] = (
                    v_curr
                    + self.kappa * (self.theta - v_curr) * dt
                    + self.xi * sqrt_v * dW2
                )

            elif method == "milstein":
                # Milstein scheme for variance
                S[:, i + 1] = S[:, i] * np.exp(
                    (self.mu - 0.5 * v_curr) * dt + sqrt_v * dW1
                )
                v[:, i + 1] = (
                    v_curr
                    + self.kappa * (self.theta - v_curr) * dt
                    + self.xi * sqrt_v * dW2
                    + 0.25 * self.xi**2 * (dW2**2 - dt)
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            # Ensure non-negative (reflection or absorption)
            v[:, i + 1] = np.maximum(v[:, i + 1], 0)
            S[:, i + 1] = np.maximum(S[:, i + 1], 0)

        if return_variance:
            return S, v
        return S

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
        # Use enough steps for accuracy
        steps = max(int(t * 252), 100)
        paths = self.simulate(T=t, steps=steps, n_paths=n_samples)
        return paths[:, -1]

    def mean(self, t: float) -> float:
        """
        Expected value of price at time t.

        For Heston: E[S_t] = S_0 * exp(mu * t)

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
        Variance of price at time t (approximate).

        This is an approximation; the exact formula is complex.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Approximate variance.
        """
        # Approximate using average variance
        avg_var = self.theta + (self.v0 - self.theta) * (1 - np.exp(-self.kappa * t)) / (self.kappa * t) if t > 0 else self.v0
        return self.S0**2 * np.exp(2 * self.mu * t) * (np.exp(avg_var * t) - 1)

    def characteristic_function(self, u: complex, t: float, r: float) -> complex:
        """
        Characteristic function for option pricing.

        Used in the Heston closed-form option pricing formula.

        Parameters
        ----------
        u : complex
            Argument of characteristic function.
        t : float
            Time to maturity.
        r : float
            Risk-free rate.

        Returns
        -------
        complex
            Value of characteristic function.
        """
        # Heston characteristic function parameters
        a = self.kappa * self.theta
        b = self.kappa

        d = np.sqrt((self.rho * self.xi * 1j * u - b)**2 + self.xi**2 * (1j * u + u**2))
        g = (b - self.rho * self.xi * 1j * u - d) / (b - self.rho * self.xi * 1j * u + d)

        C = r * 1j * u * t + (a / self.xi**2) * (
            (b - self.rho * self.xi * 1j * u - d) * t
            - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
        )

        D = ((b - self.rho * self.xi * 1j * u - d) / self.xi**2) * (
            (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))
        )

        return np.exp(C + D * self.v0 + 1j * u * np.log(self.S0))

    def price_european_call(
        self,
        K: float,
        T: float,
        r: float,
        n_points: int = 1000,
    ) -> float:
        """
        Price a European call option using the Heston closed-form formula.

        Uses numerical integration of the characteristic function.

        Parameters
        ----------
        K : float
            Strike price.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.
        n_points : int, optional
            Integration points. Default is 1000.

        Returns
        -------
        float
            Call option price.

        Examples
        --------
        >>> heston = Heston(S0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04,
        ...                 xi=0.3, rho=-0.7)
        >>> heston.price_european_call(K=100, T=1, r=0.05)
        10.5...
        """
        def integrand1(u):
            cf = self.characteristic_function(u - 1j, T, r)
            return np.real(
                np.exp(-1j * u * np.log(K)) * cf / (1j * u * self.S0 * np.exp(r * T))
            )

        def integrand2(u):
            cf = self.characteristic_function(u, T, r)
            return np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u))

        # Numerical integration
        P1 = 0.5 + (1 / np.pi) * integrate.quad(integrand1, 0, 100, limit=n_points)[0]
        P2 = 0.5 + (1 / np.pi) * integrate.quad(integrand2, 0, 100, limit=n_points)[0]

        call_price = self.S0 * P1 - K * np.exp(-r * T) * P2
        return max(call_price, 0)

    def __repr__(self) -> str:
        return (
            f"Heston(S0={self.S0}, v0={self.v0}, mu={self.mu}, "
            f"kappa={self.kappa}, theta={self.theta}, xi={self.xi}, rho={self.rho})"
        )

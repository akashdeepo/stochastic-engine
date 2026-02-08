"""Black-Scholes-Merton option pricing model."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass
class Greeks:
    """
    Container for option Greeks.

    Attributes
    ----------
    delta : float
        Rate of change of option price with respect to underlying price.
    gamma : float
        Rate of change of delta with respect to underlying price.
    vega : float
        Rate of change of option price with respect to volatility.
        Expressed per 1% change in volatility.
    theta : float
        Rate of change of option price with respect to time.
        Expressed as daily decay (divide by 365).
    rho : float
        Rate of change of option price with respect to interest rate.
        Expressed per 1% change in rate.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __repr__(self) -> str:
        return (
            f"Greeks(delta={self.delta:.4f}, gamma={self.gamma:.4f}, "
            f"vega={self.vega:.4f}, theta={self.theta:.4f}, rho={self.rho:.4f})"
        )


class BlackScholes:
    """
    Black-Scholes-Merton option pricing model.

    Calculates European option prices and Greeks using the closed-form
    Black-Scholes-Merton formula.

    The model assumes:
    - European-style options (exercise only at expiration)
    - No dividends (or continuous dividend yield via adjustment)
    - Constant volatility and interest rate
    - Log-normal distribution of returns
    - No transaction costs

    Parameters
    ----------
    S : float or np.ndarray
        Current underlying asset price(s).
    K : float or np.ndarray
        Strike price(s).
    T : float or np.ndarray
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float or np.ndarray
        Volatility (annualized).
    q : float, optional
        Continuous dividend yield. Default is 0.
    option_type : {"call", "put"}, optional
        Type of option. Default is "call".

    Attributes
    ----------
    price : float or np.ndarray
        Option price(s).
    delta : float or np.ndarray
        Option delta(s).
    gamma : float or np.ndarray
        Option gamma(s).
    vega : float or np.ndarray
        Option vega(s) (per 1% vol change).
    theta : float or np.ndarray
        Option theta(s) (daily decay).
    rho : float or np.ndarray
        Option rho(s) (per 1% rate change).
    greeks : Greeks
        All Greeks in a single object (only for scalar inputs).

    Examples
    --------
    >>> # Price a call option
    >>> opt = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2)
    >>> print(f"Price: {opt.price:.2f}")
    Price: 8.02

    >>> print(f"Delta: {opt.delta:.4f}")
    Delta: 0.5462

    >>> # Price a put option
    >>> put = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=0.2, option_type='put')
    >>> print(f"Put price: {put.price:.2f}")
    Put price: 7.90

    >>> # Vectorized pricing
    >>> strikes = np.array([95, 100, 105, 110])
    >>> prices = BlackScholes(S=100, K=strikes, T=1, r=0.05, sigma=0.2).price
    >>> prices.shape
    (4,)

    Notes
    -----
    The Black-Scholes formula for a European call option is:

    .. math::
        C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)

    where:

    .. math::
        d_1 = \\frac{\\ln(S_0/K) + (r - q + \\sigma^2/2)T}{\\sigma\\sqrt{T}}

        d_2 = d_1 - \\sigma\\sqrt{T}

    and N(x) is the standard normal CDF.
    """

    def __init__(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        r: float,
        sigma: float | np.ndarray,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> None:
        """Initialize Black-Scholes model and compute price/Greeks."""
        # Store parameters
        self.S = np.asarray(S)
        self.K = np.asarray(K)
        self.T = np.asarray(T)
        self.r = r
        self.sigma = np.asarray(sigma)
        self.q = q
        self.option_type = option_type.lower()

        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

        # Compute d1, d2
        self._d1, self._d2 = self._compute_d1_d2()

        # Standard normal PDF and CDF at d1, d2
        self._N_d1 = stats.norm.cdf(self._d1)
        self._N_d2 = stats.norm.cdf(self._d2)
        self._n_d1 = stats.norm.pdf(self._d1)

        # Discount factors
        self._df = np.exp(-self.r * self.T)  # Discount factor
        self._qf = np.exp(-self.q * self.T)  # Dividend discount factor

        # Compute all values
        self._price = self._compute_price()
        self._delta = self._compute_delta()
        self._gamma = self._compute_gamma()
        self._vega = self._compute_vega()
        self._theta = self._compute_theta()
        self._rho = self._compute_rho()

    def _compute_d1_d2(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute d1 and d2 terms."""
        sqrt_T = np.sqrt(self.T)
        d1 = (
            np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2

    def _compute_price(self) -> np.ndarray:
        """Compute option price."""
        if self.option_type == "call":
            price = (
                self.S * self._qf * self._N_d1
                - self.K * self._df * self._N_d2
            )
        else:  # put
            price = (
                self.K * self._df * (1 - self._N_d2)
                - self.S * self._qf * (1 - self._N_d1)
            )
        return price

    def _compute_delta(self) -> np.ndarray:
        """Compute delta."""
        if self.option_type == "call":
            return self._qf * self._N_d1
        else:  # put
            return -self._qf * (1 - self._N_d1)

    def _compute_gamma(self) -> np.ndarray:
        """Compute gamma (same for calls and puts)."""
        return self._qf * self._n_d1 / (self.S * self.sigma * np.sqrt(self.T))

    def _compute_vega(self) -> np.ndarray:
        """Compute vega (per 1% vol change)."""
        # Raw vega is dV/dsigma
        raw_vega = self.S * self._qf * self._n_d1 * np.sqrt(self.T)
        # Return per 1% change (0.01)
        return raw_vega * 0.01

    def _compute_theta(self) -> np.ndarray:
        """Compute theta (daily decay)."""
        sqrt_T = np.sqrt(self.T)
        term1 = -(self.S * self._qf * self._n_d1 * self.sigma) / (2 * sqrt_T)

        if self.option_type == "call":
            term2 = self.q * self.S * self._qf * self._N_d1
            term3 = -self.r * self.K * self._df * self._N_d2
            raw_theta = term1 - term2 - term3
        else:  # put
            term2 = self.q * self.S * self._qf * (1 - self._N_d1)
            term3 = self.r * self.K * self._df * (1 - self._N_d2)
            raw_theta = term1 + term2 + term3

        # Return daily theta (divide annual by 365)
        return raw_theta / 365

    def _compute_rho(self) -> np.ndarray:
        """Compute rho (per 1% rate change)."""
        if self.option_type == "call":
            raw_rho = self.K * self.T * self._df * self._N_d2
        else:  # put
            raw_rho = -self.K * self.T * self._df * (1 - self._N_d2)
        # Return per 1% change (0.01)
        return raw_rho * 0.01

    @property
    def price(self) -> float | np.ndarray:
        """Option price."""
        return float(self._price) if self._price.ndim == 0 else self._price

    @property
    def delta(self) -> float | np.ndarray:
        """
        Option delta.

        The rate of change of option price with respect to the underlying price.
        - Call delta: 0 to 1 (increases as S increases)
        - Put delta: -1 to 0 (decreases as S increases)
        """
        return float(self._delta) if self._delta.ndim == 0 else self._delta

    @property
    def gamma(self) -> float | np.ndarray:
        """
        Option gamma.

        The rate of change of delta with respect to the underlying price.
        Always positive for both calls and puts.
        Highest for at-the-money options near expiration.
        """
        return float(self._gamma) if self._gamma.ndim == 0 else self._gamma

    @property
    def vega(self) -> float | np.ndarray:
        """
        Option vega (per 1% vol change).

        The rate of change of option price with respect to volatility.
        Always positive for both calls and puts.
        """
        return float(self._vega) if self._vega.ndim == 0 else self._vega

    @property
    def theta(self) -> float | np.ndarray:
        """
        Option theta (daily decay).

        The rate of change of option price with respect to time.
        Usually negative (time decay).
        """
        return float(self._theta) if self._theta.ndim == 0 else self._theta

    @property
    def rho(self) -> float | np.ndarray:
        """
        Option rho (per 1% rate change).

        The rate of change of option price with respect to interest rate.
        - Call rho: positive
        - Put rho: negative
        """
        return float(self._rho) if self._rho.ndim == 0 else self._rho

    @property
    def greeks(self) -> Greeks:
        """
        All Greeks as a Greeks object.

        Only available for scalar inputs.

        Returns
        -------
        Greeks
            Container with all Greek values.

        Raises
        ------
        ValueError
            If inputs are arrays (use individual properties instead).
        """
        if self._price.ndim != 0:
            raise ValueError(
                "greeks property only available for scalar inputs. "
                "Use individual properties (delta, gamma, etc.) for arrays."
            )
        return Greeks(
            delta=self.delta,
            gamma=self.gamma,
            vega=self.vega,
            theta=self.theta,
            rho=self.rho,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        if self._price.ndim == 0:
            return (
                f"BlackScholes({self.option_type}, S={float(self.S):.2f}, "
                f"K={float(self.K):.2f}, T={float(self.T):.4f}, "
                f"price={self.price:.4f})"
            )
        return (
            f"BlackScholes({self.option_type}, {len(self._price)} options)"
        )


# Convenience functions for quick calculations


def bs_call(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate Black-Scholes European call option price.

    Parameters
    ----------
    S : float or np.ndarray
        Current underlying asset price(s).
    K : float or np.ndarray
        Strike price(s).
    T : float or np.ndarray
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float or np.ndarray
        Volatility (annualized).
    q : float, optional
        Continuous dividend yield. Default is 0.

    Returns
    -------
    float or np.ndarray
        Call option price(s).

    Examples
    --------
    >>> bs_call(S=100, K=105, T=1, r=0.05, sigma=0.2)
    8.021...
    """
    return BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call").price


def bs_put(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float,
    sigma: float | np.ndarray,
    q: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate Black-Scholes European put option price.

    Parameters
    ----------
    S : float or np.ndarray
        Current underlying asset price(s).
    K : float or np.ndarray
        Strike price(s).
    T : float or np.ndarray
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float or np.ndarray
        Volatility (annualized).
    q : float, optional
        Continuous dividend yield. Default is 0.

    Returns
    -------
    float or np.ndarray
        Put option price(s).

    Examples
    --------
    >>> bs_put(S=100, K=105, T=1, r=0.05, sigma=0.2)
    7.900...
    """
    return BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put").price

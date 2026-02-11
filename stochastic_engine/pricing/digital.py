"""Digital (Binary) option pricing."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass
class DigitalResult:
    """
    Result from digital option pricing.

    Attributes
    ----------
    price : float
        Option price.
    delta : float
        Price sensitivity to underlying.
    gamma : float
        Delta sensitivity to underlying.
    vega : float
        Price sensitivity to volatility.
    digital_type : str
        "cash-or-nothing" or "asset-or-nothing".
    option_type : str
        "call" or "put".
    """

    price: float
    delta: float
    gamma: float
    vega: float
    digital_type: str
    option_type: str

    def __repr__(self) -> str:
        return (
            f"DigitalResult(price={self.price:.4f}, delta={self.delta:.4f}, "
            f"type={self.digital_type} {self.option_type})"
        )


class DigitalOption:
    """
    Digital (Binary) option pricer.

    Types:
    - Cash-or-nothing: pays fixed amount Q if in-the-money at expiry
    - Asset-or-nothing: pays asset value S_T if in-the-money at expiry

    Closed-form pricing under Black-Scholes assumptions.

    Parameters
    ----------
    S : float or np.ndarray
        Current asset price.
    K : float or np.ndarray
        Strike price.
    T : float or np.ndarray
        Time to expiration in years.
    r : float
        Risk-free rate.
    sigma : float or np.ndarray
        Volatility.
    q : float, optional
        Continuous dividend yield. Default is 0.
    Q : float, optional
        Cash amount for cash-or-nothing. Default is 1.
    option_type : {"call", "put"}, optional
        Option type. Default is "call".
    digital_type : {"cash-or-nothing", "asset-or-nothing"}, optional
        Digital option type. Default is "cash-or-nothing".

    Examples
    --------
    >>> opt = DigitalOption(S=100, K=105, T=1, r=0.05, sigma=0.2)
    >>> opt.price
    0.41...

    Notes
    -----
    Key identity: ``BS_call = AoN_call - K * CoN_call``

    This can be used for cross-validation with BlackScholes prices.
    """

    def __init__(
        self,
        S: float | np.ndarray,
        K: float | np.ndarray,
        T: float | np.ndarray,
        r: float,
        sigma: float | np.ndarray,
        q: float = 0.0,
        Q: float = 1.0,
        option_type: Literal["call", "put"] = "call",
        digital_type: Literal["cash-or-nothing", "asset-or-nothing"] = "cash-or-nothing",
    ) -> None:
        """Initialize DigitalOption."""
        self.S = np.asarray(S, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.r = r
        self.sigma = np.asarray(sigma, dtype=float)
        self.q = q
        self.Q = Q
        self.option_type = option_type
        self.digital_type = digital_type

        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        if digital_type not in ("cash-or-nothing", "asset-or-nothing"):
            raise ValueError(
                f"digital_type must be 'cash-or-nothing' or 'asset-or-nothing', "
                f"got {digital_type}"
            )

        # Compute d1, d2
        sqrt_T = np.sqrt(self.T)
        self._d1 = (
            np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * sqrt_T)
        self._d2 = self._d1 - self.sigma * sqrt_T

        # Discount factors
        self._df = np.exp(-self.r * self.T)
        self._qf = np.exp(-self.q * self.T)

    @property
    def price(self) -> float | np.ndarray:
        """Option price."""
        if self.digital_type == "cash-or-nothing":
            if self.option_type == "call":
                p = self.Q * self._df * stats.norm.cdf(self._d2)
            else:
                p = self.Q * self._df * stats.norm.cdf(-self._d2)
        else:  # asset-or-nothing
            if self.option_type == "call":
                p = self.S * self._qf * stats.norm.cdf(self._d1)
            else:
                p = self.S * self._qf * stats.norm.cdf(-self._d1)

        return float(p) if np.ndim(p) == 0 else p

    @property
    def delta(self) -> float | np.ndarray:
        """Delta - price sensitivity to underlying."""
        sqrt_T = np.sqrt(self.T)
        n_d2 = stats.norm.pdf(self._d2)

        if self.digital_type == "cash-or-nothing":
            if self.option_type == "call":
                d = self.Q * self._df * n_d2 / (self.S * self.sigma * sqrt_T)
            else:
                d = -self.Q * self._df * n_d2 / (self.S * self.sigma * sqrt_T)
        else:  # asset-or-nothing
            n_d1 = stats.norm.pdf(self._d1)
            if self.option_type == "call":
                d = self._qf * (
                    stats.norm.cdf(self._d1) + n_d1 / (self.sigma * sqrt_T)
                )
            else:
                d = self._qf * (
                    -stats.norm.cdf(-self._d1) + n_d1 / (self.sigma * sqrt_T)
                )

        return float(d) if np.ndim(d) == 0 else d

    @property
    def gamma(self) -> float | np.ndarray:
        """Gamma - delta sensitivity to underlying."""
        sqrt_T = np.sqrt(self.T)
        n_d2 = stats.norm.pdf(self._d2)

        if self.digital_type == "cash-or-nothing":
            if self.option_type == "call":
                g = -self.Q * self._df * n_d2 * self._d1 / (
                    self.S**2 * self.sigma**2 * self.T
                )
            else:
                g = self.Q * self._df * n_d2 * self._d1 / (
                    self.S**2 * self.sigma**2 * self.T
                )
        else:  # asset-or-nothing
            n_d1 = stats.norm.pdf(self._d1)
            if self.option_type == "call":
                g = self._qf * n_d1 * (1 - self._d1 / (self.sigma * sqrt_T)) / (
                    self.S * self.sigma * sqrt_T
                )
            else:
                g = -self._qf * n_d1 * (1 + self._d1 / (self.sigma * sqrt_T)) / (
                    self.S * self.sigma * sqrt_T
                )

        return float(g) if np.ndim(g) == 0 else g

    @property
    def vega(self) -> float | np.ndarray:
        """Vega - price sensitivity to volatility (per 1% change)."""
        sqrt_T = np.sqrt(self.T)
        n_d2 = stats.norm.pdf(self._d2)

        if self.digital_type == "cash-or-nothing":
            if self.option_type == "call":
                v = -self.Q * self._df * n_d2 * (self._d1 / self.sigma - sqrt_T)
            else:
                v = self.Q * self._df * n_d2 * (self._d1 / self.sigma - sqrt_T)
        else:  # asset-or-nothing
            n_d1 = stats.norm.pdf(self._d1)
            if self.option_type == "call":
                v = self.S * self._qf * n_d1 * (
                    -self._d2 / self.sigma
                )
            else:
                v = -self.S * self._qf * n_d1 * (
                    -self._d2 / self.sigma
                )

        # Scale to per 1% vol change
        result = v * 0.01
        return float(result) if np.ndim(result) == 0 else result

    @property
    def result(self) -> DigitalResult:
        """Full result with all Greeks."""
        return DigitalResult(
            price=float(np.asarray(self.price).mean()),
            delta=float(np.asarray(self.delta).mean()),
            gamma=float(np.asarray(self.gamma).mean()),
            vega=float(np.asarray(self.vega).mean()),
            digital_type=self.digital_type,
            option_type=self.option_type,
        )

    def __repr__(self) -> str:
        return (
            f"DigitalOption({self.digital_type} {self.option_type}, "
            f"S={self.S}, K={self.K}, T={self.T})"
        )


# Convenience functions

def cash_or_nothing_call(
    S: float, K: float, T: float, r: float, sigma: float,
    Q: float = 1.0, q: float = 0.0,
) -> float:
    """
    Price a cash-or-nothing call option.

    Pays Q if S_T > K at expiry.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    Q : float, optional
        Cash amount paid. Default is 1.
    q : float, optional
        Dividend yield. Default is 0.

    Returns
    -------
    float
        Option price.
    """
    return DigitalOption(S, K, T, r, sigma, q, Q, "call", "cash-or-nothing").price


def cash_or_nothing_put(
    S: float, K: float, T: float, r: float, sigma: float,
    Q: float = 1.0, q: float = 0.0,
) -> float:
    """
    Price a cash-or-nothing put option.

    Pays Q if S_T < K at expiry.
    """
    return DigitalOption(S, K, T, r, sigma, q, Q, "put", "cash-or-nothing").price


def asset_or_nothing_call(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """
    Price an asset-or-nothing call option.

    Pays S_T if S_T > K at expiry.
    """
    return DigitalOption(S, K, T, r, sigma, q, 1.0, "call", "asset-or-nothing").price


def asset_or_nothing_put(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0,
) -> float:
    """
    Price an asset-or-nothing put option.

    Pays S_T if S_T < K at expiry.
    """
    return DigitalOption(S, K, T, r, sigma, q, 1.0, "put", "asset-or-nothing").price

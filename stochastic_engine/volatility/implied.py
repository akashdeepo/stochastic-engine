"""Implied volatility calculation."""

from typing import Literal

import numpy as np
from scipy import optimize

from stochastic_engine.pricing.black_scholes import BlackScholes


class ImpliedVolSolver:
    """
    Implied volatility solver using various numerical methods.

    Finds the volatility that makes the Black-Scholes price match
    the market price.

    Parameters
    ----------
    S : float
        Current underlying asset price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate.
    q : float, optional
        Continuous dividend yield. Default is 0.
    option_type : {"call", "put"}, optional
        Type of option. Default is "call".

    Examples
    --------
    >>> solver = ImpliedVolSolver(S=100, K=105, T=1, r=0.05)
    >>> iv = solver.solve(market_price=8.02)
    >>> np.abs(iv - 0.2) < 0.001  # Should recover ~20% vol
    True

    Notes
    -----
    The implied volatility σ_iv satisfies:

    .. math::
        BS(S, K, T, r, \\sigma_{iv}) = P_{market}

    This is solved numerically since there's no closed-form inverse.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
    ) -> None:
        """Initialize the implied volatility solver."""
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.option_type = option_type

    def _bs_price(self, sigma: float) -> float:
        """Calculate Black-Scholes price for a given volatility."""
        return BlackScholes(
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=sigma,
            q=self.q,
            option_type=self.option_type,
        ).price

    def _bs_vega(self, sigma: float) -> float:
        """Calculate Black-Scholes vega for Newton-Raphson."""
        bs = BlackScholes(
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=sigma,
            q=self.q,
            option_type=self.option_type,
        )
        # Raw vega (not scaled)
        return bs.vega / 0.01

    def solve(
        self,
        market_price: float,
        method: Literal["newton", "bisection", "brent"] = "brent",
        initial_guess: float = 0.2,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """
        Solve for implied volatility.

        Parameters
        ----------
        market_price : float
            Observed market price of the option.
        method : {"newton", "bisection", "brent"}, optional
            Numerical method to use. Default is "brent".
            - "newton": Newton-Raphson (fast but may diverge)
            - "bisection": Bisection (robust but slow)
            - "brent": Brent's method (recommended - fast and robust)
        initial_guess : float, optional
            Initial volatility guess for Newton-Raphson. Default is 0.2.
        tol : float, optional
            Tolerance for convergence. Default is 1e-8.
        max_iter : int, optional
            Maximum iterations. Default is 100.

        Returns
        -------
        float
            Implied volatility.

        Raises
        ------
        ValueError
            If no solution is found (e.g., arbitrage price).

        Examples
        --------
        >>> solver = ImpliedVolSolver(S=100, K=105, T=1, r=0.05)
        >>> solver.solve(market_price=8.02, method="newton")
        0.1999...

        >>> solver.solve(market_price=8.02, method="brent")
        0.1999...
        """
        # Check for valid price bounds
        if self.option_type == "call":
            # Call price bounds: max(S*exp(-qT) - K*exp(-rT), 0) <= C <= S*exp(-qT)
            lower_bound = max(
                self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T),
                0,
            )
            upper_bound = self.S * np.exp(-self.q * self.T)
        else:
            # Put price bounds: max(K*exp(-rT) - S*exp(-qT), 0) <= P <= K*exp(-rT)
            lower_bound = max(
                self.K * np.exp(-self.r * self.T) - self.S * np.exp(-self.q * self.T),
                0,
            )
            upper_bound = self.K * np.exp(-self.r * self.T)

        if market_price < lower_bound or market_price > upper_bound:
            raise ValueError(
                f"Market price {market_price} is outside valid bounds "
                f"[{lower_bound:.4f}, {upper_bound:.4f}]"
            )

        # Objective function: f(sigma) = BS(sigma) - market_price
        def objective(sigma: float) -> float:
            return self._bs_price(sigma) - market_price

        if method == "newton":
            return self._newton_raphson(
                market_price, initial_guess, tol, max_iter
            )
        elif method == "bisection":
            return self._bisection(market_price, tol, max_iter)
        elif method == "brent":
            return self._brent(market_price, tol, max_iter)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _newton_raphson(
        self,
        market_price: float,
        initial_guess: float,
        tol: float,
        max_iter: int,
    ) -> float:
        """Newton-Raphson method for implied volatility."""
        sigma = initial_guess

        for _ in range(max_iter):
            price = self._bs_price(sigma)
            vega = self._bs_vega(sigma)

            if abs(vega) < 1e-12:
                # Vega too small, switch to bisection
                return self._bisection(market_price, tol, max_iter)

            diff = price - market_price

            if abs(diff) < tol:
                return sigma

            sigma = sigma - diff / vega

            # Keep sigma in reasonable bounds
            sigma = max(0.001, min(sigma, 5.0))

        # If not converged, try bisection
        return self._bisection(market_price, tol, max_iter)

    def _bisection(
        self,
        market_price: float,
        tol: float,
        max_iter: int,
    ) -> float:
        """Bisection method for implied volatility."""
        sigma_low = 0.001
        sigma_high = 5.0

        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price_mid = self._bs_price(sigma_mid)

            if abs(price_mid - market_price) < tol:
                return sigma_mid

            if price_mid < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid

        return (sigma_low + sigma_high) / 2

    def _brent(
        self,
        market_price: float,
        tol: float,
        max_iter: int,
    ) -> float:
        """Brent's method using scipy."""

        def objective(sigma: float) -> float:
            return self._bs_price(sigma) - market_price

        result = optimize.brentq(
            objective,
            a=0.001,
            b=5.0,
            xtol=tol,
            maxiter=max_iter,
        )
        return float(result)

    def __repr__(self) -> str:
        return (
            f"ImpliedVolSolver(S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, option_type={self.option_type})"
        )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    method: Literal["newton", "bisection", "brent"] = "brent",
) -> float:
    """
    Calculate implied volatility from market price.

    Convenience function that creates a solver and returns IV directly.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S : float
        Current underlying asset price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate.
    q : float, optional
        Continuous dividend yield. Default is 0.
    option_type : {"call", "put"}, optional
        Type of option. Default is "call".
    method : {"newton", "bisection", "brent"}, optional
        Numerical method. Default is "brent".

    Returns
    -------
    float
        Implied volatility.

    Examples
    --------
    >>> # Find IV that produces price of 8.02
    >>> iv = implied_volatility(
    ...     market_price=8.02,
    ...     S=100, K=105, T=1, r=0.05
    ... )
    >>> np.abs(iv - 0.2) < 0.001
    True

    >>> # Verify: plug IV back into Black-Scholes
    >>> from stochastic_engine import BlackScholes
    >>> bs = BlackScholes(S=100, K=105, T=1, r=0.05, sigma=iv)
    >>> np.abs(bs.price - 8.02) < 0.01
    True
    """
    solver = ImpliedVolSolver(
        S=S, K=K, T=T, r=r, q=q, option_type=option_type
    )
    return solver.solve(market_price=market_price, method=method)


def implied_volatility_surface(
    market_prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    expirations: np.ndarray,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> np.ndarray:
    """
    Calculate implied volatility surface from option prices.

    Parameters
    ----------
    market_prices : np.ndarray
        2D array of market prices, shape (n_strikes, n_expirations).
    S : float
        Current underlying asset price.
    strikes : np.ndarray
        Array of strike prices.
    expirations : np.ndarray
        Array of expiration times in years.
    r : float
        Risk-free interest rate.
    q : float, optional
        Continuous dividend yield. Default is 0.
    option_type : {"call", "put"}, optional
        Type of option. Default is "call".

    Returns
    -------
    np.ndarray
        2D array of implied volatilities, same shape as market_prices.
        NaN for prices that couldn't be inverted.

    Examples
    --------
    >>> strikes = np.array([95, 100, 105])
    >>> expirations = np.array([0.25, 0.5, 1.0])
    >>> prices = np.array([
    ...     [6.5, 8.0, 10.5],   # K=95
    ...     [3.5, 5.5, 8.0],    # K=100
    ...     [1.5, 3.0, 5.5],    # K=105
    ... ])
    >>> iv_surface = implied_volatility_surface(
    ...     market_prices=prices,
    ...     S=100, strikes=strikes, expirations=expirations, r=0.05
    ... )
    >>> iv_surface.shape
    (3, 3)
    """
    market_prices = np.asarray(market_prices)
    strikes = np.asarray(strikes)
    expirations = np.asarray(expirations)

    iv_surface = np.full_like(market_prices, np.nan, dtype=float)

    for i, K in enumerate(strikes):
        for j, T in enumerate(expirations):
            try:
                iv_surface[i, j] = implied_volatility(
                    market_price=market_prices[i, j],
                    S=S,
                    K=K,
                    T=T,
                    r=r,
                    q=q,
                    option_type=option_type,
                )
            except (ValueError, RuntimeError):
                # Keep NaN for invalid prices
                pass

    return iv_surface

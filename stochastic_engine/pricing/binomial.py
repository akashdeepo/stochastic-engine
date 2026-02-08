"""Binomial tree option pricing (Cox-Ross-Rubinstein model)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class BinomialResult:
    """
    Result from binomial tree pricing.

    Attributes
    ----------
    price : float
        Option price.
    delta : float
        Option delta (from first step).
    gamma : float
        Option gamma (from first two steps).
    tree : np.ndarray | None
        Full price tree if requested.
    """

    price: float
    delta: float
    gamma: float
    tree: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"BinomialResult(price={self.price:.4f}, delta={self.delta:.4f}, gamma={self.gamma:.4f})"


class BinomialTree:
    """
    Binomial tree option pricing model (Cox-Ross-Rubinstein).

    Prices European and American options using a recombining binomial tree.
    Converges to Black-Scholes as the number of steps increases.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    q : float, optional
        Continuous dividend yield. Default is 0.
    steps : int, optional
        Number of tree steps. Default is 100.

    Examples
    --------
    >>> tree = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=100)
    >>> result = tree.price(option_type='call', style='european')
    >>> print(f"Price: {result.price:.4f}")
    Price: 10.4506

    >>> # American put (early exercise)
    >>> result = tree.price(option_type='put', style='american')
    >>> print(f"American Put: {result.price:.4f}")

    Notes
    -----
    The CRR binomial model uses:

    .. math::
        u = e^{\\sigma \\sqrt{\\Delta t}}, \\quad d = 1/u

        p = \\frac{e^{(r-q)\\Delta t} - d}{u - d}

    where u is the up factor, d is the down factor, and p is the
    risk-neutral probability of an up move.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        steps: int = 100,
    ) -> None:
        """Initialize binomial tree."""
        if S <= 0:
            raise ValueError(f"Price S must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike K must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Time T must be positive, got {T}")
        if sigma < 0:
            raise ValueError(f"Volatility sigma must be non-negative, got {sigma}")
        if steps < 1:
            raise ValueError(f"Steps must be at least 1, got {steps}")

        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.steps = steps

        # Calculate tree parameters
        self.dt = T / steps
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Up factor
        self.d = 1 / self.u  # Down factor

        # Risk-neutral probability
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)

        # Discount factor
        self.df = np.exp(-r * self.dt)

    def _build_stock_tree(self) -> np.ndarray:
        """Build the stock price tree."""
        n = self.steps + 1
        tree = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1):
                tree[j, i] = self.S * (self.u ** (i - j)) * (self.d ** j)

        return tree

    def _calculate_payoff(
        self,
        stock_tree: np.ndarray,
        option_type: Literal["call", "put"],
    ) -> np.ndarray:
        """Calculate option payoff at expiration."""
        n = self.steps + 1
        payoff = np.zeros(n)

        terminal_prices = stock_tree[:, -1]

        if option_type == "call":
            payoff = np.maximum(terminal_prices - self.K, 0)
        else:  # put
            payoff = np.maximum(self.K - terminal_prices, 0)

        return payoff

    def price(
        self,
        option_type: Literal["call", "put"] = "call",
        style: Literal["european", "american"] = "european",
        return_tree: bool = False,
    ) -> BinomialResult:
        """
        Price an option using the binomial tree.

        Parameters
        ----------
        option_type : {"call", "put"}, optional
            Type of option. Default is "call".
        style : {"european", "american"}, optional
            Exercise style. Default is "european".
        return_tree : bool, optional
            Whether to return the full option price tree. Default is False.

        Returns
        -------
        BinomialResult
            Pricing result with price, delta, gamma, and optionally the tree.

        Examples
        --------
        >>> tree = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=200)
        >>> result = tree.price(option_type='call', style='european')
        >>> result.price
        10.45...

        >>> # American put has early exercise value
        >>> euro_put = tree.price(option_type='put', style='european').price
        >>> amer_put = tree.price(option_type='put', style='american').price
        >>> amer_put >= euro_put
        True
        """
        n = self.steps + 1

        # Build stock tree
        stock_tree = self._build_stock_tree()

        # Initialize option value tree
        option_tree = np.zeros((n, n))

        # Terminal payoff
        option_tree[:, -1] = self._calculate_payoff(stock_tree, option_type)

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Continuation value
                hold_value = self.df * (
                    self.p * option_tree[j, i + 1]
                    + (1 - self.p) * option_tree[j + 1, i + 1]
                )

                if style == "american":
                    # Early exercise value
                    if option_type == "call":
                        exercise_value = max(stock_tree[j, i] - self.K, 0)
                    else:
                        exercise_value = max(self.K - stock_tree[j, i], 0)

                    option_tree[j, i] = max(hold_value, exercise_value)
                else:
                    option_tree[j, i] = hold_value

        # Extract price
        price = option_tree[0, 0]

        # Calculate delta from first step
        delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])

        # Calculate gamma from first two steps
        if self.steps >= 2:
            delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (stock_tree[0, 2] - stock_tree[1, 2])
            delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (stock_tree[1, 2] - stock_tree[2, 2])
            h = 0.5 * (stock_tree[0, 2] - stock_tree[2, 2])
            gamma = (delta_up - delta_down) / h
        else:
            gamma = 0.0

        return BinomialResult(
            price=price,
            delta=delta,
            gamma=gamma,
            tree=option_tree if return_tree else None,
        )

    def implied_volatility(
        self,
        market_price: float,
        option_type: Literal["call", "put"] = "call",
        style: Literal["european", "american"] = "american",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Find implied volatility from market price using binomial tree.

        Useful for American options where Black-Scholes doesn't apply.

        Parameters
        ----------
        market_price : float
            Observed market price.
        option_type : {"call", "put"}, optional
            Option type. Default is "call".
        style : {"european", "american"}, optional
            Exercise style. Default is "american".
        tol : float, optional
            Tolerance for convergence. Default is 1e-6.
        max_iter : int, optional
            Maximum iterations. Default is 100.

        Returns
        -------
        float
            Implied volatility.

        Examples
        --------
        >>> tree = BinomialTree(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=100)
        >>> price = tree.price(option_type='put', style='american').price
        >>> tree.implied_volatility(market_price=price, option_type='put')
        0.2...
        """
        sigma_low = 0.001
        sigma_high = 5.0

        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2

            # Create tree with mid volatility
            mid_tree = BinomialTree(
                S=self.S, K=self.K, T=self.T, r=self.r,
                sigma=sigma_mid, q=self.q, steps=self.steps
            )
            mid_price = mid_tree.price(option_type=option_type, style=style).price

            if abs(mid_price - market_price) < tol:
                return sigma_mid

            if mid_price < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid

        return (sigma_low + sigma_high) / 2

    def __repr__(self) -> str:
        return (
            f"BinomialTree(S={self.S}, K={self.K}, T={self.T}, "
            f"r={self.r}, sigma={self.sigma}, steps={self.steps})"
        )


def american_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    steps: int = 200,
) -> float:
    """
    Price an American put option.

    Convenience function for quick American put pricing.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    q : float, optional
        Dividend yield. Default is 0.
    steps : int, optional
        Tree steps. Default is 200.

    Returns
    -------
    float
        American put price.

    Examples
    --------
    >>> american_put(S=100, K=100, T=1, r=0.05, sigma=0.2)
    6.09...
    """
    tree = BinomialTree(S=S, K=K, T=T, r=r, sigma=sigma, q=q, steps=steps)
    return tree.price(option_type="put", style="american").price


def american_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    steps: int = 200,
) -> float:
    """
    Price an American call option.

    Note: American calls on non-dividend paying stocks equal European calls.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    q : float, optional
        Dividend yield. Default is 0.
    steps : int, optional
        Tree steps. Default is 200.

    Returns
    -------
    float
        American call price.
    """
    tree = BinomialTree(S=S, K=K, T=T, r=r, sigma=sigma, q=q, steps=steps)
    return tree.price(option_type="call", style="american").price

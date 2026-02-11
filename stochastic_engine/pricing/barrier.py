"""Barrier option pricing (Reiner-Rubinstein closed-form + Monte Carlo)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats


@dataclass
class BarrierResult:
    """
    Result from barrier option pricing.

    Attributes
    ----------
    price : float
        Option price.
    barrier_type : str
        Barrier type (e.g., "down-and-out").
    option_type : str
        "call" or "put".
    method : str
        "closed_form" or "monte_carlo".
    std_error : float or None
        Standard error (MC only).
    confidence_interval : tuple[float, float] or None
        95% CI (MC only).
    """

    price: float
    barrier_type: str
    option_type: str
    method: str
    std_error: float | None = None
    confidence_interval: tuple[float, float] | None = None

    def __repr__(self) -> str:
        ci_str = ""
        if self.std_error is not None:
            ci_str = f", std_error={self.std_error:.4f}"
        return (
            f"BarrierResult(price={self.price:.4f}, "
            f"{self.barrier_type} {self.option_type}{ci_str})"
        )


class BarrierOption:
    """
    Barrier option pricer.

    Supports all 8 combinations of (up/down, in/out, call/put)
    using Reiner-Rubinstein (1991) closed-form formulas and Monte Carlo.

    Parameters
    ----------
    S : float
        Current asset price.
    K : float
        Strike price.
    B : float
        Barrier level.
    T : float
        Time to expiration in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    q : float, optional
        Continuous dividend yield. Default is 0.
    option_type : {"call", "put"}, optional
        Option type. Default is "call".
    barrier_type : {"up-and-in", "up-and-out", "down-and-in", "down-and-out"}, optional
        Barrier type. Default is "down-and-out".
    rebate : float, optional
        Rebate paid when barrier is hit (out) or not hit (in). Default is 0.

    Notes
    -----
    In/out parity: ``V_in + V_out = V_vanilla``

    This identity provides a useful cross-check.

    Examples
    --------
    >>> opt = BarrierOption(S=100, K=105, B=90, T=1, r=0.05, sigma=0.2,
    ...                     barrier_type="down-and-out")
    >>> opt.price()
    BarrierResult(price=5.48..., down-and-out call)
    """

    def __init__(
        self,
        S: float,
        K: float,
        B: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: Literal["call", "put"] = "call",
        barrier_type: Literal[
            "up-and-in", "up-and-out", "down-and-in", "down-and-out"
        ] = "down-and-out",
        rebate: float = 0.0,
    ) -> None:
        """Initialize BarrierOption."""
        self.S = float(S)
        self.K = float(K)
        self.B = float(B)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.rebate = float(rebate)

        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")
        if barrier_type not in ("up-and-in", "up-and-out", "down-and-in", "down-and-out"):
            raise ValueError(f"Invalid barrier_type: {barrier_type}")

        # Precompute parameters
        self._mu = (self.r - self.q - 0.5 * self.sigma**2) / self.sigma**2
        self._lam = np.sqrt(self._mu**2 + 2 * self.r / self.sigma**2)
        self._sqrt_T = np.sqrt(self.T)

    def price(
        self,
        method: Literal["closed_form", "monte_carlo"] = "closed_form",
        **kwargs,
    ) -> BarrierResult:
        """
        Price the barrier option.

        Parameters
        ----------
        method : {"closed_form", "monte_carlo"}, optional
            Pricing method. Default is "closed_form".
        **kwargs
            Additional arguments for MC: n_paths, steps, seed.

        Returns
        -------
        BarrierResult
            Pricing result.
        """
        if method == "closed_form":
            p = self._price_closed_form()
            return BarrierResult(
                price=p,
                barrier_type=self.barrier_type,
                option_type=self.option_type,
                method="closed_form",
            )
        elif method == "monte_carlo":
            return self._price_monte_carlo(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _price_closed_form(self) -> float:
        """Price using Reiner-Rubinstein analytical formulas."""
        S, K, B, T, r, q, sigma = (
            self.S, self.K, self.B, self.T, self.r, self.q, self.sigma,
        )

        # Vanilla price for in/out parity
        from stochastic_engine.pricing.black_scholes import BlackScholes

        vanilla = BlackScholes(
            S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=self.option_type
        ).price

        is_call = self.option_type == "call"
        phi = 1 if is_call else -1  # +1 for call, -1 for put

        # Helper functions
        def _N(x):
            return stats.norm.cdf(x)

        mu = (r - q) / sigma**2 - 0.5
        lam = np.sqrt(mu**2 + 2 * r / sigma**2)
        sqrt_T = np.sqrt(T)

        x1 = np.log(S / K) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        x2 = np.log(S / B) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        y1 = np.log(B**2 / (S * K)) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        y2 = np.log(B / S) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        z = np.log(B / S) / (sigma * sqrt_T) + lam * sigma * sqrt_T

        df = np.exp(-r * T)
        qf = np.exp(-q * T)

        def _A(phi):
            return (
                phi * S * qf * _N(phi * x1)
                - phi * K * df * _N(phi * x1 - phi * sigma * sqrt_T)
            )

        def _B_func(phi):
            return (
                phi * S * qf * _N(phi * x2)
                - phi * K * df * _N(phi * x2 - phi * sigma * sqrt_T)
            )

        def _C(phi, eta):
            return (
                phi * S * qf * (B / S) ** (2 * (mu + 1)) * _N(eta * y1)
                - phi * K * df * (B / S) ** (2 * mu) * _N(eta * y1 - eta * sigma * sqrt_T)
            )

        def _D(phi, eta):
            return (
                phi * S * qf * (B / S) ** (2 * (mu + 1)) * _N(eta * y2)
                - phi * K * df * (B / S) ** (2 * mu) * _N(eta * y2 - eta * sigma * sqrt_T)
            )

        def _E(eta):
            return self.rebate * df * (
                _N(eta * x2 - eta * sigma * sqrt_T)
                - (B / S) ** (2 * mu) * _N(eta * y2 - eta * sigma * sqrt_T)
            )

        def _F(eta):
            return self.rebate * (
                (B / S) ** (mu + lam) * _N(eta * z)
                + (B / S) ** (mu - lam) * _N(eta * z - 2 * eta * lam * sigma * sqrt_T)
            )

        # Dispatch based on barrier type and option type
        if self.barrier_type == "down-and-in":
            eta = 1
            if is_call:
                if K > B:
                    price = _C(1, eta) + _E(eta)
                else:
                    price = _A(1) - _B_func(1) + _D(1, eta) + _E(eta)
            else:
                if K > B:
                    price = _A(-1) - _B_func(-1) + _D(-1, eta) + _E(eta)
                else:
                    price = _C(-1, eta) + _E(eta)

        elif self.barrier_type == "down-and-out":
            eta = 1
            # Use in/out parity: out = vanilla - in
            if is_call:
                if K > B:
                    in_price = _C(1, eta) + _E(eta)
                else:
                    in_price = _A(1) - _B_func(1) + _D(1, eta) + _E(eta)
            else:
                if K > B:
                    in_price = _A(-1) - _B_func(-1) + _D(-1, eta) + _E(eta)
                else:
                    in_price = _C(-1, eta) + _E(eta)
            price = vanilla - in_price + _F(eta)

        elif self.barrier_type == "up-and-in":
            eta = -1
            if is_call:
                if K > B:
                    price = _A(1) + _E(eta)
                else:
                    price = _B_func(1) - _C(1, eta) + _D(1, eta) + _E(eta)
            else:
                if K > B:
                    price = _B_func(-1) - _C(-1, eta) + _D(-1, eta) + _E(eta)
                else:
                    price = _A(-1) + _E(eta)

        elif self.barrier_type == "up-and-out":
            eta = -1
            if is_call:
                if K > B:
                    in_price = _A(1) + _E(eta)
                else:
                    in_price = _B_func(1) - _C(1, eta) + _D(1, eta) + _E(eta)
            else:
                if K > B:
                    in_price = _B_func(-1) - _C(-1, eta) + _D(-1, eta) + _E(eta)
                else:
                    in_price = _A(-1) + _E(eta)
            price = vanilla - in_price + _F(eta)

        return max(float(price), 0.0)

    def _price_monte_carlo(
        self,
        n_paths: int = 100000,
        steps: int = 252,
        seed: int | None = None,
    ) -> BarrierResult:
        """
        Price using Monte Carlo with barrier monitoring.

        Parameters
        ----------
        n_paths : int, optional
            Number of paths. Default is 100000.
        steps : int, optional
            Number of time steps. Default is 252.
        seed : int or None, optional
            Random seed.

        Returns
        -------
        BarrierResult
            MC pricing result.
        """
        rng = np.random.default_rng(seed)
        dt = self.T / steps
        sqrt_dt = np.sqrt(dt)

        # Simulate GBM paths
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.S

        for i in range(steps):
            dW = sqrt_dt * rng.standard_normal(n_paths)
            paths[:, i + 1] = paths[:, i] * np.exp(drift + self.sigma * dW)

        # Determine barrier hits
        if "down" in self.barrier_type:
            barrier_hit = np.any(paths[:, 1:] <= self.B, axis=1)
        else:  # up
            barrier_hit = np.any(paths[:, 1:] >= self.B, axis=1)

        # Compute payoffs
        terminal = paths[:, -1]
        if self.option_type == "call":
            intrinsic = np.maximum(terminal - self.K, 0)
        else:
            intrinsic = np.maximum(self.K - terminal, 0)

        if "out" in self.barrier_type:
            payoffs = np.where(barrier_hit, self.rebate, intrinsic)
        else:  # in
            payoffs = np.where(barrier_hit, intrinsic, self.rebate)

        # Discount
        discount_factor = np.exp(-self.r * self.T)
        discounted = payoffs * discount_factor

        price = discounted.mean()
        std_error = discounted.std() / np.sqrt(n_paths)
        ci = (price - 1.96 * std_error, price + 1.96 * std_error)

        return BarrierResult(
            price=float(price),
            barrier_type=self.barrier_type,
            option_type=self.option_type,
            method="monte_carlo",
            std_error=float(std_error),
            confidence_interval=ci,
        )

    def __repr__(self) -> str:
        return (
            f"BarrierOption({self.barrier_type} {self.option_type}, "
            f"S={self.S}, K={self.K}, B={self.B})"
        )


# Convenience functions

def barrier_call(
    S: float, K: float, B: float, T: float, r: float, sigma: float,
    barrier_type: str = "down-and-out", q: float = 0.0, rebate: float = 0.0,
) -> float:
    """
    Price a barrier call option (closed-form).

    Parameters
    ----------
    S, K, B, T, r, sigma : float
        Standard barrier option parameters.
    barrier_type : str, optional
        Barrier type. Default is "down-and-out".
    q : float, optional
        Dividend yield. Default is 0.
    rebate : float, optional
        Rebate amount. Default is 0.

    Returns
    -------
    float
        Option price.
    """
    opt = BarrierOption(S, K, B, T, r, sigma, q, "call", barrier_type, rebate)
    return opt.price().price


def barrier_put(
    S: float, K: float, B: float, T: float, r: float, sigma: float,
    barrier_type: str = "down-and-out", q: float = 0.0, rebate: float = 0.0,
) -> float:
    """
    Price a barrier put option (closed-form).

    Parameters
    ----------
    S, K, B, T, r, sigma : float
        Standard barrier option parameters.
    barrier_type : str, optional
        Barrier type. Default is "down-and-out".
    q : float, optional
        Dividend yield. Default is 0.
    rebate : float, optional
        Rebate amount. Default is 0.

    Returns
    -------
    float
        Option price.
    """
    opt = BarrierOption(S, K, B, T, r, sigma, q, "put", barrier_type, rebate)
    return opt.price().price

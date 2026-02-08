"""Conditional Value at Risk (CVaR / Expected Shortfall) calculations."""

from typing import Literal

import numpy as np
from scipy import stats


class CVaR:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall calculator.

    CVaR measures the expected loss given that the loss exceeds VaR.
    It is also known as Expected Shortfall (ES), Average Value at Risk (AVaR),
    or Expected Tail Loss (ETL).

    Unlike VaR, CVaR is a coherent risk measure (subadditive), making it
    better for portfolio risk aggregation.

    Parameters
    ----------
    returns : np.ndarray
        Array of historical returns (as decimals, e.g., 0.01 for 1%).
    confidence : float, optional
        Confidence level (e.g., 0.95 for 95%). Default is 0.95.

    Examples
    --------
    >>> returns = np.array([-0.05, -0.03, -0.02, 0.01, 0.02, 0.03, -0.04])
    >>> cvar = CVaR(returns, confidence=0.95)
    >>> cvar.historical()
    0.05

    >>> # CVaR is always >= VaR
    >>> from stochastic_engine.risk import VaR
    >>> var = VaR(returns, confidence=0.95).historical()
    >>> cvar = CVaR(returns, confidence=0.95).historical()
    >>> cvar >= var
    True

    Notes
    -----
    CVaR answers: "Given that losses exceed VaR, what is the expected loss?"

    Advantages over VaR:
    - Captures tail risk
    - Coherent risk measure (subadditive)
    - Recommended by Basel III for market risk

    CVaR formula:
    .. math::
        CVaR_\\alpha = E[L | L > VaR_\\alpha]
    """

    def __init__(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> None:
        """
        Initialize CVaR calculator.

        Parameters
        ----------
        returns : np.ndarray
            Array of historical returns.
        confidence : float, optional
            Confidence level (0 < confidence < 1). Default is 0.95.

        Raises
        ------
        ValueError
            If confidence is not between 0 and 1, or if returns is empty.
        """
        self.returns = np.asarray(returns).flatten()
        self.confidence = confidence

        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        if len(self.returns) == 0:
            raise ValueError("Returns array cannot be empty")

    def historical(self) -> float:
        """
        Calculate Historical CVaR (Expected Shortfall).

        Computes the average of returns below the VaR threshold.

        Returns
        -------
        float
            CVaR as a positive number (expected loss in tail).

        Examples
        --------
        >>> returns = np.random.randn(1000) * 0.02
        >>> cvar = CVaR(returns, confidence=0.95)
        >>> cvar.historical() > 0
        True
        """
        alpha = 1 - self.confidence
        var_threshold = np.percentile(self.returns, alpha * 100)

        # Average of returns below VaR (tail losses)
        tail_returns = self.returns[self.returns <= var_threshold]

        if len(tail_returns) == 0:
            # Edge case: no returns below threshold
            return float(-var_threshold)

        return float(-tail_returns.mean())

    def parametric(self, distribution: Literal["normal", "t"] = "normal") -> float:
        """
        Calculate Parametric CVaR.

        Uses analytical formula based on distributional assumption.

        Parameters
        ----------
        distribution : {"normal", "t"}, optional
            Distribution assumption. Default is "normal".

        Returns
        -------
        float
            CVaR as a positive number (expected loss in tail).

        Examples
        --------
        >>> returns = np.random.randn(1000) * 0.02
        >>> cvar = CVaR(returns, confidence=0.95)
        >>> cvar.parametric()  # Normal assumption
        ...
        """
        mu = self.returns.mean()
        sigma = self.returns.std()
        alpha = 1 - self.confidence

        if distribution == "normal":
            # ES = mu - sigma * phi(z_alpha) / alpha
            # where phi is the standard normal PDF
            z_alpha = stats.norm.ppf(alpha)
            phi_z = stats.norm.pdf(z_alpha)
            es = -mu + sigma * phi_z / alpha
            return float(es)

        elif distribution == "t":
            # Fit t-distribution
            df, loc, scale = stats.t.fit(self.returns)
            t_alpha = stats.t.ppf(alpha, df)
            pdf_t = stats.t.pdf(t_alpha, df)

            # ES for t-distribution
            es = (
                -loc
                + scale
                * (df + t_alpha**2)
                / (df - 1)
                * pdf_t
                / alpha
            )
            return float(es)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def monte_carlo(
        self,
        n_simulations: int = 10000,
        horizon: int = 1,
        seed: int | None = None,
    ) -> float:
        """
        Calculate Monte Carlo CVaR.

        Simulates future returns and computes expected tail loss.

        Parameters
        ----------
        n_simulations : int, optional
            Number of Monte Carlo simulations. Default is 10000.
        horizon : int, optional
            Number of periods ahead. Default is 1.
        seed : int | None, optional
            Random seed for reproducibility.

        Returns
        -------
        float
            CVaR as a positive number (expected loss in tail).
        """
        rng = np.random.default_rng(seed)

        mu = self.returns.mean()
        sigma = self.returns.std()

        # Simulate returns
        if horizon == 1:
            simulated = rng.normal(mu, sigma, n_simulations)
        else:
            simulated = rng.normal(mu * horizon, sigma * np.sqrt(horizon), n_simulations)

        # Calculate CVaR from simulated returns
        alpha = 1 - self.confidence
        var_threshold = np.percentile(simulated, alpha * 100)
        tail_returns = simulated[simulated <= var_threshold]

        return float(-tail_returns.mean())

    def scale_to_horizon(self, cvar_1d: float, horizon: int) -> float:
        """
        Scale 1-day CVaR to a different time horizon.

        Uses the square root of time rule.

        Parameters
        ----------
        cvar_1d : float
            1-day CVaR.
        horizon : int
            Target horizon in days.

        Returns
        -------
        float
            Scaled CVaR.
        """
        return cvar_1d * np.sqrt(horizon)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"CVaR(n={len(self.returns)}, confidence={self.confidence})"


# Alias for expected shortfall
ExpectedShortfall = CVaR

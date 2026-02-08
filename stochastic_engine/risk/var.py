"""Value at Risk (VaR) calculations."""

from typing import Literal

import numpy as np
from scipy import stats


class VaR:
    """
    Value at Risk (VaR) calculator.

    VaR estimates the maximum potential loss over a specified time period
    at a given confidence level. It answers: "What is the worst loss I
    can expect with X% confidence over the next N days?"

    Parameters
    ----------
    returns : np.ndarray
        Array of historical returns (as decimals, e.g., 0.01 for 1%).
    confidence : float, optional
        Confidence level (e.g., 0.95 for 95%). Default is 0.95.

    Examples
    --------
    >>> returns = np.array([-0.02, 0.01, -0.03, 0.02, -0.01, 0.015, -0.025])
    >>> var = VaR(returns, confidence=0.95)
    >>> var.historical()  # Historical VaR
    0.029...

    >>> var.parametric()  # Parametric (variance-covariance) VaR
    0.026...

    Notes
    -----
    VaR limitations:
    - Does not measure tail risk beyond the VaR threshold
    - Assumes historical patterns continue
    - Not subadditive (portfolio VaR can exceed sum of individual VaRs)

    For tail risk, use CVaR (Expected Shortfall) instead.
    """

    def __init__(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> None:
        """
        Initialize VaR calculator.

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
        Calculate Historical Simulation VaR.

        Uses the empirical distribution of historical returns.
        No distributional assumptions required.

        Returns
        -------
        float
            VaR as a positive number (loss).

        Examples
        --------
        >>> returns = np.random.randn(1000) * 0.02
        >>> var = VaR(returns, confidence=0.95)
        >>> var.historical() > 0
        True
        """
        # VaR is the negative of the percentile (to express as loss)
        alpha = 1 - self.confidence
        return float(-np.percentile(self.returns, alpha * 100))

    def parametric(self, distribution: Literal["normal", "t"] = "normal") -> float:
        """
        Calculate Parametric (Variance-Covariance) VaR.

        Assumes returns follow a specified distribution.

        Parameters
        ----------
        distribution : {"normal", "t"}, optional
            Distribution assumption. Default is "normal".
            - "normal": Normal distribution
            - "t": Student's t-distribution (heavier tails)

        Returns
        -------
        float
            VaR as a positive number (loss).

        Examples
        --------
        >>> returns = np.random.randn(1000) * 0.02
        >>> var = VaR(returns, confidence=0.95)
        >>> var.parametric()  # Normal assumption
        ...
        >>> var.parametric(distribution="t")  # t-distribution
        ...
        """
        mu = self.returns.mean()
        sigma = self.returns.std()
        alpha = 1 - self.confidence

        if distribution == "normal":
            z = stats.norm.ppf(alpha)
            return float(-(mu + sigma * z))

        elif distribution == "t":
            # Fit t-distribution to get degrees of freedom
            df, loc, scale = stats.t.fit(self.returns)
            t_val = stats.t.ppf(alpha, df)
            return float(-(loc + scale * t_val))

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def monte_carlo(
        self,
        n_simulations: int = 10000,
        horizon: int = 1,
        seed: int | None = None,
    ) -> float:
        """
        Calculate Monte Carlo VaR.

        Simulates future returns based on historical distribution parameters.

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
            VaR as a positive number (loss).

        Examples
        --------
        >>> returns = np.random.randn(1000) * 0.02
        >>> var = VaR(returns, confidence=0.95)
        >>> var.monte_carlo(n_simulations=10000, seed=42)
        ...
        """
        rng = np.random.default_rng(seed)

        mu = self.returns.mean()
        sigma = self.returns.std()

        # Simulate returns over horizon
        if horizon == 1:
            simulated = rng.normal(mu, sigma, n_simulations)
        else:
            # Sum of returns over horizon (assuming independence)
            simulated = rng.normal(mu * horizon, sigma * np.sqrt(horizon), n_simulations)

        alpha = 1 - self.confidence
        return float(-np.percentile(simulated, alpha * 100))

    def cornish_fisher(self) -> float:
        """
        Calculate Cornish-Fisher VaR.

        Adjusts the normal VaR for skewness and kurtosis using
        the Cornish-Fisher expansion.

        Returns
        -------
        float
            VaR as a positive number (loss).

        Notes
        -----
        The Cornish-Fisher expansion provides a better VaR estimate
        when returns exhibit significant skewness or excess kurtosis.
        """
        mu = self.returns.mean()
        sigma = self.returns.std()
        skew = stats.skew(self.returns)
        kurt = stats.kurtosis(self.returns)  # Excess kurtosis

        alpha = 1 - self.confidence
        z = stats.norm.ppf(alpha)

        # Cornish-Fisher adjustment
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )

        return float(-(mu + sigma * z_cf))

    def scale_to_horizon(self, var_1d: float, horizon: int) -> float:
        """
        Scale 1-day VaR to a different time horizon.

        Uses the square root of time rule (assumes IID returns).

        Parameters
        ----------
        var_1d : float
            1-day VaR.
        horizon : int
            Target horizon in days.

        Returns
        -------
        float
            Scaled VaR.

        Examples
        --------
        >>> var = VaR(returns, confidence=0.95)
        >>> var_1d = var.historical()
        >>> var_10d = var.scale_to_horizon(var_1d, horizon=10)
        """
        return var_1d * np.sqrt(horizon)

    def backtest(
        self,
        actual_returns: np.ndarray | None = None,
        window: int = 250,
    ) -> dict:
        """
        Backtest VaR model using historical data.

        Parameters
        ----------
        actual_returns : np.ndarray | None, optional
            Returns to test against. If None, uses self.returns.
        window : int, optional
            Rolling window size for VaR calculation. Default is 250.

        Returns
        -------
        dict
            Backtest results including:
            - violations: Number of times actual loss exceeded VaR
            - violation_rate: Actual violation rate
            - expected_rate: Expected violation rate (1 - confidence)
            - kupiec_pvalue: p-value from Kupiec's POF test

        Examples
        --------
        >>> var = VaR(returns, confidence=0.95)
        >>> results = var.backtest()
        >>> results['violation_rate']
        """
        if actual_returns is None:
            actual_returns = self.returns

        actual_returns = np.asarray(actual_returns)
        n = len(actual_returns)

        if n <= window:
            raise ValueError(f"Need more data than window size ({window})")

        violations = 0
        var_estimates = []

        for i in range(window, n):
            # Calculate VaR using rolling window
            window_returns = actual_returns[i - window : i]
            var_calc = VaR(window_returns, confidence=self.confidence)
            var_est = var_calc.historical()
            var_estimates.append(var_est)

            # Check if actual loss exceeded VaR
            actual_loss = -actual_returns[i]
            if actual_loss > var_est:
                violations += 1

        n_tests = n - window
        violation_rate = violations / n_tests
        expected_rate = 1 - self.confidence

        # Kupiec's Proportion of Failures (POF) test
        if violations > 0 and violations < n_tests:
            lr_stat = -2 * (
                violations * np.log(expected_rate / violation_rate)
                + (n_tests - violations) * np.log((1 - expected_rate) / (1 - violation_rate))
            )
            kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
        else:
            kupiec_pvalue = np.nan

        return {
            "violations": violations,
            "n_tests": n_tests,
            "violation_rate": violation_rate,
            "expected_rate": expected_rate,
            "kupiec_pvalue": kupiec_pvalue,
            "var_estimates": np.array(var_estimates),
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return f"VaR(n={len(self.returns)}, confidence={self.confidence})"

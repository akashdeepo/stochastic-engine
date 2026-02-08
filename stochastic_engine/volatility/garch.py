"""GARCH volatility models."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize


@dataclass
class GARCHResult:
    """
    Result from GARCH model fitting.

    Attributes
    ----------
    omega : float
        Constant term in variance equation.
    alpha : float
        ARCH parameter (reaction to shocks).
    beta : float
        GARCH parameter (persistence).
    long_run_variance : float
        Unconditional (long-run) variance.
    log_likelihood : float
        Log-likelihood of the fitted model.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    """

    omega: float
    alpha: float
    beta: float
    long_run_variance: float
    log_likelihood: float
    aic: float
    bic: float

    @property
    def long_run_volatility(self) -> float:
        """Long-run volatility (annualized, assuming daily data)."""
        return np.sqrt(self.long_run_variance * 252)

    @property
    def persistence(self) -> float:
        """Persistence of volatility shocks (alpha + beta)."""
        return self.alpha + self.beta

    def __repr__(self) -> str:
        return (
            f"GARCHResult(omega={self.omega:.6f}, alpha={self.alpha:.4f}, "
            f"beta={self.beta:.4f}, persistence={self.persistence:.4f})"
        )


class GARCH:
    """
    GARCH(1,1) volatility model.

    Generalized Autoregressive Conditional Heteroskedasticity model
    for volatility forecasting.

    The GARCH(1,1) model specifies:

    .. math::
        \\sigma^2_t = \\omega + \\alpha \\epsilon^2_{t-1} + \\beta \\sigma^2_{t-1}

    where ε_t = r_t - μ are the return innovations.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns (as decimals).

    Attributes
    ----------
    returns : np.ndarray
        The return series.
    n : int
        Number of observations.
    fitted : GARCHResult | None
        Fitted model parameters (after calling fit()).

    Examples
    --------
    >>> returns = np.random.normal(0, 0.01, 1000)
    >>> garch = GARCH(returns)
    >>> result = garch.fit()
    >>> print(f"Persistence: {result.persistence:.4f}")

    >>> # Forecast volatility
    >>> forecast = garch.forecast(horizon=5)
    >>> print(f"5-day ahead vol: {forecast[-1]:.4f}")

    Notes
    -----
    GARCH(1,1) is the most widely used volatility model because:
    - Captures volatility clustering
    - Parsimonious (only 3 parameters)
    - Works well in practice

    For stationarity, we need α + β < 1.
    """

    def __init__(self, returns: np.ndarray) -> None:
        """Initialize GARCH model."""
        self.returns = np.asarray(returns).flatten()
        self.n = len(self.returns)
        self.fitted: GARCHResult | None = None

        if self.n < 10:
            raise ValueError(f"Need at least 10 observations, got {self.n}")

    def _compute_variance_series(
        self,
        omega: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Compute conditional variance series given parameters."""
        sigma2 = np.zeros(self.n)

        # Initialize with unconditional variance
        if alpha + beta < 1:
            sigma2[0] = omega / (1 - alpha - beta)
        else:
            sigma2[0] = self.returns.var()

        # Demean returns
        mu = self.returns.mean()
        eps = self.returns - mu

        for t in range(1, self.n):
            sigma2[t] = omega + alpha * eps[t - 1]**2 + beta * sigma2[t - 1]

        return sigma2

    def _neg_log_likelihood(self, params: np.ndarray) -> float:
        """Negative log-likelihood for optimization."""
        omega, alpha, beta = params

        # Constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        sigma2 = self._compute_variance_series(omega, alpha, beta)

        # Ensure positive variance
        sigma2 = np.maximum(sigma2, 1e-10)

        # Log-likelihood (Gaussian)
        mu = self.returns.mean()
        eps = self.returns - mu
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)

        return -ll

    def fit(
        self,
        method: Literal["mle", "variance_targeting"] = "mle",
    ) -> GARCHResult:
        """
        Fit the GARCH(1,1) model to the return series.

        Parameters
        ----------
        method : {"mle", "variance_targeting"}, optional
            Estimation method. Default is "mle".
            - "mle": Maximum likelihood estimation
            - "variance_targeting": Fix omega to match unconditional variance

        Returns
        -------
        GARCHResult
            Fitted model parameters.

        Examples
        --------
        >>> garch = GARCH(returns)
        >>> result = garch.fit()
        >>> print(result)
        """
        # Initial guesses
        sample_var = self.returns.var()

        if method == "mle":
            # Initial parameters
            omega0 = sample_var * 0.05
            alpha0 = 0.1
            beta0 = 0.85

            # Bounds
            bounds = [
                (1e-10, sample_var),  # omega
                (0, 0.5),              # alpha
                (0, 0.999),            # beta
            ]

            # Optimize
            result = optimize.minimize(
                self._neg_log_likelihood,
                x0=[omega0, alpha0, beta0],
                method='L-BFGS-B',
                bounds=bounds,
            )

            omega, alpha, beta = result.x
            neg_ll = result.fun

        elif method == "variance_targeting":
            # Fix omega based on unconditional variance
            def neg_ll_targeting(params):
                alpha, beta = params
                if alpha + beta >= 1 or alpha < 0 or beta < 0:
                    return 1e10
                omega = sample_var * (1 - alpha - beta)
                return self._neg_log_likelihood([omega, alpha, beta])

            result = optimize.minimize(
                neg_ll_targeting,
                x0=[0.1, 0.85],
                method='L-BFGS-B',
                bounds=[(0, 0.5), (0, 0.999)],
            )

            alpha, beta = result.x
            omega = sample_var * (1 - alpha - beta)
            neg_ll = result.fun

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate statistics
        log_likelihood = -neg_ll
        k = 3  # Number of parameters
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(self.n) - 2 * log_likelihood

        if alpha + beta < 1:
            long_run_var = omega / (1 - alpha - beta)
        else:
            long_run_var = sample_var

        self.fitted = GARCHResult(
            omega=omega,
            alpha=alpha,
            beta=beta,
            long_run_variance=long_run_var,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
        )

        return self.fitted

    def forecast(
        self,
        horizon: int = 1,
        annualize: bool = False,
    ) -> np.ndarray:
        """
        Forecast future volatility.

        Parameters
        ----------
        horizon : int, optional
            Number of periods to forecast. Default is 1.
        annualize : bool, optional
            Whether to annualize (assumes daily data). Default is False.

        Returns
        -------
        np.ndarray
            Forecasted volatility for each period.

        Examples
        --------
        >>> garch = GARCH(returns)
        >>> garch.fit()
        >>> vol_forecast = garch.forecast(horizon=10)
        >>> vol_forecast.shape
        (10,)
        """
        if self.fitted is None:
            raise ValueError("Model must be fitted first. Call fit().")

        omega = self.fitted.omega
        alpha = self.fitted.alpha
        beta = self.fitted.beta

        # Get current variance
        sigma2 = self._compute_variance_series(omega, alpha, beta)
        current_var = sigma2[-1]

        # Last shock
        mu = self.returns.mean()
        last_shock = (self.returns[-1] - mu)**2

        # Forecast
        forecasts = np.zeros(horizon)
        persistence = alpha + beta
        long_run_var = self.fitted.long_run_variance

        # h-step ahead forecast
        # E[σ²_{t+h}] = ω(1 + ... + (α+β)^{h-1}) + (α+β)^h * σ²_t
        for h in range(1, horizon + 1):
            if h == 1:
                forecasts[h - 1] = omega + alpha * last_shock + beta * current_var
            else:
                forecasts[h - 1] = long_run_var + (persistence ** h) * (current_var - long_run_var)

        # Convert to volatility
        vol_forecast = np.sqrt(forecasts)

        if annualize:
            vol_forecast = vol_forecast * np.sqrt(252)

        return vol_forecast

    def conditional_volatility(self, annualize: bool = False) -> np.ndarray:
        """
        Get the fitted conditional volatility series.

        Parameters
        ----------
        annualize : bool, optional
            Whether to annualize. Default is False.

        Returns
        -------
        np.ndarray
            Conditional volatility series.
        """
        if self.fitted is None:
            raise ValueError("Model must be fitted first. Call fit().")

        sigma2 = self._compute_variance_series(
            self.fitted.omega,
            self.fitted.alpha,
            self.fitted.beta,
        )
        vol = np.sqrt(sigma2)

        if annualize:
            vol = vol * np.sqrt(252)

        return vol

    def simulate(
        self,
        n_periods: int,
        n_paths: int = 1,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate future returns using GARCH dynamics.

        Parameters
        ----------
        n_periods : int
            Number of periods to simulate.
        n_paths : int, optional
            Number of simulation paths. Default is 1.
        seed : int | None, optional
            Random seed.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (returns, volatilities), each of shape (n_paths, n_periods).
        """
        if self.fitted is None:
            raise ValueError("Model must be fitted first. Call fit().")

        rng = np.random.default_rng(seed)

        omega = self.fitted.omega
        alpha = self.fitted.alpha
        beta = self.fitted.beta
        mu = self.returns.mean()

        # Initial conditions
        sigma2 = self._compute_variance_series(omega, alpha, beta)
        current_var = sigma2[-1]
        last_shock = (self.returns[-1] - mu)**2

        # Simulate
        returns_sim = np.zeros((n_paths, n_periods))
        vol_sim = np.zeros((n_paths, n_periods))

        for i in range(n_paths):
            var_t = current_var
            shock_t = last_shock

            for t in range(n_periods):
                # Update variance
                var_t = omega + alpha * shock_t + beta * var_t
                vol_sim[i, t] = np.sqrt(var_t)

                # Generate return
                z = rng.standard_normal()
                returns_sim[i, t] = mu + np.sqrt(var_t) * z
                shock_t = (returns_sim[i, t] - mu)**2

        return returns_sim, vol_sim

    def __repr__(self) -> str:
        if self.fitted:
            return f"GARCH(1,1) fitted: α={self.fitted.alpha:.4f}, β={self.fitted.beta:.4f}"
        return f"GARCH(1,1) unfitted, n={self.n}"

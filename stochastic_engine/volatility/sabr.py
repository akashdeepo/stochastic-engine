"""SABR stochastic volatility model."""

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class SABRResult:
    """
    Result from SABR model calibration.

    Attributes
    ----------
    alpha : float
        Initial volatility level.
    beta : float
        CEV exponent.
    rho : float
        Correlation between forward and volatility.
    nu : float
        Vol of vol.
    rmse : float
        Root mean squared error of calibration.
    max_error : float
        Maximum absolute calibration error.
    """

    alpha: float
    beta: float
    rho: float
    nu: float
    rmse: float
    max_error: float

    def __repr__(self) -> str:
        return (
            f"SABRResult(alpha={self.alpha:.4f}, beta={self.beta:.2f}, "
            f"rho={self.rho:.4f}, nu={self.nu:.4f}, rmse={self.rmse:.6f})"
        )


class SABR:
    """
    SABR stochastic volatility model.

    The industry-standard model for the volatility smile in interest rate
    and FX options markets.

    .. math::
        dF = \\alpha F^\\beta \\, dW_1

        d\\alpha = \\nu \\alpha \\, dW_2

        dW_1 \\cdot dW_2 = \\rho \\, dt

    Uses Hagan et al. (2002) approximate implied volatility formula.

    Parameters
    ----------
    alpha : float
        Initial volatility level (must be positive).
    beta : float
        CEV exponent, 0 <= beta <= 1.
    rho : float
        Correlation, -1 < rho < 1.
    nu : float
        Vol of vol (must be non-negative).

    Notes
    -----
    Common beta values:
    - beta=0: Normal SABR (Bachelier-like)
    - beta=0.5: CIR-like
    - beta=1: Lognormal SABR

    Examples
    --------
    >>> sabr = SABR(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    >>> sabr.implied_vol(f=100, K=105, T=1)
    0.20...
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> None:
        """Initialize SABR model."""
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if not -1 < rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {rho}")
        if nu < 0:
            raise ValueError(f"nu must be non-negative, got {nu}")

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self, f: float, K: float, T: float) -> float:
        """
        Compute SABR implied volatility using Hagan's approximation.

        Parameters
        ----------
        f : float
            Forward price.
        K : float
            Strike price.
        T : float
            Time to expiration.

        Returns
        -------
        float
            Black implied volatility.

        Examples
        --------
        >>> sabr = SABR(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        >>> sabr.implied_vol(f=100, K=100, T=1)
        0.20...
        """
        if f <= 0 or K <= 0 or T <= 0:
            raise ValueError("f, K, and T must be positive")

        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        # ATM special case
        if abs(f - K) < 1e-12:
            return self._atm_vol(f, T)

        fk = f * K
        fk_beta = fk ** ((1 - beta) / 2)
        log_fk = np.log(f / K)

        # z and x(z)
        z = (nu / alpha) * fk_beta * log_fk
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        if abs(x_z) < 1e-12:
            zx_ratio = 1.0
        else:
            zx_ratio = z / x_z

        # Prefactor
        prefactor = alpha / (
            fk_beta
            * (
                1
                + (1 - beta) ** 2 / 24 * log_fk**2
                + (1 - beta) ** 4 / 1920 * log_fk**4
            )
        )

        # Correction term
        correction = 1 + (
            (1 - beta) ** 2 / 24 * alpha**2 / fk ** (1 - beta)
            + 0.25 * rho * beta * nu * alpha / fk_beta
            + (2 - 3 * rho**2) / 24 * nu**2
        ) * T

        return float(prefactor * zx_ratio * correction)

    def _atm_vol(self, f: float, T: float) -> float:
        """ATM implied volatility (F = K)."""
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu
        f_beta = f ** (1 - beta)

        vol = (alpha / f_beta) * (
            1
            + (
                (1 - beta) ** 2 / 24 * alpha**2 / f ** (2 - 2 * beta)
                + 0.25 * rho * beta * nu * alpha / f_beta
                + (2 - 3 * rho**2) / 24 * nu**2
            )
            * T
        )
        return float(vol)

    def smile(
        self,
        f: float,
        T: float,
        n_strikes: int = 50,
        strike_range: tuple[float, float] = (0.7, 1.3),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a volatility smile curve.

        Parameters
        ----------
        f : float
            Forward price.
        T : float
            Time to expiration.
        n_strikes : int, optional
            Number of strike points. Default is 50.
        strike_range : tuple[float, float], optional
            Strike range as fraction of forward (min, max).
            Default is (0.7, 1.3).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (strikes, implied_vols) arrays.
        """
        strikes = np.linspace(f * strike_range[0], f * strike_range[1], n_strikes)
        vols = np.array([self.implied_vol(f, K, T) for K in strikes])
        return strikes, vols

    @classmethod
    def fit(
        cls,
        f: float,
        strikes: np.ndarray,
        T: float,
        market_vols: np.ndarray,
        beta: float | None = None,
    ) -> "SABRResult":
        """
        Calibrate SABR parameters to market implied volatilities.

        Parameters
        ----------
        f : float
            Forward price.
        strikes : np.ndarray
            Array of strike prices.
        T : float
            Time to expiration.
        market_vols : np.ndarray
            Array of market implied volatilities.
        beta : float or None, optional
            If provided, beta is fixed. Otherwise, it is also calibrated.
            Common practice is to fix beta (e.g., 0.5).

        Returns
        -------
        SABRResult
            Calibration result with fitted parameters and error metrics.

        Examples
        --------
        >>> strikes = np.linspace(90, 110, 11)
        >>> # Generate synthetic market vols from known SABR
        >>> true_sabr = SABR(alpha=0.25, beta=0.5, rho=-0.3, nu=0.4)
        >>> market_vols = np.array([true_sabr.implied_vol(100, K, 1) for K in strikes])
        >>> result = SABR.fit(100, strikes, 1, market_vols, beta=0.5)
        >>> abs(result.alpha - 0.25) < 0.01
        True
        """
        strikes = np.asarray(strikes, dtype=float)
        market_vols = np.asarray(market_vols, dtype=float)

        if len(strikes) != len(market_vols):
            raise ValueError("strikes and market_vols must have same length")

        fix_beta = beta is not None

        def objective(params):
            if fix_beta:
                alpha, rho, nu = params
                b = beta
            else:
                alpha, b, rho, nu = params

            try:
                sabr = cls(alpha=alpha, beta=b, rho=rho, nu=nu)
                model_vols = np.array([sabr.implied_vol(f, K, T) for K in strikes])
                return np.sum((model_vols - market_vols) ** 2)
            except (ValueError, RuntimeWarning):
                return 1e10

        if fix_beta:
            # Initial guess: alpha from ATM vol, rho=0, nu=0.3
            atm_idx = np.argmin(np.abs(strikes - f))
            alpha_init = market_vols[atm_idx] * f ** (1 - beta)
            x0 = [alpha_init, 0.0, 0.3]
            bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]
        else:
            atm_idx = np.argmin(np.abs(strikes - f))
            alpha_init = market_vols[atm_idx]
            x0 = [alpha_init, 0.5, 0.0, 0.3]
            bounds = [(1e-6, 5.0), (0.0, 1.0), (-0.999, 0.999), (1e-6, 5.0)]

        result = optimize.minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
        )

        if fix_beta:
            alpha_fit, rho_fit, nu_fit = result.x
            beta_fit = beta
        else:
            alpha_fit, beta_fit, rho_fit, nu_fit = result.x

        # Compute error metrics
        sabr_fit = cls(alpha=alpha_fit, beta=beta_fit, rho=rho_fit, nu=nu_fit)
        model_vols = np.array([sabr_fit.implied_vol(f, K, T) for K in strikes])
        errors = model_vols - market_vols
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_error = float(np.max(np.abs(errors)))

        return SABRResult(
            alpha=float(alpha_fit),
            beta=float(beta_fit),
            rho=float(rho_fit),
            nu=float(nu_fit),
            rmse=rmse,
            max_error=max_error,
        )

    def __repr__(self) -> str:
        return (
            f"SABR(alpha={self.alpha}, beta={self.beta}, "
            f"rho={self.rho}, nu={self.nu})"
        )

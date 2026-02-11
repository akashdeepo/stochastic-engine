"""Heston model calibration to market implied volatilities."""

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from stochastic_engine.processes.heston import Heston
from stochastic_engine.volatility.implied import implied_volatility


@dataclass
class HestonCalibrationResult:
    """
    Result from Heston model calibration.

    Attributes
    ----------
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run variance.
    xi : float
        Vol of vol.
    rho : float
        Price-vol correlation.
    rmse : float
        Root mean squared error (in vol points).
    max_error : float
        Maximum absolute error.
    n_iterations : int
        Number of optimizer iterations.
    success : bool
        Whether optimization converged.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    rmse: float
    max_error: float
    n_iterations: int
    success: bool

    @property
    def feller_satisfied(self) -> bool:
        """Check if Feller condition is satisfied."""
        return 2 * self.kappa * self.theta > self.xi**2

    def to_heston(self, S0: float, mu: float = 0.05) -> Heston:
        """
        Create a Heston process from calibrated parameters.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        mu : float, optional
            Drift. Default is 0.05.

        Returns
        -------
        Heston
            Configured Heston process.
        """
        return Heston(
            S0=S0, v0=self.v0, mu=mu,
            kappa=self.kappa, theta=self.theta,
            xi=self.xi, rho=self.rho,
        )

    def __repr__(self) -> str:
        return (
            f"HestonCalibrationResult(v0={self.v0:.4f}, kappa={self.kappa:.4f}, "
            f"theta={self.theta:.4f}, xi={self.xi:.4f}, rho={self.rho:.4f}, "
            f"rmse={self.rmse:.6f}, feller={self.feller_satisfied})"
        )


class HestonCalibrator:
    """
    Calibrate Heston model parameters to market implied volatilities.

    Finds (v0, kappa, theta, xi, rho) that best match observed
    implied volatilities across strikes and maturities.

    Parameters
    ----------
    S : float
        Current spot price.
    r : float
        Risk-free rate.
    q : float, optional
        Dividend yield. Default is 0.

    Examples
    --------
    >>> calibrator = HestonCalibrator(S=100, r=0.05)
    >>> # market_ivs shape: (n_strikes, n_maturities)
    >>> result = calibrator.calibrate(strikes, maturities, market_ivs)
    >>> print(result.rmse)
    0.002...
    """

    # Parameter bounds
    BOUNDS = {
        "v0": (0.001, 1.0),
        "kappa": (0.01, 10.0),
        "theta": (0.001, 1.0),
        "xi": (0.01, 2.0),
        "rho": (-0.999, 0.999),
    }

    def __init__(
        self,
        S: float,
        r: float,
        q: float = 0.0,
    ) -> None:
        """Initialize HestonCalibrator."""
        self.S = float(S)
        self.r = float(r)
        self.q = float(q)

    def calibrate(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_ivs: np.ndarray,
        initial_params: dict | None = None,
        method: str = "least_squares",
    ) -> HestonCalibrationResult:
        """
        Calibrate Heston parameters to market data.

        Parameters
        ----------
        strikes : np.ndarray
            Strike prices, shape (n_strikes,).
        maturities : np.ndarray
            Maturities in years, shape (n_maturities,).
        market_ivs : np.ndarray
            Market implied volatilities, shape (n_strikes, n_maturities).
        initial_params : dict or None, optional
            Initial parameter guess with keys: v0, kappa, theta, xi, rho.
            If None, uses sensible defaults from market data.
        method : {"least_squares", "differential_evolution"}, optional
            Optimization method. Default is "least_squares".

        Returns
        -------
        HestonCalibrationResult
            Calibration result with fitted parameters.
        """
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        market_ivs = np.asarray(market_ivs, dtype=float)

        if market_ivs.shape != (len(strikes), len(maturities)):
            raise ValueError(
                f"market_ivs shape {market_ivs.shape} doesn't match "
                f"({len(strikes)}, {len(maturities)})"
            )

        # Initial guess
        if initial_params is None:
            atm_vol = float(np.median(market_ivs))
            initial_params = {
                "v0": atm_vol**2,
                "kappa": 2.0,
                "theta": atm_vol**2,
                "xi": 0.3,
                "rho": -0.5,
            }

        x0 = [
            initial_params["v0"],
            initial_params["kappa"],
            initial_params["theta"],
            initial_params["xi"],
            initial_params["rho"],
        ]

        bounds_list = [
            self.BOUNDS["v0"],
            self.BOUNDS["kappa"],
            self.BOUNDS["theta"],
            self.BOUNDS["xi"],
            self.BOUNDS["rho"],
        ]

        # Build flat arrays of (K, T, market_iv) for optimization
        K_flat = []
        T_flat = []
        iv_flat = []
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if not np.isnan(market_ivs[i, j]) and market_ivs[i, j] > 0:
                    K_flat.append(K)
                    T_flat.append(T)
                    iv_flat.append(market_ivs[i, j])

        K_flat = np.array(K_flat)
        T_flat = np.array(T_flat)
        iv_flat = np.array(iv_flat)

        if len(iv_flat) == 0:
            raise ValueError("No valid market IVs provided")

        def residuals(params):
            v0, kappa, theta, xi, rho = params
            model_ivs = np.zeros(len(K_flat))

            try:
                heston = Heston(
                    S0=self.S, v0=v0, mu=self.r - self.q,
                    kappa=kappa, theta=theta, xi=xi, rho=rho,
                )
            except ValueError:
                return np.full(len(K_flat), 1.0)

            for idx in range(len(K_flat)):
                try:
                    price = heston.price_european_call(
                        K=K_flat[idx], T=T_flat[idx], r=self.r,
                    )
                    if price > 0:
                        iv = implied_volatility(
                            market_price=price,
                            S=self.S, K=K_flat[idx], T=T_flat[idx], r=self.r,
                            q=self.q,
                        )
                        model_ivs[idx] = iv
                    else:
                        model_ivs[idx] = 0.01
                except Exception:
                    model_ivs[idx] = 0.01

            return model_ivs - iv_flat

        if method == "least_squares":
            result = optimize.least_squares(
                residuals, x0,
                bounds=(
                    [b[0] for b in bounds_list],
                    [b[1] for b in bounds_list],
                ),
                method="trf",
                max_nfev=500,
            )
            x_opt = result.x
            n_iter = result.nfev
            success = result.success

        elif method == "differential_evolution":
            def objective(params):
                res = residuals(params)
                return np.sum(res**2)

            result = optimize.differential_evolution(
                objective, bounds_list,
                x0=x0,
                maxiter=200,
                tol=1e-8,
                seed=42,
            )
            x_opt = result.x
            n_iter = result.nit
            success = result.success

        else:
            raise ValueError(f"Unknown method: {method}")

        v0_fit, kappa_fit, theta_fit, xi_fit, rho_fit = x_opt

        # Compute final errors
        final_residuals = residuals(x_opt)
        rmse = float(np.sqrt(np.mean(final_residuals**2)))
        max_error = float(np.max(np.abs(final_residuals)))

        return HestonCalibrationResult(
            v0=float(v0_fit),
            kappa=float(kappa_fit),
            theta=float(theta_fit),
            xi=float(xi_fit),
            rho=float(rho_fit),
            rmse=rmse,
            max_error=max_error,
            n_iterations=n_iter,
            success=success,
        )

    def __repr__(self) -> str:
        return f"HestonCalibrator(S={self.S}, r={self.r}, q={self.q})"

"""Interest rate models: Vasicek and Cox-Ingersoll-Ross (CIR)."""

from typing import Literal

import numpy as np
from scipy import stats

from stochastic_engine.processes.base import StochasticProcess


class Vasicek(StochasticProcess):
    """
    Vasicek interest rate model.

    A mean-reverting Gaussian process for short rates.

    .. math::
        dr_t = \\kappa (\\theta - r_t) \\, dt + \\sigma \\, dW_t

    Parameters
    ----------
    r0 : float
        Initial short rate.
    kappa : float
        Speed of mean reversion (must be positive).
    theta : float
        Long-run mean rate.
    sigma : float
        Volatility (must be non-negative).
    seed : int or None, optional
        Random seed for reproducibility.

    Notes
    -----
    The Vasicek model can produce negative rates, which is realistic
    in negative-rate environments (e.g., EUR, JPY post-2014).

    Has closed-form bond pricing formula.

    Examples
    --------
    >>> vasicek = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
    >>> paths = vasicek.simulate(T=10, steps=2520, n_paths=100)
    >>> paths.shape
    (100, 2521)
    """

    def __init__(
        self,
        r0: float,
        kappa: float,
        theta: float,
        sigma: float,
        seed: int | None = None,
    ) -> None:
        """Initialize Vasicek model."""
        super().__init__(seed=seed)

        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")

        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["euler", "exact"] = "exact",
    ) -> np.ndarray:
        """
        Simulate paths of the Vasicek model.

        Parameters
        ----------
        T : float
            Time horizon in years.
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths. Default is 1.
        method : {"euler", "exact"}, optional
            Discretization scheme. Default is "exact".

        Returns
        -------
        np.ndarray
            Rate paths of shape (n_paths, steps + 1).
        """
        dt = T / steps
        rng = self._get_rng()

        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.r0

        if method == "exact":
            # Exact transition: r_{t+dt} | r_t ~ N(mean, var)
            exp_kdt = np.exp(-self.kappa * dt)
            var_dt = (self.sigma**2 / (2 * self.kappa)) * (1 - exp_kdt**2)
            std_dt = np.sqrt(var_dt)

            for i in range(steps):
                mean_dt = self.theta + (paths[:, i] - self.theta) * exp_kdt
                paths[:, i + 1] = mean_dt + std_dt * rng.standard_normal(n_paths)

        elif method == "euler":
            sqrt_dt = np.sqrt(dt)
            for i in range(steps):
                dW = sqrt_dt * rng.standard_normal(n_paths)
                paths[:, i + 1] = (
                    paths[:, i]
                    + self.kappa * (self.theta - paths[:, i]) * dt
                    + self.sigma * dW
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        return paths

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample the rate at a specific time.

        Parameters
        ----------
        t : float
            Time point.
        n_samples : int, optional
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples of shape (n_samples,).
        """
        rng = self._get_rng()
        m = self.mean(t)
        v = self.variance(t)
        return rng.normal(m, np.sqrt(v), n_samples)

    def mean(self, t: float) -> float:
        """
        Expected value of rate at time t.

        .. math::
            E[r_t] = \\theta + (r_0 - \\theta) e^{-\\kappa t}

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Expected rate.
        """
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t: float) -> float:
        """
        Variance of rate at time t.

        .. math::
            Var(r_t) = \\frac{\\sigma^2}{2\\kappa} (1 - e^{-2\\kappa t})

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Variance.
        """
        return (self.sigma**2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa * t))

    def bond_price(self, T: float, t: float = 0.0, r: float | None = None) -> float:
        """
        Analytical zero-coupon bond price P(t, T).

        Parameters
        ----------
        T : float
            Maturity time.
        t : float, optional
            Current time. Default is 0.
        r : float or None, optional
            Current short rate. If None, uses r0.

        Returns
        -------
        float
            Zero-coupon bond price.

        Examples
        --------
        >>> vasicek = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
        >>> vasicek.bond_price(T=5)
        0.85...
        """
        if r is None:
            r = self.r0

        tau = T - t
        B = (1 - np.exp(-self.kappa * tau)) / self.kappa
        A = np.exp(
            (B - tau) * (self.kappa**2 * self.theta - self.sigma**2 / 2) / self.kappa**2
            - self.sigma**2 * B**2 / (4 * self.kappa)
        )
        return float(A * np.exp(-B * r))

    def yield_curve(self, maturities: np.ndarray) -> np.ndarray:
        """
        Compute yield curve for given maturities.

        Parameters
        ----------
        maturities : np.ndarray
            Array of maturities (in years).

        Returns
        -------
        np.ndarray
            Array of continuously compounded yields.

        Examples
        --------
        >>> vasicek = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
        >>> maturities = np.array([0.5, 1, 2, 5, 10])
        >>> yields = vasicek.yield_curve(maturities)
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.array([-np.log(self.bond_price(T)) / T for T in maturities])
        return yields

    @property
    def half_life(self) -> float:
        """Time to revert halfway to the mean."""
        return np.log(2) / self.kappa

    def __repr__(self) -> str:
        return (
            f"Vasicek(r0={self.r0}, kappa={self.kappa}, "
            f"theta={self.theta}, sigma={self.sigma})"
        )


class CIR(StochasticProcess):
    """
    Cox-Ingersoll-Ross (CIR) interest rate model.

    A mean-reverting, non-negative process for short rates.

    .. math::
        dr_t = \\kappa (\\theta - r_t) \\, dt + \\sigma \\sqrt{r_t} \\, dW_t

    The ``sqrt(r_t)`` diffusion term ensures non-negativity when the
    Feller condition ``2 * kappa * theta > sigma^2`` is satisfied.

    Parameters
    ----------
    r0 : float
        Initial short rate (must be non-negative).
    kappa : float
        Speed of mean reversion (must be positive).
    theta : float
        Long-run mean rate (must be non-negative).
    sigma : float
        Volatility (must be non-negative).
    seed : int or None, optional
        Random seed for reproducibility.

    Attributes
    ----------
    feller_satisfied : bool
        Whether the Feller condition is met.

    Examples
    --------
    >>> cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
    >>> paths = cir.simulate(T=10, steps=2520, n_paths=100)
    >>> (paths >= 0).all()  # Non-negative
    True
    """

    def __init__(
        self,
        r0: float,
        kappa: float,
        theta: float,
        sigma: float,
        seed: int | None = None,
    ) -> None:
        """Initialize CIR model."""
        super().__init__(seed=seed)

        if r0 < 0:
            raise ValueError(f"r0 must be non-negative, got {r0}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if theta < 0:
            raise ValueError(f"theta must be non-negative, got {theta}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")

        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    @property
    def feller_satisfied(self) -> bool:
        """
        Check if the Feller condition is satisfied.

        The Feller condition ``2 * kappa * theta > sigma^2`` ensures
        the rate process stays strictly positive.
        """
        return 2 * self.kappa * self.theta > self.sigma**2

    def simulate(
        self,
        T: float,
        steps: int,
        n_paths: int = 1,
        method: Literal["euler", "exact"] = "euler",
    ) -> np.ndarray:
        """
        Simulate paths of the CIR model.

        Parameters
        ----------
        T : float
            Time horizon in years.
        steps : int
            Number of time steps.
        n_paths : int, optional
            Number of paths. Default is 1.
        method : {"euler", "exact"}, optional
            Discretization scheme. Default is "euler".
            - "euler": Euler-Maruyama with reflection at zero.
            - "exact": Uses non-central chi-squared transition density.

        Returns
        -------
        np.ndarray
            Rate paths of shape (n_paths, steps + 1).
        """
        dt = T / steps
        rng = self._get_rng()

        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.r0

        if method == "exact":
            # Exact simulation using non-central chi-squared
            exp_kdt = np.exp(-self.kappa * dt)
            df = 4 * self.kappa * self.theta / self.sigma**2

            for i in range(steps):
                r_curr = np.maximum(paths[:, i], 0)
                c = self.sigma**2 * (1 - exp_kdt) / (4 * self.kappa)
                nc = 4 * self.kappa * exp_kdt / (self.sigma**2 * (1 - exp_kdt)) * r_curr
                # Non-central chi-squared
                paths[:, i + 1] = c * rng.noncentral_chisquare(df, nc)

        elif method == "euler":
            sqrt_dt = np.sqrt(dt)
            for i in range(steps):
                r_curr = np.maximum(paths[:, i], 0)
                dW = sqrt_dt * rng.standard_normal(n_paths)
                paths[:, i + 1] = (
                    r_curr
                    + self.kappa * (self.theta - r_curr) * dt
                    + self.sigma * np.sqrt(r_curr) * dW
                )
                # Reflection at zero
                paths[:, i + 1] = np.maximum(paths[:, i + 1], 0)
        else:
            raise ValueError(f"Unknown method: {method}")

        return paths

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """
        Sample the rate at a specific time using non-central chi-squared.

        Parameters
        ----------
        t : float
            Time point.
        n_samples : int, optional
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples of shape (n_samples,).
        """
        rng = self._get_rng()
        exp_kt = np.exp(-self.kappa * t)
        c = self.sigma**2 * (1 - exp_kt) / (4 * self.kappa)
        df = 4 * self.kappa * self.theta / self.sigma**2
        nc = 4 * self.kappa * exp_kt / (self.sigma**2 * (1 - exp_kt)) * self.r0
        return c * rng.noncentral_chisquare(df, nc, size=n_samples)

    def mean(self, t: float) -> float:
        """
        Expected value of rate at time t.

        .. math::
            E[r_t] = \\theta + (r_0 - \\theta) e^{-\\kappa t}

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Expected rate.
        """
        return self.theta + (self.r0 - self.theta) * np.exp(-self.kappa * t)

    def variance(self, t: float) -> float:
        """
        Variance of rate at time t.

        Parameters
        ----------
        t : float
            Time point.

        Returns
        -------
        float
            Variance.
        """
        exp_kt = np.exp(-self.kappa * t)
        return (
            self.r0 * (self.sigma**2 / self.kappa) * (exp_kt - exp_kt**2)
            + (self.theta * self.sigma**2 / (2 * self.kappa)) * (1 - exp_kt) ** 2
        )

    def bond_price(self, T: float, t: float = 0.0, r: float | None = None) -> float:
        """
        Analytical zero-coupon bond price P(t, T).

        Parameters
        ----------
        T : float
            Maturity time.
        t : float, optional
            Current time. Default is 0.
        r : float or None, optional
            Current short rate. If None, uses r0.

        Returns
        -------
        float
            Zero-coupon bond price.

        Examples
        --------
        >>> cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
        >>> cir.bond_price(T=5)
        0.85...
        """
        if r is None:
            r = self.r0

        tau = T - t
        gamma = np.sqrt(self.kappa**2 + 2 * self.sigma**2)
        exp_gamma_tau = np.exp(gamma * tau)

        denom = (gamma + self.kappa) * (exp_gamma_tau - 1) + 2 * gamma
        B = 2 * (exp_gamma_tau - 1) / denom
        A_exp = 2 * self.kappa * self.theta / self.sigma**2
        A = (2 * gamma * np.exp((self.kappa + gamma) * tau / 2) / denom) ** A_exp

        return float(A * np.exp(-B * r))

    def yield_curve(self, maturities: np.ndarray) -> np.ndarray:
        """
        Compute yield curve for given maturities.

        Parameters
        ----------
        maturities : np.ndarray
            Array of maturities (in years).

        Returns
        -------
        np.ndarray
            Array of continuously compounded yields.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.array([-np.log(self.bond_price(T)) / T for T in maturities])
        return yields

    def __repr__(self) -> str:
        return (
            f"CIR(r0={self.r0}, kappa={self.kappa}, "
            f"theta={self.theta}, sigma={self.sigma})"
        )

"""Tests for volatility calculations."""

import numpy as np
import pytest

from stochastic_engine.volatility import implied_volatility, ImpliedVolSolver
from stochastic_engine.volatility.implied import implied_volatility_surface
from stochastic_engine.pricing import BlackScholes
from stochastic_engine.volatility.sabr import SABR, SABRResult
from stochastic_engine.volatility.heston_calibration import (
    HestonCalibrator,
    HestonCalibrationResult,
)


class TestImpliedVolatility:
    """Tests for implied volatility solver."""

    def test_recover_known_volatility(self):
        """Test that IV solver recovers the input volatility."""
        # Calculate BS price with known vol
        sigma_true = 0.25
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=sigma_true)
        price = bs.price

        # Recover IV
        iv = implied_volatility(price, S=100, K=100, T=1, r=0.05)

        assert np.isclose(iv, sigma_true, atol=1e-6)

    def test_different_moneyness(self):
        """Test IV solver at different moneyness levels."""
        sigma_true = 0.2

        for K in [80, 90, 100, 110, 120]:
            bs = BlackScholes(S=100, K=K, T=1, r=0.05, sigma=sigma_true)
            iv = implied_volatility(bs.price, S=100, K=K, T=1, r=0.05)

            assert np.isclose(iv, sigma_true, atol=1e-5)

    def test_different_maturities(self):
        """Test IV solver at different maturities."""
        sigma_true = 0.2

        for T in [0.1, 0.25, 0.5, 1.0, 2.0]:
            bs = BlackScholes(S=100, K=100, T=T, r=0.05, sigma=sigma_true)
            iv = implied_volatility(bs.price, S=100, K=100, T=T, r=0.05)

            assert np.isclose(iv, sigma_true, atol=1e-5)

    def test_put_option(self):
        """Test IV solver for put options."""
        sigma_true = 0.2
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=sigma_true, option_type="put")

        iv = implied_volatility(
            bs.price, S=100, K=100, T=1, r=0.05, option_type="put"
        )

        assert np.isclose(iv, sigma_true, atol=1e-5)

    def test_different_methods(self):
        """Test different numerical methods give same result."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)

        iv_newton = implied_volatility(bs.price, S=100, K=100, T=1, r=0.05, method="newton")
        iv_bisection = implied_volatility(bs.price, S=100, K=100, T=1, r=0.05, method="bisection")
        iv_brent = implied_volatility(bs.price, S=100, K=100, T=1, r=0.05, method="brent")

        assert np.isclose(iv_newton, iv_bisection, atol=1e-5)
        assert np.isclose(iv_brent, iv_bisection, atol=1e-5)

    def test_with_dividends(self):
        """Test IV solver with continuous dividend yield."""
        sigma_true = 0.2
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=sigma_true, q=0.02)

        iv = implied_volatility(bs.price, S=100, K=100, T=1, r=0.05, q=0.02)

        assert np.isclose(iv, sigma_true, atol=1e-5)

    def test_invalid_price_raises_error(self):
        """Test that arbitrage prices raise error."""
        # Price below intrinsic value
        with pytest.raises(ValueError, match="outside valid bounds"):
            implied_volatility(market_price=1.0, S=100, K=90, T=1, r=0.05)

    def test_solver_class(self):
        """Test ImpliedVolSolver class."""
        solver = ImpliedVolSolver(S=100, K=100, T=1, r=0.05)
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)

        iv = solver.solve(market_price=bs.price)

        assert np.isclose(iv, 0.2, atol=1e-5)

    def test_iv_surface(self):
        """Test implied volatility surface calculation."""
        # Create option prices with constant vol
        sigma = 0.2
        strikes = np.array([95, 100, 105])
        expirations = np.array([0.25, 0.5])

        prices = np.zeros((3, 2))
        for i, K in enumerate(strikes):
            for j, T in enumerate(expirations):
                prices[i, j] = BlackScholes(
                    S=100, K=K, T=T, r=0.05, sigma=sigma
                ).price

        iv_surface = implied_volatility_surface(
            market_prices=prices,
            S=100,
            strikes=strikes,
            expirations=expirations,
            r=0.05,
        )

        # All IVs should be close to 0.2
        assert np.allclose(iv_surface, sigma, atol=1e-4)

    def test_extreme_moneyness(self):
        """Test IV for deep ITM/OTM options."""
        sigma = 0.3

        # Deep ITM call
        bs_itm = BlackScholes(S=150, K=100, T=0.25, r=0.05, sigma=sigma)
        iv_itm = implied_volatility(bs_itm.price, S=150, K=100, T=0.25, r=0.05)
        assert np.isclose(iv_itm, sigma, atol=1e-3)

        # Deep OTM call
        bs_otm = BlackScholes(S=100, K=150, T=0.25, r=0.05, sigma=sigma)
        iv_otm = implied_volatility(bs_otm.price, S=100, K=150, T=0.25, r=0.05)
        assert np.isclose(iv_otm, sigma, atol=1e-3)


class TestSABR:
    """Tests for SABR volatility model."""

    def test_atm_vol(self):
        """ATM vol should be close to alpha / f^(1-beta)."""
        sabr = SABR(alpha=0.2, beta=0.5, rho=0.0, nu=0.0)
        # With nu=0, ATM vol should be very close to alpha / f^(1-beta)
        f = 100.0
        vol = sabr.implied_vol(f, f, T=1.0)
        expected = 0.2 / f ** (1 - 0.5)
        assert np.isclose(vol, expected, rtol=0.01)

    def test_implied_vol_positive(self):
        """Implied vol should always be positive."""
        sabr = SABR(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        for K in [80, 90, 100, 110, 120]:
            vol = sabr.implied_vol(f=100, K=K, T=1.0)
            assert vol > 0, f"Negative vol for K={K}"

    def test_smile_shape_negative_rho(self):
        """Negative rho should produce downward-sloping skew."""
        sabr = SABR(alpha=0.2, beta=0.5, rho=-0.5, nu=0.4)
        vol_low = sabr.implied_vol(f=100, K=85, T=1.0)
        vol_high = sabr.implied_vol(f=100, K=115, T=1.0)
        # Low strikes should have higher vol with negative rho
        assert vol_low > vol_high

    def test_smile_method(self):
        """Test smile method returns correct shapes."""
        sabr = SABR(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        strikes, vols = sabr.smile(f=100, T=1.0, n_strikes=20)
        assert strikes.shape == (20,)
        assert vols.shape == (20,)
        assert np.all(vols > 0)

    def test_fit_recovers_params(self):
        """Calibration should recover known parameters."""
        true_sabr = SABR(alpha=0.25, beta=0.5, rho=-0.3, nu=0.4)
        strikes = np.linspace(80, 120, 21)
        market_vols = np.array([true_sabr.implied_vol(100, K, 1.0) for K in strikes])

        result = SABR.fit(f=100, strikes=strikes, T=1.0,
                          market_vols=market_vols, beta=0.5)

        assert isinstance(result, SABRResult)
        assert np.isclose(result.alpha, 0.25, atol=0.02)
        assert np.isclose(result.rho, -0.3, atol=0.05)
        assert np.isclose(result.nu, 0.4, atol=0.05)
        assert result.rmse < 0.001

    def test_fit_free_beta(self):
        """Calibration with free beta."""
        true_sabr = SABR(alpha=0.25, beta=0.7, rho=-0.2, nu=0.3)
        strikes = np.linspace(80, 120, 21)
        market_vols = np.array([true_sabr.implied_vol(100, K, 1.0) for K in strikes])

        result = SABR.fit(f=100, strikes=strikes, T=1.0, market_vols=market_vols)
        assert result.rmse < 0.005

    def test_invalid_alpha(self):
        """Negative alpha should raise ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            SABR(alpha=-0.1, beta=0.5, rho=0.0, nu=0.3)

    def test_invalid_beta(self):
        """Beta outside [0,1] should raise ValueError."""
        with pytest.raises(ValueError, match="beta"):
            SABR(alpha=0.2, beta=1.5, rho=0.0, nu=0.3)

    def test_invalid_rho(self):
        """Rho outside (-1,1) should raise ValueError."""
        with pytest.raises(ValueError, match="rho"):
            SABR(alpha=0.2, beta=0.5, rho=1.0, nu=0.3)

    def test_invalid_nu(self):
        """Negative nu should raise ValueError."""
        with pytest.raises(ValueError, match="nu"):
            SABR(alpha=0.2, beta=0.5, rho=0.0, nu=-0.1)

    def test_beta_zero_normal(self):
        """Beta=0 (normal SABR) should produce valid vols."""
        sabr = SABR(alpha=0.2, beta=0.0, rho=-0.2, nu=0.3)
        vol = sabr.implied_vol(f=100, K=100, T=1.0)
        assert vol > 0

    def test_beta_one_lognormal(self):
        """Beta=1 (lognormal SABR) should produce valid vols."""
        sabr = SABR(alpha=0.2, beta=1.0, rho=-0.2, nu=0.3)
        vol = sabr.implied_vol(f=100, K=105, T=1.0)
        assert vol > 0


class TestHestonCalibration:
    """Tests for Heston model calibration."""

    @pytest.fixture
    def synthetic_market_data(self):
        """Generate synthetic market data from a known Heston model."""
        from stochastic_engine.processes.heston import Heston

        # Known Heston parameters
        S = 100.0
        r = 0.05
        v0 = 0.04
        kappa = 2.0
        theta = 0.04
        xi = 0.3
        rho = -0.7

        heston = Heston(S0=S, v0=v0, mu=r, kappa=kappa, theta=theta, xi=xi, rho=rho)

        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        maturities = np.array([0.25, 0.5, 1.0], dtype=float)

        market_ivs = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                price = heston.price_european_call(K=K, T=T, r=r)
                if price > 0:
                    try:
                        iv = implied_volatility(price, S=S, K=K, T=T, r=r)
                        market_ivs[i, j] = iv
                    except (ValueError, RuntimeError):
                        market_ivs[i, j] = 0.2
                else:
                    market_ivs[i, j] = 0.2

        return {
            "S": S, "r": r, "strikes": strikes,
            "maturities": maturities, "market_ivs": market_ivs,
            "true_params": {"v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho},
        }

    def test_calibrator_construction(self):
        """Test HestonCalibrator can be constructed."""
        cal = HestonCalibrator(S=100, r=0.05)
        assert cal.S == 100
        assert cal.r == 0.05
        assert cal.q == 0.0

    def test_calibration_runs(self, synthetic_market_data):
        """Test that calibration completes without error."""
        data = synthetic_market_data
        cal = HestonCalibrator(S=data["S"], r=data["r"])
        result = cal.calibrate(
            data["strikes"], data["maturities"], data["market_ivs"],
        )
        assert isinstance(result, HestonCalibrationResult)
        assert result.rmse >= 0
        assert result.max_error >= 0

    def test_calibration_reasonable_params(self, synthetic_market_data):
        """Calibrated params should be in reasonable ranges."""
        data = synthetic_market_data
        cal = HestonCalibrator(S=data["S"], r=data["r"])
        result = cal.calibrate(
            data["strikes"], data["maturities"], data["market_ivs"],
        )
        assert 0 < result.v0 < 1
        assert 0 < result.kappa < 10
        assert 0 < result.theta < 1
        assert 0 < result.xi < 2
        assert -1 < result.rho < 1

    def test_result_to_heston(self, synthetic_market_data):
        """Test to_heston method creates valid Heston process."""
        data = synthetic_market_data
        cal = HestonCalibrator(S=data["S"], r=data["r"])
        result = cal.calibrate(
            data["strikes"], data["maturities"], data["market_ivs"],
        )
        heston = result.to_heston(S0=100, mu=0.05)
        # Should be able to simulate
        paths = heston.simulate(T=1.0, steps=100, n_paths=10)
        assert paths.shape == (10, 101)

    def test_feller_condition_property(self):
        """Test feller_satisfied property."""
        # Feller satisfied: 2*kappa*theta > xi^2 => 2*2*0.04 > 0.3^2 => 0.16 > 0.09
        result = HestonCalibrationResult(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            rmse=0.001, max_error=0.002, n_iterations=10, success=True,
        )
        assert result.feller_satisfied

        # Feller NOT satisfied: 2*0.5*0.01 > 1.0^2 => 0.01 > 1.0 => False
        result2 = HestonCalibrationResult(
            v0=0.01, kappa=0.5, theta=0.01, xi=1.0, rho=-0.5,
            rmse=0.01, max_error=0.02, n_iterations=5, success=True,
        )
        assert not result2.feller_satisfied

    def test_invalid_iv_shape(self):
        """Mismatched shapes should raise ValueError."""
        cal = HestonCalibrator(S=100, r=0.05)
        with pytest.raises(ValueError, match="shape"):
            cal.calibrate(
                strikes=np.array([90, 100, 110]),
                maturities=np.array([0.5, 1.0]),
                market_ivs=np.ones((2, 3)),  # Wrong shape
            )

    def test_repr(self):
        """Test repr strings."""
        cal = HestonCalibrator(S=100, r=0.05)
        assert "100" in repr(cal)

        result = HestonCalibrationResult(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
            rmse=0.001, max_error=0.002, n_iterations=10, success=True,
        )
        assert "0.04" in repr(result)

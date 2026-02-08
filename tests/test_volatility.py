"""Tests for volatility calculations."""

import numpy as np
import pytest

from stochastic_engine.volatility import implied_volatility, ImpliedVolSolver
from stochastic_engine.volatility.implied import implied_volatility_surface
from stochastic_engine.pricing import BlackScholes


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

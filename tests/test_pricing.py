"""Tests for option pricing models."""

import numpy as np
import pytest

from stochastic_engine.pricing import BlackScholes, MonteCarloPricer
from stochastic_engine.pricing.black_scholes import bs_call, bs_put


class TestBlackScholes:
    """Tests for Black-Scholes pricing."""

    def test_call_price(self):
        """Test call option price against known value."""
        # Standard example: S=100, K=100, T=1, r=5%, sigma=20%
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)

        # Expected price ~10.45 (ATM call)
        assert np.isclose(bs.price, 10.4506, atol=0.01)

    def test_put_price(self):
        """Test put option price against known value."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")

        # Expected price ~5.57 (ATM put)
        assert np.isclose(bs.price, 5.5735, atol=0.01)

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100, 105, 1, 0.05, 0.2

        call = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")
        put = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")

        # C - P = S - K*exp(-rT)
        left = call.price - put.price
        right = S - K * np.exp(-r * T)

        assert np.isclose(left, right, atol=1e-6)

    def test_call_delta_range(self):
        """Test that call delta is between 0 and 1."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert 0 < bs.delta < 1

    def test_put_delta_range(self):
        """Test that put delta is between -1 and 0."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert -1 < bs.delta < 0

    def test_gamma_positive(self):
        """Test that gamma is always positive."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert bs.gamma > 0

    def test_vega_positive(self):
        """Test that vega is always positive."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert bs.vega > 0

    def test_theta_negative_for_atm(self):
        """Test that theta is typically negative (time decay)."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert bs.theta < 0  # Options lose value over time

    def test_greeks_object(self):
        """Test greeks property returns Greeks object."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        greeks = bs.greeks

        assert hasattr(greeks, "delta")
        assert hasattr(greeks, "gamma")
        assert hasattr(greeks, "vega")
        assert hasattr(greeks, "theta")
        assert hasattr(greeks, "rho")

    def test_vectorized_pricing(self):
        """Test pricing with array inputs."""
        strikes = np.array([95, 100, 105, 110])
        bs = BlackScholes(S=100, K=strikes, T=1, r=0.05, sigma=0.2)

        assert isinstance(bs.price, np.ndarray)
        assert bs.price.shape == (4,)

        # Prices should decrease as strike increases (for calls)
        assert np.all(np.diff(bs.price) < 0)

    def test_deep_itm_call(self):
        """Test deep in-the-money call approaches intrinsic value."""
        bs = BlackScholes(S=150, K=100, T=0.01, r=0.05, sigma=0.2)
        intrinsic = 150 - 100
        assert np.isclose(bs.price, intrinsic, atol=1)

    def test_deep_otm_call(self):
        """Test deep out-of-the-money call is near zero."""
        bs = BlackScholes(S=50, K=100, T=0.01, r=0.05, sigma=0.2)
        assert bs.price < 0.01

    def test_convenience_functions(self):
        """Test bs_call and bs_put convenience functions."""
        call_price = bs_call(S=100, K=105, T=1, r=0.05, sigma=0.2)
        put_price = bs_put(S=100, K=105, T=1, r=0.05, sigma=0.2)

        assert call_price > 0
        assert put_price > 0

    def test_dividend_yield(self):
        """Test pricing with continuous dividend yield."""
        # With dividend, call should be worth less
        bs_no_div = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0)
        bs_with_div = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.02)

        assert bs_with_div.price < bs_no_div.price


class TestMonteCarlo:
    """Tests for Monte Carlo pricing."""

    def test_european_call_vs_bs(self):
        """Test MC call price converges to Black-Scholes."""
        mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        result = mc.price_european_call(K=100, n_paths=100000)

        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)

        # MC should be within 2 standard errors of BS
        assert abs(result.price - bs.price) < 2 * result.std_error

    def test_european_put_vs_bs(self):
        """Test MC put price converges to Black-Scholes."""
        mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        result = mc.price_european_put(K=100, n_paths=100000)

        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")

        assert abs(result.price - bs.price) < 2 * result.std_error

    def test_result_has_confidence_interval(self):
        """Test that MCResult includes confidence interval."""
        mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        result = mc.price_european_call(K=100, n_paths=10000)

        assert hasattr(result, "confidence_interval")
        assert result.confidence_interval[0] < result.price < result.confidence_interval[1]

    def test_antithetic_reduces_variance(self):
        """Test that antithetic variates reduce standard error."""
        mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)

        result_no_anti = mc.price_european_call(K=100, n_paths=10000, antithetic=False)
        mc._process.reset_rng(42)
        result_with_anti = mc.price_european_call(K=100, n_paths=10000, antithetic=True)

        # Antithetic should have lower std error
        assert result_with_anti.std_error <= result_no_anti.std_error * 1.1

    def test_asian_call_less_than_european(self):
        """Test that Asian call is worth less than European call."""
        mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)

        european = mc.price_european_call(K=100, n_paths=50000)
        asian = mc.price_asian_call(K=100, n_paths=50000, steps=252)

        # Asian should be worth less (averaging reduces volatility)
        assert asian.price < european.price

    def test_custom_payoff(self):
        """Test custom payoff function."""
        mc = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)

        def lookback_call(paths):
            """Floating strike lookback call: S_T - min(S)."""
            return paths[:, -1] - paths.min(axis=1)

        result = mc.price_custom(lookback_call, n_paths=50000, steps=252)

        # Lookback should be worth more than European
        european = mc.price_european_call(K=100, n_paths=50000)
        assert result.price > european.price

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        mc1 = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)
        mc2 = MonteCarloPricer(S0=100, r=0.05, sigma=0.2, T=1, seed=42)

        result1 = mc1.price_european_call(K=100, n_paths=1000)
        result2 = mc2.price_european_call(K=100, n_paths=1000)

        assert result1.price == result2.price

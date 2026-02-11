"""Tests for option pricing models."""

import numpy as np
import pytest

from stochastic_engine.pricing import BlackScholes, MonteCarloPricer
from stochastic_engine.pricing.black_scholes import bs_call, bs_put
from stochastic_engine.pricing.digital import (
    DigitalOption,
    cash_or_nothing_call,
    cash_or_nothing_put,
    asset_or_nothing_call,
    asset_or_nothing_put,
)
from stochastic_engine.pricing.barrier import BarrierOption, barrier_call, barrier_put


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


class TestDigitalOptions:
    """Tests for digital (binary) option pricing."""

    def test_cash_or_nothing_call_price(self):
        """Test cash-or-nothing call has reasonable price."""
        opt = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2)
        # ATM CoN call should be around 0.5 * df
        assert 0.3 < opt.price < 0.7

    def test_cash_or_nothing_put_price(self):
        """Test cash-or-nothing put has reasonable price."""
        opt = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert 0.3 < opt.price < 0.7

    def test_call_put_sum_to_discount(self):
        """CoN call + CoN put = Q * exp(-rT)."""
        S, K, T, r, sigma, Q = 100, 105, 1, 0.05, 0.2, 1.0
        call = DigitalOption(S=S, K=K, T=T, r=r, sigma=sigma, Q=Q, option_type="call")
        put = DigitalOption(S=S, K=K, T=T, r=r, sigma=sigma, Q=Q, option_type="put")
        assert np.isclose(call.price + put.price, Q * np.exp(-r * T), atol=1e-10)

    def test_bs_decomposition_identity(self):
        """BS_call = AoN_call - K * CoN_call."""
        S, K, T, r, sigma = 100, 105, 1, 0.05, 0.2
        bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)

        aon = DigitalOption(S=S, K=K, T=T, r=r, sigma=sigma,
                            option_type="call", digital_type="asset-or-nothing")
        con = DigitalOption(S=S, K=K, T=T, r=r, sigma=sigma,
                            option_type="call", digital_type="cash-or-nothing")

        assert np.isclose(bs.price, aon.price - K * con.price, atol=1e-6)

    def test_deep_itm_call_near_discount(self):
        """Deep ITM CoN call should approach Q * exp(-rT)."""
        opt = DigitalOption(S=200, K=100, T=1, r=0.05, sigma=0.2)
        assert np.isclose(opt.price, np.exp(-0.05), atol=0.05)

    def test_deep_otm_call_near_zero(self):
        """Deep OTM CoN call should approach 0."""
        opt = DigitalOption(S=50, K=100, T=0.1, r=0.05, sigma=0.2)
        assert opt.price < 0.01

    def test_cash_amount_scaling(self):
        """Price should scale linearly with Q."""
        p1 = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2, Q=1).price
        p10 = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2, Q=10).price
        assert np.isclose(p10, 10 * p1, atol=1e-10)

    def test_greeks_exist(self):
        """Test that Greeks are computed."""
        opt = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert isinstance(opt.delta, float)
        assert isinstance(opt.gamma, float)
        assert isinstance(opt.vega, float)

    def test_vectorized_strikes(self):
        """Test vectorized pricing over strikes."""
        strikes = np.array([95, 100, 105, 110])
        opt = DigitalOption(S=100, K=strikes, T=1, r=0.05, sigma=0.2)
        prices = opt.price
        assert isinstance(prices, np.ndarray)
        assert prices.shape == (4,)
        # CoN call price should decrease as strike increases
        assert np.all(np.diff(prices) < 0)

    def test_convenience_functions(self):
        """Test convenience functions match class."""
        S, K, T, r, sigma = 100, 105, 1, 0.05, 0.2
        assert np.isclose(
            cash_or_nothing_call(S, K, T, r, sigma),
            DigitalOption(S, K, T, r, sigma, option_type="call",
                          digital_type="cash-or-nothing").price,
        )
        assert np.isclose(
            cash_or_nothing_put(S, K, T, r, sigma),
            DigitalOption(S, K, T, r, sigma, option_type="put",
                          digital_type="cash-or-nothing").price,
        )
        assert np.isclose(
            asset_or_nothing_call(S, K, T, r, sigma),
            DigitalOption(S, K, T, r, sigma, option_type="call",
                          digital_type="asset-or-nothing").price,
        )
        assert np.isclose(
            asset_or_nothing_put(S, K, T, r, sigma),
            DigitalOption(S, K, T, r, sigma, option_type="put",
                          digital_type="asset-or-nothing").price,
        )

    def test_dividend_yield(self):
        """CoN call with dividend should be cheaper."""
        p_no_div = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0).price
        p_div = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.03).price
        assert p_div < p_no_div

    def test_invalid_option_type(self):
        """Test invalid option type raises error."""
        with pytest.raises(ValueError):
            DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="invalid")

    def test_invalid_digital_type(self):
        """Test invalid digital type raises error."""
        with pytest.raises(ValueError):
            DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2, digital_type="invalid")

    def test_result_object(self):
        """Test result property returns DigitalResult."""
        opt = DigitalOption(S=100, K=100, T=1, r=0.05, sigma=0.2)
        result = opt.result
        assert result.digital_type == "cash-or-nothing"
        assert result.option_type == "call"
        assert result.price > 0


class TestBarrierOptions:
    """Tests for barrier option pricing."""

    def test_down_and_out_call_less_than_vanilla(self):
        """Down-and-out call should be worth less than vanilla call."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        bo = BarrierOption(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                           barrier_type="down-and-out")
        assert bo.price().price < bs.price
        assert bo.price().price > 0

    def test_in_out_parity_call(self):
        """V_in + V_out = V_vanilla for calls."""
        S, K, B, T, r, sigma = 100, 100, 80, 1, 0.05, 0.2
        bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
        bo_in = BarrierOption(S=S, K=K, B=B, T=T, r=r, sigma=sigma,
                              barrier_type="down-and-in")
        bo_out = BarrierOption(S=S, K=K, B=B, T=T, r=r, sigma=sigma,
                               barrier_type="down-and-out")
        assert np.isclose(
            bo_in.price().price + bo_out.price().price,
            bs.price,
            atol=0.01,
        )

    def test_in_out_parity_put(self):
        """V_in + V_out = V_vanilla for puts."""
        S, K, B, T, r, sigma = 100, 100, 120, 1, 0.05, 0.2
        bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")
        bo_in = BarrierOption(S=S, K=K, B=B, T=T, r=r, sigma=sigma,
                              option_type="put", barrier_type="up-and-in")
        bo_out = BarrierOption(S=S, K=K, B=B, T=T, r=r, sigma=sigma,
                               option_type="put", barrier_type="up-and-out")
        assert np.isclose(
            bo_in.price().price + bo_out.price().price,
            bs.price,
            atol=0.01,
        )

    def test_far_barrier_equals_vanilla(self):
        """Very far down barrier should give price close to vanilla."""
        bs = BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        bo = BarrierOption(S=100, K=100, B=1, T=1, r=0.05, sigma=0.2,
                           barrier_type="down-and-out")
        assert np.isclose(bo.price().price, bs.price, atol=0.1)

    def test_up_and_out_call(self):
        """Up-and-out call has positive price."""
        bo = BarrierOption(S=100, K=100, B=130, T=1, r=0.05, sigma=0.2,
                           barrier_type="up-and-out")
        result = bo.price()
        assert result.price >= 0
        assert result.barrier_type == "up-and-out"
        assert result.option_type == "call"

    def test_down_and_in_put(self):
        """Down-and-in put has positive price."""
        bo = BarrierOption(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                           option_type="put", barrier_type="down-and-in")
        assert bo.price().price > 0

    def test_monte_carlo_vs_closed_form(self):
        """MC price should be close to closed-form."""
        bo = BarrierOption(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                           barrier_type="down-and-out")
        cf = bo.price(method="closed_form")
        mc = bo.price(method="monte_carlo", n_paths=200000, steps=500, seed=42)
        # MC should be within a reasonable tolerance of closed form
        assert abs(mc.price - cf.price) < 3 * mc.std_error + 0.5

    def test_convenience_functions(self):
        """Test barrier_call and barrier_put convenience functions."""
        p_call = barrier_call(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2)
        p_put = barrier_put(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                            barrier_type="down-and-out")
        assert p_call > 0
        assert p_put > 0

    def test_invalid_barrier_type(self):
        """Test invalid barrier type raises error."""
        with pytest.raises(ValueError):
            BarrierOption(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                          barrier_type="invalid")

    def test_result_has_method(self):
        """Test result reports the method used."""
        bo = BarrierOption(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                           barrier_type="down-and-out")
        assert bo.price(method="closed_form").method == "closed_form"

    def test_mc_result_has_std_error(self):
        """Test MC result has std_error and confidence interval."""
        bo = BarrierOption(S=100, K=100, B=80, T=1, r=0.05, sigma=0.2,
                           barrier_type="down-and-out")
        mc = bo.price(method="monte_carlo", n_paths=10000, steps=100, seed=42)
        assert mc.std_error is not None
        assert mc.confidence_interval is not None
        assert mc.confidence_interval[0] < mc.price < mc.confidence_interval[1]

    def test_all_barrier_types_run(self):
        """Verify all 8 barrier type + option type combos run."""
        for bt in ["up-and-in", "up-and-out", "down-and-in", "down-and-out"]:
            for ot in ["call", "put"]:
                B = 120 if "up" in bt else 80
                bo = BarrierOption(S=100, K=100, B=B, T=1, r=0.05, sigma=0.2,
                                   option_type=ot, barrier_type=bt)
                result = bo.price()
                assert result.price >= 0, f"Negative price for {bt} {ot}"

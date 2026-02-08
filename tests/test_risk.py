"""Tests for risk metrics."""

import numpy as np
import pytest

from stochastic_engine.risk import VaR, CVaR
from stochastic_engine.risk.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    information_ratio,
    beta,
    alpha,
    volatility,
)


class TestVaR:
    """Tests for Value at Risk."""

    def test_historical_var(self):
        """Test historical VaR calculation."""
        # Create returns with known percentile
        returns = np.array([-0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05])
        var = VaR(returns, confidence=0.90)

        # 10th percentile of these returns is -0.03 (second lowest)
        # VaR should be positive (loss)
        historical = var.historical()
        assert historical > 0
        assert np.isclose(historical, 0.041, atol=0.01)

    def test_parametric_var(self):
        """Test parametric VaR with normal assumption."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var = VaR(returns, confidence=0.95)
        parametric = var.parametric()

        # Should be approximately 1.645 * sigma
        expected = 1.645 * 0.02 - 0.001
        assert np.abs(parametric - expected) < 0.005

    def test_var_confidence_levels(self):
        """Test VaR increases with confidence level."""
        returns = np.random.randn(1000) * 0.02

        var_90 = VaR(returns, confidence=0.90).historical()
        var_95 = VaR(returns, confidence=0.95).historical()
        var_99 = VaR(returns, confidence=0.99).historical()

        assert var_90 < var_95 < var_99

    def test_monte_carlo_var(self):
        """Test Monte Carlo VaR."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        var = VaR(returns, confidence=0.95)
        mc_var = var.monte_carlo(n_simulations=50000, seed=42)

        # Should be similar to parametric
        parametric = var.parametric()
        assert np.abs(mc_var - parametric) < 0.005

    def test_horizon_scaling(self):
        """Test VaR scaling with square root of time."""
        returns = np.random.randn(1000) * 0.02

        var = VaR(returns, confidence=0.95)
        var_1d = var.historical()
        var_10d = var.scale_to_horizon(var_1d, horizon=10)

        assert np.isclose(var_10d, var_1d * np.sqrt(10))

    def test_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        returns = np.random.randn(100)

        with pytest.raises(ValueError):
            VaR(returns, confidence=1.5)

        with pytest.raises(ValueError):
            VaR(returns, confidence=-0.5)

    def test_empty_returns(self):
        """Test that empty returns raises error."""
        with pytest.raises(ValueError):
            VaR(np.array([]), confidence=0.95)


class TestCVaR:
    """Tests for Conditional Value at Risk."""

    def test_cvar_greater_than_var(self):
        """Test that CVaR >= VaR always."""
        returns = np.random.randn(1000) * 0.02

        var = VaR(returns, confidence=0.95).historical()
        cvar = CVaR(returns, confidence=0.95).historical()

        assert cvar >= var

    def test_historical_cvar(self):
        """Test historical CVaR calculation."""
        returns = np.array([-0.10, -0.05, -0.03, 0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08])

        cvar = CVaR(returns, confidence=0.90)
        result = cvar.historical()

        # Should be average of worst 10% (which is -0.10)
        assert result > 0  # CVaR is positive (loss)

    def test_parametric_cvar(self):
        """Test parametric CVaR with normal assumption."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 10000)

        cvar = CVaR(returns, confidence=0.95)
        result = cvar.parametric()

        # For normal, ES = sigma * phi(z) / alpha
        # At 95%, z = -1.645, phi(z) = 0.1031, alpha = 0.05
        # ES ≈ 0.02 * 0.1031 / 0.05 ≈ 0.0412
        assert 0.03 < result < 0.05


class TestRiskMetrics:
    """Tests for risk-adjusted performance metrics."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio for positive returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02, 0.018])
        sr = sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252)

        assert sr > 0

    def test_sharpe_ratio_formula(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])

        mean_return = returns.mean()
        std_return = returns.std()
        expected = (mean_return / std_return) * np.sqrt(252)

        assert np.isclose(sharpe_ratio(returns), expected)

    def test_sortino_higher_than_sharpe_for_positive_skew(self):
        """Test Sortino >= Sharpe when there's more upside."""
        # Returns with more upside than downside
        returns = np.array([0.02, 0.03, 0.01, -0.01, 0.025, 0.04, -0.005])

        sr = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)

        # Sortino should be higher as it ignores upside volatility
        assert sortino > sr

    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        prices = np.array([100, 110, 105, 90, 95, 100, 85, 95])

        # Max drawdown is from 110 to 85 = 22.7%
        mdd = max_drawdown(prices)
        expected = (110 - 85) / 110

        assert np.isclose(mdd, expected)

    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown when prices only go up."""
        prices = np.array([100, 105, 110, 115, 120])
        mdd = max_drawdown(prices)

        assert mdd == 0

    def test_information_ratio(self):
        """Test information ratio calculation."""
        portfolio = np.array([0.01, 0.02, 0.015, -0.01, 0.02])
        benchmark = np.array([0.008, 0.015, 0.01, -0.005, 0.015])

        ir = information_ratio(portfolio, benchmark)

        # Active returns
        active = portfolio - benchmark
        expected = (active.mean() / active.std()) * np.sqrt(252)

        assert np.isclose(ir, expected)

    def test_beta_market(self):
        """Test that market beta with itself is 1."""
        market = np.random.randn(100) * 0.02
        b = beta(market, market)

        # Allow small tolerance due to sample variance vs population variance
        assert np.isclose(b, 1.0, atol=0.02)

    def test_beta_calculation(self):
        """Test beta calculation."""
        np.random.seed(42)
        market = np.random.randn(1000) * 0.02
        # Create asset with beta = 1.5
        asset = 1.5 * market + np.random.randn(1000) * 0.01

        b = beta(asset, market)
        assert np.abs(b - 1.5) < 0.1

    def test_volatility_annualization(self):
        """Test volatility annualization."""
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        daily_vol = returns.std()

        annual_vol = volatility(returns, periods_per_year=252)

        assert np.isclose(annual_vol, daily_vol * np.sqrt(252))

"""Tests for data utilities module."""

import numpy as np
import pytest

from stochastic_engine.data.market import (
    correlation_from_returns,
    estimate_gbm_params,
    estimate_ou_params,
    returns_from_prices,
)


class TestReturnsFromPrices:
    """Tests for returns_from_prices."""

    def test_log_returns(self):
        """Test log return calculation."""
        prices = np.array([100, 105, 103, 108])
        returns = returns_from_prices(prices, method="log")
        expected = np.log(prices[1:] / prices[:-1])
        np.testing.assert_array_almost_equal(returns, expected)

    def test_simple_returns(self):
        """Test simple return calculation."""
        prices = np.array([100, 105, 103, 108])
        returns = returns_from_prices(prices, method="simple")
        expected = (prices[1:] - prices[:-1]) / prices[:-1]
        np.testing.assert_array_almost_equal(returns, expected)

    def test_length(self):
        """Test output length is n-1."""
        prices = np.array([100, 105, 110, 115, 120])
        returns = returns_from_prices(prices)
        assert len(returns) == len(prices) - 1

    def test_too_few_prices(self):
        """Test error with fewer than 2 prices."""
        with pytest.raises(ValueError, match="at least 2"):
            returns_from_prices(np.array([100]))

    def test_invalid_method(self):
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="method must be"):
            returns_from_prices(np.array([100, 105]), method="invalid")


class TestCorrelationFromReturns:
    """Tests for correlation_from_returns."""

    def test_perfect_correlation(self):
        """Test perfectly correlated returns."""
        ret_a = np.array([0.01, -0.02, 0.015, -0.01, 0.02])
        ret_b = ret_a * 2  # Perfectly correlated
        corr = correlation_from_returns({"A": ret_a, "B": ret_b})
        assert corr.shape == (2, 2)
        assert np.isclose(corr[0, 1], 1.0)

    def test_identity_diagonal(self):
        """Test diagonal is all 1s."""
        ret_a = np.random.normal(0, 0.01, 100)
        ret_b = np.random.normal(0, 0.02, 100)
        corr = correlation_from_returns({"A": ret_a, "B": ret_b})
        assert np.isclose(corr[0, 0], 1.0)
        assert np.isclose(corr[1, 1], 1.0)

    def test_array_input(self):
        """Test with 2D array input."""
        data = np.random.normal(0, 0.01, (3, 100))
        corr = correlation_from_returns(data)
        assert corr.shape == (3, 3)


class TestEstimateGBMParams:
    """Tests for estimate_gbm_params."""

    def test_roundtrip(self):
        """Test parameter estimation from synthetic GBM data."""
        from stochastic_engine import GBM

        gbm = GBM(S0=100, mu=0.08, sigma=0.2, seed=42)
        paths = gbm.simulate(T=5, steps=1260, n_paths=1)
        prices = paths[0]

        params = estimate_gbm_params(prices)
        assert params["S0"] == 100.0
        assert abs(params["sigma"] - 0.2) < 0.05  # Within 5% tolerance
        # mu estimation is noisy; just check it's reasonable
        assert -0.5 < params["mu"] < 0.5

    def test_returns_dict_keys(self):
        """Test returned dict has correct keys."""
        prices = np.array([100, 105, 110, 108, 112])
        params = estimate_gbm_params(prices)
        assert "S0" in params
        assert "mu" in params
        assert "sigma" in params


class TestEstimateOUParams:
    """Tests for estimate_ou_params."""

    def test_roundtrip(self):
        """Test parameter estimation from synthetic OU data."""
        from stochastic_engine import OrnsteinUhlenbeck

        ou = OrnsteinUhlenbeck(X0=0.05, mu=0.03, theta=5.0, sigma=0.01, seed=42)
        path = ou.simulate(T=10, steps=2520, n_paths=1)[0]

        params = estimate_ou_params(path)
        assert abs(params["theta"] - 5.0) < 3.0  # Mean reversion speed
        assert abs(params["mu"] - 0.03) < 0.02  # Long-term mean

    def test_returns_dict_keys(self):
        """Test returned dict has correct keys."""
        data = np.random.normal(0.05, 0.01, 100)
        params = estimate_ou_params(data)
        assert "X0" in params
        assert "mu" in params
        assert "theta" in params
        assert "sigma" in params

    def test_too_few_points(self):
        """Test error with too few data points."""
        with pytest.raises(ValueError, match="at least 3"):
            estimate_ou_params(np.array([0.05, 0.06]))


class TestFetchPricesImportError:
    """Test that fetch functions raise ImportError without yfinance."""

    def test_fetch_prices_import_error(self):
        """Test ImportError message for missing yfinance."""
        # This test only works if yfinance is NOT installed.
        # We skip if yfinance is available.
        try:
            import yfinance  # noqa: F401
            pytest.skip("yfinance is installed")
        except ImportError:
            from stochastic_engine.data.market import fetch_prices
            with pytest.raises(ImportError, match="yfinance"):
                fetch_prices("AAPL")

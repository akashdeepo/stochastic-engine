"""Tests for stochastic processes."""

import numpy as np
import pytest

from stochastic_engine.processes import GBM, OrnsteinUhlenbeck


class TestGBM:
    """Tests for Geometric Brownian Motion."""

    def test_init(self):
        """Test GBM initialization."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2)
        assert gbm.S0 == 100
        assert gbm.mu == 0.05
        assert gbm.sigma == 0.2

    def test_init_invalid_S0(self):
        """Test that negative S0 raises error."""
        with pytest.raises(ValueError, match="positive"):
            GBM(S0=-100, mu=0.05, sigma=0.2)

    def test_init_invalid_sigma(self):
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            GBM(S0=100, mu=0.05, sigma=-0.2)

    def test_simulate_shape(self):
        """Test that simulate returns correct shape."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
        paths = gbm.simulate(T=1, steps=252, n_paths=100)

        assert paths.shape == (100, 253)  # n_paths x (steps + 1)

    def test_simulate_starts_at_S0(self):
        """Test that all paths start at S0."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
        paths = gbm.simulate(T=1, steps=252, n_paths=100)

        np.testing.assert_array_equal(paths[:, 0], 100)

    def test_simulate_exact_vs_euler(self):
        """Test that exact and Euler methods give similar results."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)

        gbm.reset_rng(42)
        paths_exact = gbm.simulate(T=1, steps=252, n_paths=1000, method="exact")

        gbm.reset_rng(42)
        paths_euler = gbm.simulate(T=1, steps=252, n_paths=1000, method="euler")

        # Terminal values should be close (not exact due to discretization)
        mean_exact = paths_exact[:, -1].mean()
        mean_euler = paths_euler[:, -1].mean()

        assert np.abs(mean_exact - mean_euler) / mean_exact < 0.02  # Within 2%

    def test_simulate_reproducibility(self):
        """Test that same seed gives same results."""
        gbm1 = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
        gbm2 = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)

        paths1 = gbm1.simulate(T=1, steps=100, n_paths=10)
        paths2 = gbm2.simulate(T=1, steps=100, n_paths=10)

        np.testing.assert_array_equal(paths1, paths2)

    def test_sample(self):
        """Test sample method."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
        samples = gbm.sample(t=1, n_samples=10000)

        assert samples.shape == (10000,)
        assert samples.min() > 0  # GBM always positive

    def test_mean(self):
        """Test analytical mean."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2)

        # E[S_1] = S_0 * exp(mu * t)
        expected = 100 * np.exp(0.05)
        assert np.isclose(gbm.mean(1), expected)

    def test_variance(self):
        """Test analytical variance."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2)

        # Var[S_1] = S_0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)
        expected = 100**2 * np.exp(2 * 0.05) * (np.exp(0.2**2) - 1)
        assert np.isclose(gbm.variance(1), expected)

    def test_sample_matches_moments(self):
        """Test that sampled values match theoretical moments."""
        gbm = GBM(S0=100, mu=0.05, sigma=0.2, seed=42)
        samples = gbm.sample(t=1, n_samples=100000)

        # Sample mean should be close to theoretical mean
        assert np.abs(samples.mean() - gbm.mean(1)) / gbm.mean(1) < 0.01


class TestOrnsteinUhlenbeck:
    """Tests for Ornstein-Uhlenbeck process."""

    def test_init(self):
        """Test OU initialization."""
        ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02)
        assert ou.X0 == 0.1
        assert ou.mu == 0.05
        assert ou.theta == 0.5
        assert ou.sigma == 0.02

    def test_init_invalid_theta(self):
        """Test that non-positive theta raises error."""
        with pytest.raises(ValueError, match="positive"):
            OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=-0.5, sigma=0.02)

    def test_simulate_shape(self):
        """Test that simulate returns correct shape."""
        ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02, seed=42)
        paths = ou.simulate(T=10, steps=1000, n_paths=50)

        assert paths.shape == (50, 1001)

    def test_simulate_starts_at_X0(self):
        """Test that all paths start at X0."""
        ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02, seed=42)
        paths = ou.simulate(T=10, steps=1000, n_paths=50)

        np.testing.assert_array_equal(paths[:, 0], 0.1)

    def test_mean_reversion(self):
        """Test that process mean-reverts to mu."""
        ou = OrnsteinUhlenbeck(X0=0.2, mu=0.05, theta=1.0, sigma=0.01, seed=42)
        paths = ou.simulate(T=10, steps=1000, n_paths=100)

        # At t=10 with theta=1, should be very close to mu
        terminal_mean = paths[:, -1].mean()
        assert np.abs(terminal_mean - 0.05) < 0.01

    def test_half_life(self):
        """Test half-life calculation."""
        ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02)
        expected = np.log(2) / 0.5
        assert np.isclose(ou.half_life, expected)

    def test_mean(self):
        """Test analytical mean."""
        ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02)

        # At t=0, mean is X0
        assert np.isclose(ou.mean(0), 0.1)

        # E[X_t] = mu + (X0 - mu) * exp(-theta * t)
        t = 2
        expected = 0.05 + (0.1 - 0.05) * np.exp(-0.5 * t)
        assert np.isclose(ou.mean(t), expected)

    def test_variance(self):
        """Test analytical variance."""
        ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02)

        # At t=0, variance is 0
        assert np.isclose(ou.variance(0), 0)

        # Var[X_t] = (sigma^2 / 2*theta) * (1 - exp(-2*theta*t))
        t = 2
        expected = (0.02**2 / (2 * 0.5)) * (1 - np.exp(-2 * 0.5 * t))
        assert np.isclose(ou.variance(t), expected)

    def test_long_term_variance(self):
        """Test long-term variance."""
        ou = OrnsteinUhlenbeck(X0=0.1, mu=0.05, theta=0.5, sigma=0.02)
        expected = 0.02**2 / (2 * 0.5)
        assert np.isclose(ou.long_term_variance, expected)

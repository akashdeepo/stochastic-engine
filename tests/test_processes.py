"""Tests for stochastic processes."""

import numpy as np
import pytest

from stochastic_engine.processes import (
    CIR,
    GBM,
    OrnsteinUhlenbeck,
    Vasicek,
    CorrelatedGBM,
    MertonJumpDiffusion,
)


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


class TestVasicek:
    """Tests for Vasicek interest rate model."""

    def test_init(self):
        """Test initialization."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
        assert v.r0 == 0.05
        assert v.kappa == 0.5

    def test_init_invalid_kappa(self):
        """Test that non-positive kappa raises error."""
        with pytest.raises(ValueError, match="positive"):
            Vasicek(r0=0.05, kappa=-0.5, theta=0.03, sigma=0.01)

    def test_simulate_shape(self):
        """Test output shape."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01, seed=42)
        paths = v.simulate(T=5, steps=500, n_paths=50)
        assert paths.shape == (50, 501)

    def test_simulate_starts_at_r0(self):
        """Test paths start at r0."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01, seed=42)
        paths = v.simulate(T=5, steps=500, n_paths=50)
        np.testing.assert_array_almost_equal(paths[:, 0], 0.05)

    def test_mean_reversion(self):
        """Test that rates mean-revert to theta."""
        v = Vasicek(r0=0.10, kappa=1.0, theta=0.03, sigma=0.005, seed=42)
        paths = v.simulate(T=20, steps=5000, n_paths=200)
        terminal_mean = paths[:, -1].mean()
        assert abs(terminal_mean - 0.03) < 0.005

    def test_mean_formula(self):
        """Test analytical mean."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
        t = 2.0
        expected = 0.03 + (0.05 - 0.03) * np.exp(-0.5 * t)
        assert np.isclose(v.mean(t), expected)

    def test_bond_price_at_zero(self):
        """Test bond price at T=0 is 1."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
        assert np.isclose(v.bond_price(T=0), 1.0, atol=1e-10)

    def test_bond_price_positive(self):
        """Test bond price is in (0, 1]."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
        for T in [0.5, 1, 2, 5, 10]:
            p = v.bond_price(T)
            assert 0 < p <= 1

    def test_yield_curve(self):
        """Test yield curve returns correct shape."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01)
        mats = np.array([0.5, 1, 2, 5, 10])
        yields = v.yield_curve(mats)
        assert yields.shape == (5,)
        assert all(y > 0 for y in yields)

    def test_exact_vs_euler(self):
        """Test that exact and Euler give similar results."""
        v = Vasicek(r0=0.05, kappa=0.5, theta=0.03, sigma=0.01, seed=42)
        v.reset_rng(42)
        paths_exact = v.simulate(T=5, steps=1000, n_paths=500, method="exact")
        v.reset_rng(42)
        paths_euler = v.simulate(T=5, steps=1000, n_paths=500, method="euler")
        assert abs(paths_exact[:, -1].mean() - paths_euler[:, -1].mean()) < 0.005


class TestCIR:
    """Tests for Cox-Ingersoll-Ross interest rate model."""

    def test_init(self):
        """Test initialization."""
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
        assert cir.r0 == 0.05

    def test_init_invalid_r0(self):
        """Test that negative r0 raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            CIR(r0=-0.01, kappa=0.5, theta=0.03, sigma=0.05)

    def test_feller_condition(self):
        """Test Feller condition check."""
        # 2*0.5*0.03 = 0.03 > 0.05^2 = 0.0025 -> satisfied
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
        assert cir.feller_satisfied

        # 2*0.5*0.001 = 0.001 < 0.5^2 = 0.25 -> not satisfied
        cir2 = CIR(r0=0.05, kappa=0.5, theta=0.001, sigma=0.5)
        assert not cir2.feller_satisfied

    def test_simulate_non_negative(self):
        """Test that paths stay non-negative."""
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05, seed=42)
        paths = cir.simulate(T=10, steps=2520, n_paths=100)
        assert (paths >= 0).all()

    def test_simulate_shape(self):
        """Test output shape."""
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05, seed=42)
        paths = cir.simulate(T=5, steps=500, n_paths=50)
        assert paths.shape == (50, 501)

    def test_mean_formula(self):
        """Test analytical mean."""
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
        t = 2.0
        expected = 0.03 + (0.05 - 0.03) * np.exp(-0.5 * t)
        assert np.isclose(cir.mean(t), expected)

    def test_bond_price_positive(self):
        """Test bond price is in (0, 1]."""
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
        for T in [0.5, 1, 2, 5, 10]:
            p = cir.bond_price(T)
            assert 0 < p <= 1

    def test_yield_curve(self):
        """Test yield curve."""
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05)
        mats = np.array([0.5, 1, 2, 5])
        yields = cir.yield_curve(mats)
        assert yields.shape == (4,)

    def test_exact_simulation(self):
        """Test exact simulation with non-central chi-sq."""
        cir = CIR(r0=0.05, kappa=0.5, theta=0.03, sigma=0.05, seed=42)
        paths = cir.simulate(T=5, steps=500, n_paths=200, method="exact")
        assert (paths >= 0).all()
        # Mean should approach theta
        assert abs(paths[:, -1].mean() - 0.03) < 0.01


class TestMertonJumpDiffusion:
    """Tests for Merton Jump-Diffusion model."""

    def test_init(self):
        """Test initialization."""
        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=3, mu_j=-0.02, sigma_j=0.1)
        assert mjd.S0 == 100
        assert mjd.lam == 3

    def test_init_invalid_S0(self):
        """Test that non-positive S0 raises error."""
        with pytest.raises(ValueError, match="positive"):
            MertonJumpDiffusion(S0=-100, mu=0.05, sigma=0.2, lam=3, mu_j=-0.02, sigma_j=0.1)

    def test_simulate_shape(self):
        """Test output shape."""
        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=3,
                                   mu_j=-0.02, sigma_j=0.1, seed=42)
        paths = mjd.simulate(T=1, steps=252, n_paths=50)
        assert paths.shape == (50, 253)

    def test_starts_at_S0(self):
        """Test paths start at S0."""
        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=3,
                                   mu_j=-0.02, sigma_j=0.1, seed=42)
        paths = mjd.simulate(T=1, steps=252, n_paths=50)
        np.testing.assert_array_almost_equal(paths[:, 0], 100)

    def test_zero_jumps_equals_gbm(self):
        """Test that lam=0 gives GBM-like behavior."""
        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=0,
                                   mu_j=0, sigma_j=0, seed=42)
        paths = mjd.simulate(T=1, steps=252, n_paths=5000)
        # Mean should be close to GBM mean
        expected_mean = 100 * np.exp(0.05)
        assert abs(paths[:, -1].mean() - expected_mean) / expected_mean < 0.02

    def test_mean(self):
        """Test analytical mean."""
        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=3,
                                   mu_j=-0.02, sigma_j=0.1)
        expected = 100 * np.exp(0.05)
        assert np.isclose(mjd.mean(1), expected)

    def test_merton_call_positive(self):
        """Test that call price is positive."""
        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=3,
                                   mu_j=-0.02, sigma_j=0.1)
        price = mjd.price_european_call(K=100, T=1, r=0.05)
        assert price > 0

    def test_merton_call_vs_bs_no_jumps(self):
        """Test that Merton call equals BS call when lam=0."""
        from stochastic_engine.pricing.black_scholes import bs_call

        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=0,
                                   mu_j=0, sigma_j=0.01)
        merton_price = mjd.price_european_call(K=105, T=1, r=0.05)
        bs_price = bs_call(100, 105, 1, 0.05, 0.2)
        assert np.isclose(merton_price, bs_price, atol=0.01)

    def test_put_call_parity(self):
        """Test put-call parity."""
        mjd = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lam=3,
                                   mu_j=-0.02, sigma_j=0.1)
        call = mjd.price_european_call(K=100, T=1, r=0.05)
        put = mjd.price_european_put(K=100, T=1, r=0.05)
        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = 100 - 100 * np.exp(-0.05)
        assert np.isclose(lhs, rhs, atol=0.1)


class TestCorrelatedGBM:
    """Tests for Correlated Multi-Asset GBM."""

    def test_init(self):
        """Test initialization."""
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        cgbm = CorrelatedGBM(
            S0=np.array([100, 50]),
            mu=np.array([0.05, 0.08]),
            sigma=np.array([0.2, 0.3]),
            correlation=corr,
        )
        assert cgbm.n_assets == 2

    def test_invalid_correlation_not_symmetric(self):
        """Test error for asymmetric correlation matrix."""
        corr = np.array([[1.0, 0.6], [0.3, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            CorrelatedGBM(
                S0=np.array([100, 50]),
                mu=np.array([0.05, 0.08]),
                sigma=np.array([0.2, 0.3]),
                correlation=corr,
            )

    def test_invalid_correlation_diagonal(self):
        """Test error for non-unit diagonal."""
        corr = np.array([[0.9, 0.6], [0.6, 1.0]])
        with pytest.raises(ValueError, match="diagonal"):
            CorrelatedGBM(
                S0=np.array([100, 50]),
                mu=np.array([0.05, 0.08]),
                sigma=np.array([0.2, 0.3]),
                correlation=corr,
            )

    def test_simulate_shape(self):
        """Test output shape is (n_assets, n_paths, steps+1)."""
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        cgbm = CorrelatedGBM(
            S0=np.array([100, 50]),
            mu=np.array([0.05, 0.08]),
            sigma=np.array([0.2, 0.3]),
            correlation=corr,
            seed=42,
        )
        paths = cgbm.simulate(T=1, steps=252, n_paths=100)
        assert paths.shape == (2, 100, 253)

    def test_starts_at_S0(self):
        """Test all paths start at respective S0."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        cgbm = CorrelatedGBM(
            S0=np.array([100, 50]),
            mu=np.array([0.05, 0.08]),
            sigma=np.array([0.2, 0.3]),
            correlation=corr,
            seed=42,
        )
        paths = cgbm.simulate(T=1, steps=252, n_paths=100)
        np.testing.assert_array_almost_equal(paths[0, :, 0], 100)
        np.testing.assert_array_almost_equal(paths[1, :, 0], 50)

    def test_positive_correlation(self):
        """Test that positively correlated assets move together."""
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        cgbm = CorrelatedGBM(
            S0=np.array([100, 100]),
            mu=np.array([0.05, 0.05]),
            sigma=np.array([0.2, 0.2]),
            correlation=corr,
            seed=42,
        )
        paths = cgbm.simulate(T=1, steps=252, n_paths=5000)
        # Log-returns should be highly correlated
        ret_a = np.log(paths[0, :, -1] / paths[0, :, 0])
        ret_b = np.log(paths[1, :, -1] / paths[1, :, 0])
        empirical_corr = np.corrcoef(ret_a, ret_b)[0, 1]
        assert empirical_corr > 0.8

    def test_identity_correlation(self):
        """Test that identity correlation gives independent assets."""
        corr = np.eye(2)
        cgbm = CorrelatedGBM(
            S0=np.array([100, 100]),
            mu=np.array([0.05, 0.05]),
            sigma=np.array([0.2, 0.2]),
            correlation=corr,
            seed=42,
        )
        paths = cgbm.simulate(T=1, steps=252, n_paths=5000)
        ret_a = np.log(paths[0, :, -1] / paths[0, :, 0])
        ret_b = np.log(paths[1, :, -1] / paths[1, :, 0])
        empirical_corr = np.corrcoef(ret_a, ret_b)[0, 1]
        assert abs(empirical_corr) < 0.1

    def test_mean(self):
        """Test analytical mean for each asset."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        cgbm = CorrelatedGBM(
            S0=np.array([100, 50]),
            mu=np.array([0.05, 0.08]),
            sigma=np.array([0.2, 0.3]),
            correlation=corr,
        )
        means = cgbm.mean(1)
        assert np.isclose(means[0], 100 * np.exp(0.05))
        assert np.isclose(means[1], 50 * np.exp(0.08))

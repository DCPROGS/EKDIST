"""Tests for ekdist.stationarity — runs test and Cox-Lewis trend test."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from ekdist.stationarity import StationarityResult, runs_test, cox_lewis_test


# =========================================================================== #
# StationarityResult                                                            #
# =========================================================================== #

class TestStationarityResult:
    def test_has_required_fields(self):
        r = StationarityResult(
            statistic=1.23, p_value=0.45, n=100,
            name="Test", description="desc"
        )
        assert r.statistic == 1.23
        assert r.p_value == 0.45
        assert r.n == 100

    def test_repr_contains_name_and_statistic(self):
        r = StationarityResult(1.0, 0.3, 50, "My test", "OK")
        s = repr(r)
        assert "My test" in s
        assert "1.0" in s


# =========================================================================== #
# Runs test                                                                     #
# =========================================================================== #

class TestRunsTest:
    def test_returns_result_object(self):
        X = np.array([1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.8, 2.8, 1.3, 2.3])
        r = runs_test(X)
        assert isinstance(r, StationarityResult)

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="4"):
            runs_test(np.array([1.0, 2.0, 3.0]))

    def test_all_same_side_raises(self):
        with pytest.raises(ValueError):
            runs_test(np.ones(20))

    def test_p_value_in_range(self):
        rng = np.random.default_rng(0)
        X = rng.exponential(0.01, 200)
        r = runs_test(X)
        assert 0.0 <= r.p_value <= 1.0

    def test_n_matches_input_length(self):
        rng = np.random.default_rng(1)
        X = rng.exponential(0.01, 150)
        r = runs_test(X)
        assert r.n == 150

    def test_alternating_sequence_rejects_null(self):
        """Perfectly alternating [1,2,1,2,...] has far too many runs → significant."""
        X = np.tile([1.0, 2.0], 50)   # 100 elements
        r = runs_test(X)
        assert r.p_value < 0.05, "Alternating sequence should be non-stationary"

    def test_two_block_sequence_rejects_null(self):
        """[1,1,...,2,2,...] has only 2 runs → significant."""
        X = np.concatenate([np.ones(50), np.full(50, 2.0)])
        r = runs_test(X)
        assert r.p_value < 0.05

    def test_alternating_gives_positive_Z(self):
        """Too many runs → Z > 0."""
        X = np.tile([1.0, 2.0], 50)
        r = runs_test(X)
        assert r.statistic > 0

    def test_two_block_gives_negative_Z(self):
        """Too few runs → Z < 0."""
        X = np.concatenate([np.ones(50), np.full(50, 2.0)])
        r = runs_test(X)
        assert r.statistic < 0

    def test_known_statistic_alternating(self):
        """[1,2,1,2,1,2,1,2]: n1=n2=4, runs=8.

        mu_R = 2*4*4/8 + 1 = 5
        var_R = 2*4*4*(32-4-4) / (64*7) = 32*24/448 = 12/7
        Z = (8-5) / sqrt(12/7) ≈ 2.291
        """
        X = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        r = runs_test(X)
        expected_Z = 3.0 / math.sqrt(12.0 / 7.0)
        npt.assert_almost_equal(r.statistic, expected_Z, decimal=6)

    def test_known_statistic_two_block(self):
        """[1,1,1,1,2,2,2,2]: n1=n2=4, runs=2. Z = -expected_Z above."""
        X = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        r = runs_test(X)
        expected_Z = -3.0 / math.sqrt(12.0 / 7.0)
        npt.assert_almost_equal(r.statistic, expected_Z, decimal=6)

    def test_stationary_random_does_not_reject(self):
        """Large stationary exponential sample should not reject at alpha=0.01."""
        rng = np.random.default_rng(42)
        X = rng.exponential(0.05, 1000)
        r = runs_test(X)
        assert r.p_value > 0.01

    def test_description_contains_stationary_for_random(self):
        rng = np.random.default_rng(42)
        X = rng.exponential(0.05, 1000)
        r = runs_test(X)
        assert "Stationary" in r.description or "stationary" in r.description.lower()

    def test_description_non_stationary_for_trend(self):
        X = np.concatenate([np.ones(50), np.full(50, 2.0)])
        r = runs_test(X)
        assert "Non-stationary" in r.description or "non-stationary" in r.description.lower()


# =========================================================================== #
# Cox-Lewis trend test                                                          #
# =========================================================================== #

class TestCoxLewisTest:
    def test_returns_result_object(self):
        X = np.ones(20)
        r = cox_lewis_test(X)
        assert isinstance(r, StationarityResult)

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            cox_lewis_test(np.array([1.0]))

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            cox_lewis_test(np.zeros(10))

    def test_p_value_in_range(self):
        rng = np.random.default_rng(0)
        X = rng.exponential(0.01, 200)
        r = cox_lewis_test(X)
        assert 0.0 <= r.p_value <= 1.0

    def test_n_matches_input_length(self):
        X = np.ones(75)
        r = cox_lewis_test(X)
        assert r.n == 75

    def test_equal_intervals_gives_Z_zero(self):
        """Equal intervals → U = 0.5 exactly → Z = 0."""
        X = np.ones(100)
        r = cox_lewis_test(X)
        npt.assert_almost_equal(r.statistic, 0.0, decimal=10)

    def test_equal_intervals_gives_p_one(self):
        X = np.ones(100)
        r = cox_lewis_test(X)
        npt.assert_almost_equal(r.p_value, 1.0, decimal=10)

    def test_known_statistic_simple(self):
        """X = [4, 3, 2, 1] (n=4, decreasing = activity increasing).

        cumtimes = [4, 7, 9, 10], total = 10
        U = mean([4/10, 7/10, 9/10]) = 20/30 = 2/3
        Z = (2/3 - 0.5) * sqrt(12 * 3) = (1/6) * 6 = 1.0
        """
        X = np.array([4.0, 3.0, 2.0, 1.0])
        r = cox_lewis_test(X)
        npt.assert_almost_equal(r.statistic, 1.0, decimal=10)

    def test_increasing_intervals_gives_negative_Z(self):
        """Increasing intervals → events concentrated early → U < 0.5 → Z < 0."""
        X = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0] * 10)
        r = cox_lewis_test(X)
        assert r.statistic < 0

    def test_decreasing_intervals_gives_positive_Z(self):
        """Decreasing intervals → events concentrated late → U > 0.5 → Z > 0."""
        X = np.array([32.0, 16.0, 8.0, 4.0, 2.0, 1.0] * 10)
        r = cox_lewis_test(X)
        assert r.statistic > 0

    def test_strongly_increasing_rejects_null(self):
        """Truly monotone increasing intervals are detected as non-stationary.

        X = exp(linspace(0, 6, 200)) gives strongly increasing intervals; the
        Cox-Lewis statistic should be well below 0.5 (Z << 0, p < 0.05).
        """
        X = np.exp(np.linspace(0.0, 6.0, 200))
        r = cox_lewis_test(X)
        assert r.p_value < 0.05, f"Expected p < 0.05 for monotone increasing X, got {r.p_value}"

    def test_stationary_random_does_not_reject(self):
        """Large stationary sample should not reject at alpha=0.01."""
        rng = np.random.default_rng(99)
        X = rng.exponential(0.05, 2000)
        r = cox_lewis_test(X)
        assert r.p_value > 0.01

    def test_description_stationary_for_equal_intervals(self):
        X = np.ones(50)
        r = cox_lewis_test(X)
        assert "Stationary" in r.description or "stationary" in r.description.lower()

    def test_description_mentions_direction_when_significant(self):
        """Significant result should describe direction of trend."""
        X = np.array([32.0, 16.0, 8.0, 4.0, 2.0, 1.0] * 20)
        r = cox_lewis_test(X)
        if r.p_value < 0.05:
            # Should mention direction
            assert any(w in r.description.lower() for w in ["increas", "declin", "longer", "shorter"])

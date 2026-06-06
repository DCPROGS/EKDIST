"""Tests for ekdist.correlation — serial autocorrelation and open-shut correlation.

TDD: tests written before implementation.

Algorithm reference: Colquhoun lab CORCAL.FOR (EKDIST Fortran suite).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import norm as _norm

from ekdist.correlation import (
    CorrelationResult,
    SerialCorrelationResult,
    open_shut_correlation,
    serial_correlation,
)


# =========================================================================== #
# Helpers                                                                       #
# =========================================================================== #

def _make_positively_correlated(n: int, rho: float, seed: int = 0) -> np.ndarray:
    """AR(1) sequence: X[i] = rho * X[i-1] + noise, all shifted to be positive."""
    rng = np.random.default_rng(seed)
    X = np.zeros(n)
    X[0] = rng.exponential(1.0)
    for i in range(1, n):
        X[i] = rho * X[i - 1] + (1 - rho) * rng.exponential(1.0)
    return np.abs(X) + 0.001   # ensure all positive


def _arithmetic(n: int, start: float = 1.0) -> np.ndarray:
    """[start, start+1, ..., start+n-1]."""
    return np.arange(start, start + n, dtype=float)


# =========================================================================== #
# CorrelationResult                                                              #
# =========================================================================== #

class TestCorrelationResult:
    def test_fields_accessible(self):
        cr = CorrelationResult(r=0.3, z=1.2, df=10, n=15, p_value=0.23,
                               name="Test", description="OK")
        assert cr.r == pytest.approx(0.3)
        assert cr.z == pytest.approx(1.2)
        assert cr.df == 10
        assert cr.n == 15
        assert cr.p_value == pytest.approx(0.23)

    def test_repr_contains_name_and_r(self):
        cr = CorrelationResult(r=0.5, z=2.0, df=5, n=10, p_value=0.04,
                               name="Open-shut", description="significant")
        s = repr(cr)
        assert "Open-shut" in s
        assert "0.5" in s


# =========================================================================== #
# SerialCorrelationResult                                                        #
# =========================================================================== #

class TestSerialCorrelationResult:
    @pytest.fixture
    def result(self):
        rng = np.random.default_rng(7)
        X = rng.exponential(0.01, 500)
        return serial_correlation(X, max_lag=5)

    def test_lags_array_1_to_max_lag(self, result):
        npt.assert_array_equal(result.lags, np.arange(1, 6))

    def test_r_array_length(self, result):
        assert len(result.r) == 5

    def test_z_array_length(self, result):
        assert len(result.z) == 5

    def test_df_array_length(self, result):
        assert len(result.df) == 5

    def test_n_segments_array_length(self, result):
        assert len(result.n_segments) == 5

    def test_n_total_positive(self, result):
        assert result.n_total > 0

    def test_summary_is_string(self, result):
        s = result.summary()
        assert isinstance(s, str)
        assert "Lag" in s or "lag" in s.lower()

    def test_repr_is_same_as_summary(self, result):
        assert repr(result) == result.summary()


# =========================================================================== #
# serial_correlation — input validation                                          #
# =========================================================================== #

class TestSerialCorrelationValidation:
    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="5"):
            serial_correlation(np.ones(4))

    def test_exactly_5_does_not_raise(self):
        # 5 intervals with lag=1 gives 4 pairs → n_seg=4 > 3, one segment recorded
        serial_correlation(np.arange(1.0, 6.0), max_lag=1)

    def test_negative_outlier_limit_raises(self):
        with pytest.raises(ValueError, match="outlier_limit"):
            serial_correlation(np.ones(50), outlier_limit=-1.0)

    def test_zero_outlier_limit_raises(self):
        with pytest.raises(ValueError, match="outlier_limit"):
            serial_correlation(np.ones(50), outlier_limit=0.0)

    def test_default_max_lag_is_50(self):
        rng = np.random.default_rng(1)
        X = rng.exponential(0.01, 200)
        res = serial_correlation(X)
        assert res.lags[-1] == 50
        assert len(res.r) == 50


# =========================================================================== #
# serial_correlation — known values                                              #
# =========================================================================== #

class TestSerialCorrelationKnownValues:
    """Hand-computed reference values from the CORCAL algorithm.

    For X = [1, 2, 3, 4, 5, 6] at lag=1:
      ybar = 3.5,  ssy = 17.5
      pairs: (1,2),(2,3),(3,4),(4,5),(5,6)  → n_seg = 5
      sxy = (-2.5)(-1.5) + (-1.5)(-0.5) + (-0.5)(0.5) + (0.5)(1.5) + (1.5)(2.5)
          = 3.75 + 0.75 - 0.25 + 0.75 + 3.75 = 8.75
      r   = 8.75 / 17.5 = 0.5
      z_F = 0.5 * ln(1.5/0.5) = 0.5 * ln(3)
      df  = 5 - 3 = 2
      dev = z_F * sqrt(2)
      rbar = tanh(z_F) = 0.5  (since r maps exactly through Fisher back to 0.5)
    """

    X6 = _arithmetic(6)   # [1., 2., 3., 4., 5., 6.]

    def test_r_lag1_arithmetic(self):
        res = serial_correlation(self.X6, max_lag=1)
        npt.assert_almost_equal(res.r[0], 0.5, decimal=10)

    def test_df_lag1_arithmetic(self):
        res = serial_correlation(self.X6, max_lag=1)
        assert res.df[0] == 2

    def test_z_lag1_arithmetic(self):
        expected_z = 0.5 * math.log(3.0) * math.sqrt(2.0)
        res = serial_correlation(self.X6, max_lag=1)
        npt.assert_almost_equal(res.z[0], expected_z, decimal=10)

    def test_n_segments_lag1_no_outliers(self):
        """With no outliers the entire sequence is one segment."""
        res = serial_correlation(self.X6, max_lag=1)
        assert res.n_segments[0] == 1

    def test_lag2_too_few_pairs(self):
        """[1..6] lag=2 gives only 4 pairs (n=4); df=1."""
        res = serial_correlation(self.X6, max_lag=2)
        # lag 2: pairs (1,3),(2,4),(3,5),(4,6) → n=4, df=1
        # sxy = (-2.5)(-0.5)+(-1.5)(0.5)+(-0.5)(1.5)+(0.5)(2.5) = 1.25-0.75-0.75+1.25 = 1.0
        # r = 1.0/17.5 ≈ 0.05714
        npt.assert_almost_equal(res.r[1], 1.0 / 17.5, decimal=8)
        assert res.df[1] == 1


# =========================================================================== #
# serial_correlation — outlier handling                                          #
# =========================================================================== #

class TestSerialCorrelationOutliers:
    """Reference: X = [1,2,3,4,5,100,6,7,8,9,10], outlier_limit=50, lag=1.

    Non-outlier values: [1,2,3,4,5,6,7,8,9,10]  (100 excluded)
    ybar = 5.5,  ssy = 82.5

    Segment 1: pairs (1,2),(2,3),(3,4),(4,5) → n=4 before outlier at position 5
      sxy = (-4.5)(-3.5)+(-3.5)(-2.5)+(-2.5)(-1.5)+(-1.5)(-0.5) = 15.75+8.75+3.75+0.75 = 29.0
      r   = 29/82.5 ≈ 0.35152,  df = 1

    Segment 2: pairs (6,7),(7,8),(8,9),(9,10) → n=4
      sxy = (0.5)(1.5)+(1.5)(2.5)+(2.5)(3.5)+(3.5)(4.5) = 0.75+3.75+8.75+15.75 = 29.0
      r   = 29/82.5 ≈ 0.35152,  df = 1

    Pooled: total_df=2, zbar = z(r=29/82.5), dev = zbar*sqrt(2), rbar = tanh(zbar)
    """

    X = np.array([1., 2., 3., 4., 5., 100., 6., 7., 8., 9., 10.])
    LIMIT = 50.0

    @pytest.fixture
    def res(self):
        return serial_correlation(self.X, max_lag=1, outlier_limit=self.LIMIT)

    def test_n_segments_is_2(self, res):
        assert res.n_segments[0] == 2

    def test_total_df_is_2(self, res):
        assert res.df[0] == 2

    def test_pooled_r_value(self, res):
        r_seg = 29.0 / 82.5
        z_seg = 0.5 * math.log((1 + r_seg) / (1 - r_seg))
        expected_rbar = math.tanh(z_seg)   # same since both segments identical
        npt.assert_almost_equal(res.r[0], expected_rbar, decimal=10)

    def test_outlier_limit_stored(self, res):
        assert res.outlier_limit == self.LIMIT

    def test_n_total_excludes_outlier(self, res):
        """100 is above limit so not counted in n_total."""
        assert res.n_total == 10


# =========================================================================== #
# serial_correlation — statistical properties                                   #
# =========================================================================== #

class TestSerialCorrelationStatistical:
    def test_r_in_minus1_plus1(self):
        rng = np.random.default_rng(42)
        X = rng.exponential(0.01, 300)
        res = serial_correlation(X, max_lag=10)
        assert np.all(res.r >= -1.0) and np.all(res.r <= 1.0)

    def test_iid_exponential_small_r1(self):
        """i.i.d. exponential has zero true autocorrelation; sample |r₁| should be small."""
        rng = np.random.default_rng(123)
        X = rng.exponential(0.01, 2000)
        res = serial_correlation(X, max_lag=1)
        assert abs(res.r[0]) < 0.1, f"|r_1| = {abs(res.r[0]):.4f} too large for i.i.d. data"

    def test_positively_correlated_gives_positive_r1(self):
        """Strong positive AR(1) → r₁ > 0."""
        X = _make_positively_correlated(1000, rho=0.7, seed=5)
        res = serial_correlation(X, max_lag=1)
        assert res.r[0] > 0.2, f"r_1 = {res.r[0]:.4f} expected positive for AR(1) rho=0.7"

    def test_r_decays_with_lag_for_ar1(self):
        """For AR(1) process, |r_k| should decrease with lag."""
        X = _make_positively_correlated(2000, rho=0.8, seed=9)
        res = serial_correlation(X, max_lag=5)
        # r_1 > r_2 > ... is approximate; at least r_1 > r_5
        assert res.r[0] > res.r[4], (
            f"r_1={res.r[0]:.4f} not greater than r_5={res.r[4]:.4f} for AR(1)"
        )

    def test_df_lag1_approximately_n_minus_3(self):
        """Without outliers, one segment of n-1 pairs → df = (n-1) - 3 = n - 4."""
        X = _arithmetic(20)   # n=20, lag=1 → 19 pairs → df=16
        res = serial_correlation(X, max_lag=1)
        assert res.df[0] == 16


# =========================================================================== #
# open_shut_correlation — input validation                                      #
# =========================================================================== #

class TestOpenShutValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            open_shut_correlation(np.array([1., 2., 3.]), np.array([1., 2.]))

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="4"):
            open_shut_correlation(np.array([1., 2., 3.]), np.array([1., 2., 3.]))

    def test_exactly_4_does_not_raise(self):
        open_shut_correlation(np.ones(4), np.ones(4) * 2)


# =========================================================================== #
# open_shut_correlation — known values                                           #
# =========================================================================== #

class TestOpenShutCorrelationKnownValues:
    """Hand-computed Pearson r.

    open = [1, 2, 3, 4], shut = [1, 3, 2, 4]
      x̄ = 2.5,  ȳ = 2.5
      Σ(xi-x̄)(yi-ȳ) = (-1.5)(-1.5)+(-0.5)(0.5)+(0.5)(-0.5)+(1.5)(1.5)
                     = 2.25 - 0.25 - 0.25 + 2.25 = 4.0
      SSx = 5,  SSy = 5
      r = 4 / sqrt(25) = 0.8
      z_F = 0.5 * ln(1.8/0.2) = 0.5 * ln(9)
      df  = n-3 = 1
      dev = z_F * sqrt(1) = 0.5 * ln(9)
    """

    open_arr = np.array([1., 2., 3., 4.])
    shut_arr = np.array([1., 3., 2., 4.])
    EXPECTED_R = 0.8
    EXPECTED_Z = 0.5 * math.log(9.0)     # ≈ 1.0986
    EXPECTED_DF = 1

    def test_r_value(self):
        res = open_shut_correlation(self.open_arr, self.shut_arr)
        npt.assert_almost_equal(res.r, self.EXPECTED_R, decimal=10)

    def test_z_normal_deviate(self):
        res = open_shut_correlation(self.open_arr, self.shut_arr)
        npt.assert_almost_equal(res.z, self.EXPECTED_Z, decimal=10)

    def test_df(self):
        res = open_shut_correlation(self.open_arr, self.shut_arr)
        assert res.df == self.EXPECTED_DF

    def test_n(self):
        res = open_shut_correlation(self.open_arr, self.shut_arr)
        assert res.n == 4

    def test_p_value_in_range(self):
        res = open_shut_correlation(self.open_arr, self.shut_arr)
        assert 0.0 <= res.p_value <= 1.0

    def test_perfectly_correlated_gives_r_one(self):
        """open == shut → r = 1."""
        X = np.array([1., 2., 3., 4., 5.])
        res = open_shut_correlation(X, X.copy())
        npt.assert_almost_equal(res.r, 1.0, decimal=8)

    def test_perfectly_anticorrelated_gives_r_minus_one(self):
        """shut = -open (centred) → r = -1."""
        X = np.array([1., 2., 3., 4., 5.])
        Y = np.array([5., 4., 3., 2., 1.])
        res = open_shut_correlation(X, Y)
        npt.assert_almost_equal(res.r, -1.0, decimal=8)

    def test_positive_r_gives_positive_z(self):
        X = np.array([1., 2., 3., 4., 5.])
        res = open_shut_correlation(X, X.copy())
        assert res.z > 0

    def test_negative_r_gives_negative_z(self):
        X = np.array([1., 2., 3., 4., 5.])
        Y = np.array([5., 4., 3., 2., 1.])
        res = open_shut_correlation(X, Y)
        assert res.z < 0


# =========================================================================== #
# open_shut_correlation — statistical properties                                #
# =========================================================================== #

class TestOpenShutCorrelationStatistical:
    def test_iid_gives_small_r(self):
        """Independent open and shut → |r| should be small for large n."""
        rng = np.random.default_rng(99)
        op = rng.exponential(0.005, 500)
        sh = rng.exponential(0.010, 500)
        res = open_shut_correlation(op, sh)
        assert abs(res.r) < 0.15, f"|r| = {abs(res.r):.4f} too large for independent data"

    def test_returns_correlation_result(self):
        rng = np.random.default_rng(1)
        op = rng.exponential(0.005, 50)
        sh = rng.exponential(0.010, 50)
        res = open_shut_correlation(op, sh)
        assert isinstance(res, CorrelationResult)

    def test_name_mentions_open_shut(self):
        X = np.ones(10)
        res = open_shut_correlation(X, X * 2)
        assert "open" in res.name.lower() or "shut" in res.name.lower()

    def test_significant_positive_correlation_detected(self):
        """open ≈ shut + noise → p < 0.05 for large n."""
        rng = np.random.default_rng(77)
        base = rng.exponential(0.005, 200)
        noise = rng.normal(0, 0.0001, 200)
        sh = np.abs(base + noise)
        res = open_shut_correlation(base, sh)
        assert res.p_value < 0.05, f"Expected p < 0.05 for correlated data, got {res.p_value}"

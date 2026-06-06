"""Tests for ekdist.utils — rolling_mean and existing helpers."""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist.utils import rolling_mean, moving_average, filter_risetime


class TestRollingMean:
    """rolling_mean(x, w) — valid-only sliding mean, length len(x)-w+1."""

    def test_output_length(self):
        assert len(rolling_mean(np.ones(10), 3)) == 8   # 10 - 3 + 1

    def test_window_1_is_identity(self):
        x = np.array([1., 4., 2., 7.])
        npt.assert_array_equal(rolling_mean(x, 1), x)

    def test_window_equals_length_returns_one_element(self):
        x = np.array([1., 2., 3., 4.])
        result = rolling_mean(x, 4)
        assert result.shape == (1,)
        npt.assert_almost_equal(result[0], 2.5)

    def test_constant_array(self):
        x = np.full(20, 5.0)
        npt.assert_array_almost_equal(rolling_mean(x, 5), np.full(16, 5.0))

    def test_known_values_window3(self):
        x = np.array([1., 2., 3., 4., 5.])
        npt.assert_almost_equal(rolling_mean(x, 3), [2., 3., 4.])

    def test_known_values_window2(self):
        x = np.array([0., 2., 4., 6.])
        npt.assert_almost_equal(rolling_mean(x, 2), [1., 3., 5.])

    def test_arithmetic_sequence(self):
        """Rolling mean of [1..n] with window w equals [w/2+0.5, w/2+1.5, ...]."""
        n, w = 10, 4
        x = np.arange(1., n + 1.)
        result = rolling_mean(x, w)
        expected = np.array([np.mean(x[i:i+w]) for i in range(n - w + 1)])
        npt.assert_almost_equal(result, expected)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="[Ww]indow"):
            rolling_mean(np.ones(5), 0)

    def test_window_negative_raises(self):
        with pytest.raises(ValueError, match="[Ww]indow"):
            rolling_mean(np.ones(5), -1)

    def test_window_larger_than_array_raises(self):
        with pytest.raises(ValueError):
            rolling_mean(np.ones(5), 6)

    def test_returns_ndarray(self):
        result = rolling_mean(np.ones(10), 3)
        assert isinstance(result, np.ndarray)

    def test_accepts_list_input(self):
        result = rolling_mean([1., 2., 3., 4., 5.], 3)
        npt.assert_almost_equal(result, [2., 3., 4.])

    def test_large_array_matches_numpy_convolve(self):
        rng = np.random.default_rng(42)
        x = rng.exponential(1.0, 1000)
        w = 20
        result = rolling_mean(x, w)
        kernel = np.ones(w) / w
        expected = np.convolve(x, kernel, mode="valid")
        npt.assert_almost_equal(result, expected, decimal=12)


class TestMovingAverage:
    """Existing moving_average — preserve backward-compatibility."""

    def test_returns_same_length(self):
        x = np.ones(20)
        assert len(moving_average(x, 5)) == 20

    def test_constant_array_interior(self):
        """Interior (non-padded) values should equal the constant."""
        x = np.full(10, 3.0)
        result = moving_average(x, 4)
        npt.assert_almost_equal(result[4:], np.full(6, 3.0))

    def test_same_length_as_input(self):
        x = np.arange(10.0)
        assert len(moving_average(x, 3)) == len(x)


class TestFilterRisetime:
    def test_known_value(self):
        # FILTER_RISE_COEFF = 0.3321
        npt.assert_almost_equal(filter_risetime(1000.0), 0.3321 / 1000.0)

    def test_zero_fc_raises(self):
        with pytest.raises(ValueError):
            filter_risetime(0.0)

    def test_negative_fc_raises(self):
        with pytest.raises(ValueError):
            filter_risetime(-1000.0)

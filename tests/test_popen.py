"""Tests for ekdist.popen — global_popen and burst_popen."""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist.popen import PopenResult, global_popen, burst_popen


# =========================================================================== #
# PopenResult                                                                   #
# =========================================================================== #

class TestPopenResult:
    def test_has_required_fields(self):
        r = PopenResult(
            popen=0.5,
            ci_lower=0.4,
            ci_upper=0.6,
            n_bootstrap=1000,
            confidence=0.95,
            total_open_time=10.0,
            total_time=20.0,
        )
        assert r.popen == 0.5
        assert r.ci_lower == 0.4
        assert r.ci_upper == 0.6

    def test_repr_contains_popen(self):
        r = PopenResult(0.3, 0.25, 0.35, 500, 0.95, 6.0, 20.0)
        assert "0.3" in repr(r) or "Popen" in repr(r)

    def test_ci_ordered(self):
        r = PopenResult(0.3, 0.25, 0.35, 500, 0.95, 6.0, 20.0)
        assert r.ci_lower <= r.popen <= r.ci_upper


# =========================================================================== #
# global_popen                                                                  #
# =========================================================================== #

class TestGlobalPopen:
    def test_returns_popen_result(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=50, rng=np.random.default_rng(0))
        assert isinstance(result, PopenResult)

    def test_popen_between_zero_and_one(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=50, rng=np.random.default_rng(0))
        assert 0.0 < result.popen < 1.0

    def test_ci_ordered(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=50, rng=np.random.default_rng(0))
        assert result.ci_lower <= result.popen <= result.ci_upper

    def test_ci_lower_less_than_upper(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=200, rng=np.random.default_rng(1))
        assert result.ci_lower < result.ci_upper

    def test_n_bootstrap_stored(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=123, rng=np.random.default_rng(0))
        assert result.n_bootstrap == 123

    def test_confidence_stored(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=50, confidence=0.90,
                              rng=np.random.default_rng(0))
        assert result.confidence == 0.90

    def test_total_time_is_sum_of_resolved(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=10, rng=np.random.default_rng(0))
        opens = loaded_record.periods.open_intervals
        shuts = loaded_record.periods.shut_intervals
        expected_total = float(np.sum(opens) + np.sum(shuts))
        npt.assert_almost_equal(result.total_time, expected_total, decimal=6)

    def test_total_open_time_is_sum_of_open_periods(self, loaded_record):
        result = global_popen(loaded_record, n_bootstrap=10, rng=np.random.default_rng(0))
        expected_open = float(np.sum(loaded_record.periods.open_intervals))
        npt.assert_almost_equal(result.total_open_time, expected_open, decimal=6)

    def test_point_estimate_formula(self, loaded_record):
        """popen = total_open_time / total_time."""
        result = global_popen(loaded_record, n_bootstrap=10, rng=np.random.default_rng(0))
        npt.assert_almost_equal(
            result.popen,
            result.total_open_time / result.total_time,
            decimal=10,
        )

    def test_wider_ci_for_lower_confidence(self, loaded_record):
        r95 = global_popen(loaded_record, n_bootstrap=500, confidence=0.95,
                           rng=np.random.default_rng(7))
        r50 = global_popen(loaded_record, n_bootstrap=500, confidence=0.50,
                           rng=np.random.default_rng(7))
        assert (r95.ci_upper - r95.ci_lower) >= (r50.ci_upper - r50.ci_lower)

    def test_reproducible_with_same_rng(self, loaded_record):
        r1 = global_popen(loaded_record, n_bootstrap=100, rng=np.random.default_rng(42))
        r2 = global_popen(loaded_record, n_bootstrap=100, rng=np.random.default_rng(42))
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_known_popen_synthetic(self):
        """Synthetic record with known open/shut times → exact point estimate.

        np.tile([0.01, 0.04], 200) gives 200 opens and 200 shuts.
        impose_resolution trims the trailing shut, leaving 200 opens and 199 shuts.
        Popen = 200*0.01 / (200*0.01 + 199*0.04) = 2.0 / 9.96.
        """
        from ekdist.record import SingleChannelRecord
        n = 200
        itint = np.tile([0.01, 0.04], n)
        iampl = np.tile([1.0, 0.0], n)
        rec = SingleChannelRecord.from_intervals(itint, iampl)
        result = global_popen(rec, n_bootstrap=50, rng=np.random.default_rng(0))
        expected = (n * 0.01) / (n * 0.01 + (n - 1) * 0.04)
        npt.assert_almost_equal(result.popen, expected, decimal=10)

    def test_invalid_confidence_raises(self, loaded_record):
        with pytest.raises(ValueError, match="confidence"):
            global_popen(loaded_record, confidence=1.5)

    def test_invalid_n_bootstrap_raises(self, loaded_record):
        with pytest.raises(ValueError, match="n_bootstrap"):
            global_popen(loaded_record, n_bootstrap=0)

    def test_no_shut_intervals_raises(self):
        """A record with only open intervals has no denominator for Popen."""
        from ekdist.record import SingleChannelRecord
        rec = SingleChannelRecord.from_intervals(
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 1.0, 1.0]),
        )
        with pytest.raises(ValueError, match="shut"):
            global_popen(rec, n_bootstrap=10)


# =========================================================================== #
# burst_popen                                                                   #
# =========================================================================== #

class TestBurstPopen:
    @pytest.fixture
    def bursts_obj(self, loaded_record):
        from ekdist.bursts import Bursts
        return Bursts.from_periods(loaded_record.periods, tcrit=5e-3)

    def test_returns_popen_result(self, bursts_obj):
        result = burst_popen(bursts_obj, n_bootstrap=50, rng=np.random.default_rng(0))
        assert isinstance(result, PopenResult)

    def test_popen_between_zero_and_one(self, bursts_obj):
        result = burst_popen(bursts_obj, n_bootstrap=50, rng=np.random.default_rng(0))
        assert 0.0 < result.popen < 1.0

    def test_ci_ordered(self, bursts_obj):
        result = burst_popen(bursts_obj, n_bootstrap=100, rng=np.random.default_rng(0))
        assert result.ci_lower <= result.popen <= result.ci_upper

    def test_point_estimate_matches_mean_popen(self, bursts_obj):
        """point estimate must equal Bursts.mean_popen()."""
        result = burst_popen(bursts_obj, n_bootstrap=10, rng=np.random.default_rng(0))
        npt.assert_almost_equal(result.popen, bursts_obj.mean_popen(), decimal=10)

    def test_reproducible_with_same_rng(self, bursts_obj):
        r1 = burst_popen(bursts_obj, n_bootstrap=100, rng=np.random.default_rng(5))
        r2 = burst_popen(bursts_obj, n_bootstrap=100, rng=np.random.default_rng(5))
        assert r1.ci_lower == r2.ci_lower

    def test_too_few_bursts_raises(self):
        from ekdist.bursts import Bursts
        tiny = Bursts([np.array([0.01])], tcrit=1e-3)
        with pytest.raises(ValueError, match="burst"):
            burst_popen(tiny, n_bootstrap=100)

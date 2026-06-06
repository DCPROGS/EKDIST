"""Tests for ekdist.bursts.Bursts."""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist.bursts import Bursts
from ekdist.record import Periods


def _make_periods(intervals, amplitudes):
    """Helper: make a Periods object from lists."""
    rtint = np.array(intervals)
    rampl = np.array(amplitudes)
    rprop = np.zeros(len(rtint), dtype=int)
    return Periods(rtint, rampl, rprop)


@pytest.fixture
def simple_periods():
    """Two-burst record: open-shut-open-LONG_SHUT-open-shut.
    tcrit = 0.1 s separates the two bursts.

    Burst 1: open(50ms) + shut(20ms) + open(40ms)  → 3 intervals, 2 openings
    Burst 2: open(80ms) + shut(30ms)               → 2 intervals, 1 opening
    Long shut (200ms) between bursts is consumed by the split.
    """
    intervals = [0.05, 0.02, 0.04, 0.20, 0.08, 0.03]
    amplitudes = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    return _make_periods(intervals, amplitudes)


@pytest.fixture
def two_bursts(simple_periods):
    return Bursts.from_periods(simple_periods, tcrit=0.10)


class TestBurstsFromPeriods:
    def test_two_bursts_found(self, two_bursts):
        assert len(two_bursts) == 2

    def test_tcrit_stored(self, two_bursts):
        assert two_bursts.tcrit == pytest.approx(0.10)

    def test_each_burst_is_array(self, two_bursts):
        for b in two_bursts.bursts:
            assert isinstance(b, np.ndarray)


class TestPerBurstMetrics:
    def test_n_openings(self, two_bursts):
        ns = two_bursts.list_n_openings()
        assert ns == [2, 1]  # burst 0 has 2 openings, burst 1 has 1

    def test_total_open_time(self, two_bursts):
        tots = two_bursts.list_total_open_time()
        npt.assert_almost_equal(tots[0], 0.09, decimal=6)  # 50ms + 40ms
        npt.assert_almost_equal(tots[1], 0.08, decimal=6)

    def test_total_shut_time(self, two_bursts):
        shuts = two_bursts.list_total_shut_time()
        # Both bursts have within-burst shuts (20ms and 30ms)
        npt.assert_almost_equal(shuts[0], 0.02, decimal=6)
        npt.assert_almost_equal(shuts[1], 0.03, decimal=6)

    def test_total_duration(self, two_bursts):
        durs = two_bursts.list_duration()
        npt.assert_almost_equal(durs[0], 0.11, decimal=6)  # 50+20+40 ms
        npt.assert_almost_equal(durs[1], 0.11, decimal=6)  # 80+30 ms

    def test_popen_excl_last_single_opening_burst(self, two_bursts):
        # Burst 1 has 1 opening: popen_excl_last returns 1.0 by convention
        popens = two_bursts.list_popen()
        assert popens[1] == pytest.approx(1.0)

    def test_popen_excl_last_multi_opening_burst(self, two_bursts):
        # Burst 0: excl last open → open_excl=50ms / (50ms+20ms shut) = 50/70
        popens = two_bursts.list_popen()
        npt.assert_almost_equal(popens[0], 0.05 / 0.07, decimal=5)

    def test_popen_incl_last(self, two_bursts):
        popens = two_bursts.list_popen_incl_last()
        npt.assert_almost_equal(popens[0], 0.09 / 0.11, decimal=5)
        npt.assert_almost_equal(popens[1], 0.08 / 0.11, decimal=5)

    def test_mean_open_time(self, two_bursts):
        means = two_bursts.list_mean_open_time()
        npt.assert_almost_equal(means[0], 0.045, decimal=6)  # (50+40)/2 ms
        npt.assert_almost_equal(means[1], 0.08, decimal=6)

    def test_longest_opening(self, two_bursts):
        longs = two_bursts.list_longest_opening()
        npt.assert_almost_equal(longs[0], 0.05, decimal=6)
        npt.assert_almost_equal(longs[1], 0.08, decimal=6)

    def test_longest_shut(self, two_bursts):
        longs = two_bursts.list_longest_shut()
        npt.assert_almost_equal(longs[0], 0.02, decimal=6)
        npt.assert_almost_equal(longs[1], 0.03, decimal=6)


class TestAggregateMetrics:
    def test_mean_popen_not_nan(self, two_bursts):
        mp = two_bursts.mean_popen()
        assert not np.isnan(mp)

    def test_mean_n_openings(self, two_bursts):
        assert two_bursts.mean_n_openings() == pytest.approx(1.5)  # (2+1)/2

    def test_mean_duration(self, two_bursts):
        npt.assert_almost_equal(two_bursts.mean_duration(), 0.11, decimal=5)

    def test_mean_total_open_time(self, two_bursts):
        npt.assert_almost_equal(two_bursts.mean_total_open_time(), 0.085, decimal=5)


class TestFiltering:
    def test_filter_min_openings(self, two_bursts):
        """Keeping only bursts with >= 2 openings keeps burst 0 (2 openings)."""
        filtered = two_bursts.filter_min_openings(2)
        assert len(filtered) == 1

    def test_filter_max_open_time(self, two_bursts):
        """Exclude bursts whose longest opening exceeds 60ms → drops burst 1 (80ms)."""
        filtered = two_bursts.filter_max_open_time(0.06)
        assert len(filtered) == 1
        npt.assert_almost_equal(filtered.bursts[0][0], 0.05, decimal=6)

    def test_filter_min_duration(self, two_bursts):
        """Both bursts are 110ms; threshold at 120ms leaves none."""
        filtered = two_bursts.filter_min_duration(0.12)
        assert len(filtered) == 0

    def test_filter_returns_new_object(self, two_bursts):
        filtered = two_bursts.filter_min_openings(1)
        assert filtered is not two_bursts

    def test_filter_max_open_fixes_old_bug(self):
        """Original code had `.any <= longop` (AttributeError). Verify fix."""
        intervals = [0.05, 0.02, 0.20, 0.08, 0.03]
        amplitudes = [1.0, 0.0, 0.0, 1.0, 0.0]
        p = _make_periods(intervals, amplitudes)
        b = Bursts.from_periods(p, tcrit=0.10)
        # This must not raise AttributeError
        result = b.filter_max_open_time(1.0)
        assert len(result) == 2


class TestBurstsWithNOpenings:
    def test_bursts_with_one_opening(self, two_bursts):
        ones = two_bursts.bursts_with_n_openings(1)
        assert len(ones) == 1  # only burst 1

    def test_bursts_with_two_openings(self, two_bursts):
        twos = two_bursts.bursts_with_n_openings(2)
        assert len(twos) == 1  # only burst 0


class TestBurstsRepr:
    def test_repr_contains_key_fields(self, two_bursts):
        s = repr(two_bursts)
        assert "tcrit" in s
        assert "n=" in s


class TestBurstsOnRealRecord:
    def test_bursts_from_achsim(self, loaded_record):
        bursts = Bursts.from_periods(loaded_record.periods, tcrit=5e-3)
        assert len(bursts) > 0

    def test_mean_popen_in_range(self, loaded_record):
        bursts = Bursts.from_periods(loaded_record.periods, tcrit=5e-3)
        mp = bursts.mean_popen()
        assert 0.0 < mp < 1.0

    def test_all_burst_metrics_run(self, loaded_record):
        bursts = Bursts.from_periods(loaded_record.periods, tcrit=5e-3)
        assert len(bursts.list_n_openings()) == len(bursts)
        assert len(bursts.list_duration()) == len(bursts)
        assert len(bursts.list_total_open_time()) == len(bursts)
        assert len(bursts.list_mean_open_time()) == len(bursts)
        assert len(bursts.list_longest_opening()) == len(bursts)

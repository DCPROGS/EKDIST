"""Tests for ekdist.record.SingleChannelRecord and Periods."""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist.record import SingleChannelRecord, Periods


class TestSingleChannelRecordFromIntervals:
    def test_from_intervals_basic(self):
        itint = np.array([0.1, 0.05, 0.08, 0.03])
        iampl = np.array([1.0, 0.0, 1.0, 0.0])
        rec = SingleChannelRecord.from_intervals(itint, iampl)
        assert rec._is_loaded

    def test_default_flags_are_zero(self):
        itint = np.array([0.1, 0.05])
        iampl = np.array([1.0, 0.0])
        rec = SingleChannelRecord.from_intervals(itint, iampl)
        assert np.all(rec.iprop == 0)

    def test_tres_zero_by_default(self):
        itint = np.array([0.1, 0.05])
        iampl = np.array([1.0, 0.0])
        rec = SingleChannelRecord.from_intervals(itint, iampl)
        assert rec.tres == 0.0

    def test_setting_tres_updates_resolved(self):
        itint = np.array([0.05, 0.0001, 0.08, 0.04])
        iampl = np.array([1.0, 0.0, 1.0, 0.0])
        rec = SingleChannelRecord.from_intervals(itint, iampl)
        n_before = len(rec.rtint)
        rec.tres = 0.001
        n_after = len(rec.rtint)
        assert n_after <= n_before

    def test_negative_tres_raises(self):
        itint = np.array([0.1, 0.05])
        iampl = np.array([1.0, 0.0])
        rec = SingleChannelRecord.from_intervals(itint, iampl)
        with pytest.raises(ValueError):
            rec.tres = -1e-6

    def test_repr_runs(self):
        itint = np.array([0.1, 0.05, 0.08, 0.03])
        iampl = np.array([1.0, 0.0, 1.0, 0.0])
        rec = SingleChannelRecord.from_intervals(itint, iampl)
        s = repr(rec)
        assert "SCN" not in s or "intervals" in s

    def test_simulated_flag_stored(self):
        itint = np.array([0.1, 0.05])
        iampl = np.array([1.0, 0.0])
        rec = SingleChannelRecord.from_intervals(itint, iampl, is_simulated=True)
        assert rec._is_simulated is True


class TestSingleChannelRecordFromSCN:
    def test_from_scn_loads(self, scn_file):
        rec = SingleChannelRecord.from_scn(scn_file)
        assert rec._is_loaded
        assert len(rec.itint) > 0

    def test_scn_interval_count_matches_header(self, scn_file):
        rec = SingleChannelRecord.from_scn(scn_file)
        assert abs(len(rec.itint) - rec.header.n_intervals) <= 1

    def test_scn_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SingleChannelRecord.from_scn(tmp_path / "nonexistent.scn")

    def test_scn_tres_reduces_count(self, scn_file):
        rec = SingleChannelRecord.from_scn(scn_file)
        raw_n = len(rec.itint)
        rec.tres = 30e-6
        assert len(rec.rtint) < raw_n

    def test_periods_accessible_after_load(self, loaded_record):
        periods = loaded_record.periods
        assert len(periods.intervals) > 0

    def test_repr_contains_stats(self, loaded_record):
        s = repr(loaded_record)
        assert "Open periods" in s
        assert "Shut periods" in s


class TestMultipleFileLoading:
    def test_from_scn_files_single_path_loads(self, scn_file):
        rec = SingleChannelRecord.from_scn_files([scn_file])
        assert rec._is_loaded
        assert len(rec.itint) > 0

    def test_from_scn_files_two_files_doubles_interval_count(self, scn_file):
        """Concatenating a file with itself should double the raw interval count."""
        rec1 = SingleChannelRecord.from_scn(scn_file)
        rec2 = SingleChannelRecord.from_scn_files([scn_file, scn_file])
        assert abs(len(rec2.itint) - 2 * len(rec1.itint)) <= 2

    def test_from_scn_files_arrays_same_length(self, scn_file):
        rec = SingleChannelRecord.from_scn_files([scn_file, scn_file])
        assert len(rec.itint) == len(rec.iampl) == len(rec.iprop)

    def test_from_scn_files_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SingleChannelRecord.from_scn_files([])

    def test_from_scn_files_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SingleChannelRecord.from_scn_files([tmp_path / "nonexistent.scn"])

    def test_from_scn_files_origin_contains_filenames(self, scn_file):
        rec = SingleChannelRecord.from_scn_files([scn_file, scn_file])
        assert "AChsim" in rec._origin

    def test_from_scn_files_tres_applies_after_concat(self, scn_file):
        rec = SingleChannelRecord.from_scn_files([scn_file, scn_file])
        rec.tres = 30e-6
        assert len(rec.rtint) < len(rec.itint)

    def test_from_scn_files_periods_accessible(self, scn_file):
        rec = SingleChannelRecord.from_scn_files([scn_file, scn_file])
        rec.tres = 30e-6
        assert len(rec.periods.intervals) > 0

    def test_from_scn_files_header_is_from_first_file(self, scn_file):
        rec1 = SingleChannelRecord.from_scn(scn_file)
        rec2 = SingleChannelRecord.from_scn_files([scn_file, scn_file])
        assert rec2.header.version == rec1.header.version
        assert rec2.header.record_type == rec1.header.record_type


class TestPeriods:
    def setup_method(self):
        """Two open + two shut intervals (alternating)."""
        self.rtint = np.array([0.05, 0.02, 0.08, 0.03])
        self.rampl = np.array([1.0, 0.0, 1.0, 0.0])
        self.rprop = np.array([0, 0, 0, 0])

    def test_periods_built(self):
        p = Periods(self.rtint, self.rampl, self.rprop)
        assert len(p.intervals) > 0

    def test_open_intervals_slice(self):
        p = Periods(self.rtint, self.rampl, self.rprop)
        opens = p.open_intervals
        assert len(opens) >= 1
        assert np.all(opens > 0)

    def test_shut_intervals_slice(self):
        p = Periods(self.rtint, self.rampl, self.rprop)
        shuts = p.shut_intervals
        assert len(shuts) >= 1
        assert np.all(shuts > 0)

    def test_open_intervals_in_range(self):
        p = Periods(self.rtint, self.rampl, self.rprop)
        opens = p.open_intervals_in_range(0.04, 0.06)
        assert np.all(opens >= 0.04)
        assert np.all(opens <= 0.06)

    def test_empty_periods(self):
        p = Periods(np.array([]), np.array([]), np.array([]))
        assert len(p.intervals) == 0
        assert len(p.open_intervals) == 0
        assert len(p.shut_intervals) == 0

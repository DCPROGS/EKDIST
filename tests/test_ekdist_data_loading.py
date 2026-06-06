"""Data-loading tests — SCN reading via dcio, record construction via ekdist."""

import numpy as np
import numpy.testing as npt
import pytest

from dcio.formats.scn import read as scn_read
from ekdist.record import SingleChannelRecord
from ekdist.bursts import Bursts


class TestSCNFileLoading:
    """Read AChsim.scn (simulated, version -103) via dcio and verify basic properties."""

    @pytest.fixture(autouse=True)
    def load(self, scn_file):
        self.rec = scn_read(scn_file)

    def test_infile_exists(self, scn_file):
        assert scn_file.exists()

    def test_header_version(self):
        assert self.rec.header.version == -103

    def test_header_record_type(self):
        assert self.rec.header.record_type == "simulated"

    def test_interval_number(self):
        # Simulated files have a sentinel at the end that is stripped; allow ±1.
        assert abs(len(self.rec.intervals) - 13948) <= 1

    def test_intervals_same_length_as_amplitudes(self):
        assert len(self.rec.intervals) == len(self.rec.amplitudes)

    def test_flags_all_zero(self):
        # AChsim.scn: all flags are 0 (usable) except the stripped sentinel.
        assert not self.rec.flags[:-1].any()

    def test_first_interval_is_shut(self):
        assert self.rec.amplitudes[0] == 0.0

    def test_second_interval_is_open(self):
        # Raw ADC amplitude 6, multiplied by calfac2 → non-zero pA value.
        assert self.rec.amplitudes[1] != 0.0


class TestIntervalListLoading:
    def setup_method(self):
        self.intervals = np.array([20.0, 1.0, 19.0, 100.0, 10.0, 100.0, 1.0])
        self.amplitudes = np.array([5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0])
        self.rec = SingleChannelRecord.from_intervals(self.intervals, self.amplitudes)

    def test_original_number_intervals(self):
        assert len(self.rec.itint) == 7
        assert len(self.rec.iampl) == 7
        assert len(self.rec.iprop) == 7

    def test_imposing_resolution(self):
        self.rec.tres = 2.0
        assert len(self.rec.rtint) == 4
        assert len(self.rec.rampl) == 4
        assert len(self.rec.rprop) == 4

    def test_setting_periods(self):
        self.rec.tres = 2.0
        assert len(self.rec.periods.intervals) == 3
        assert len(self.rec.periods.amplitudes) == 3
        assert len(self.rec.periods.flags) == 3

    def test_burst_number(self):
        self.rec.tres = 2.0
        br = Bursts.from_periods(self.rec.periods, tcrit=50.0)
        assert len(br.bursts) == 2

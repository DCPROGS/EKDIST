import os
import numpy as np

from ekdist import ekscn
from ekdist import ekrecord

class TestSCNFileLoading:
    def setup_method(self):
        self.infile = "./tests/AChsim.scn"
        self.header = ekscn.read_header(self.infile)
        self.itint, self.iampl, self.iprops = ekscn.read_data(self.infile, self.header)
        
    def test_infile_exists(self):
        assert os.path.isfile(self.infile)
        
    def test_SCN_header_loading(self):
        assert self.header
        
    def test_interval_number(self):
        assert self.header['nint'] == 13948
        
    def test_interval_loading(self):    
        assert self.header['nint'] == len(self.itint)
        
    def test_flags(self):
        assert not self.iprops.any()
        
    def test_amplitudes(self):
        assert self.iampl[0] == 0.
        assert self.iampl[1] == 6.
        assert self.iampl[2] == 0.

    def tearDown(self):
        self.header = None
        self.itint, self.iampl, self.iprops = None, None, None
        
    def test_intervals_are_float64(self):
        """The ms-to-s conversion must widen before it scales.

        tint is stored as float32, and under NEP 50 a float32 array times a
        Python float stays float32, so `np.array(tint) * 0.001` did the scaling
        in single precision. Over A-10 that lost 2e-5 s from a 435 s record --
        4.6e-8 relative -- and reached the fitted likelihood. It matched
        dcpyps only because dcpyps rounded the same way."""
        assert self.itint.dtype == np.float64


class TestIntervalListLoading:
    def setup_method(self):
        self.intervals = [20.0, 1.0, 19.0, 100.0, 10.0, 100.0, 1.0]
        self.amplitudes = [5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0]
        self.rec = ekrecord.SingleChannelRecord()
        self.rec.load_intervals_from_list(np.array(self.intervals), np.array(self.amplitudes))

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
        br = ekrecord.Bursts(self.rec.periods)
        br.slice_bursts(50.0)
        assert len(br.bursts) == 2 

    def test_unusable_shut_ends_a_cluster(self):
        """Clause (2) of slice_bursts: an unusable shut ends a cluster whatever
        its nominal duration. Time-course fitting leaves such an interval with
        no defined length, so it is never comparable with tcrit -- here it is
        far shorter, and used to be treated as a within-burst gap that joined
        two clusters into one."""
        intervals = [20.0, 0.04, 19.0]
        amplitudes = [5.0, 0.0, 5.0]
        flags = [0, ekrecord.FLAG_UNUSABLE, 0]
        rec = ekrecord.SingleChannelRecord()
        rec.load_intervals_from_list(np.array(intervals), np.array(amplitudes),
                                     np.array(flags))
        br = ekrecord.Bursts(rec.periods)
        br.slice_bursts(50.0)
        assert len(br.bursts) == 2
        assert [b.tolist() for b in br.bursts] == [[20.0], [19.0]]

    def test_unusable_flag_marks_its_own_period(self):
        """A period is unusable if an interval *in it* is. The flag used to be
        applied before the merge decision, so an unusable interval marked the
        period it was about to end -- flagging the opening in front of it."""
        intervals = [20.0, 0.04, 19.0]
        amplitudes = [5.0, 0.0, 5.0]
        flags = [0, ekrecord.FLAG_UNUSABLE, 0]
        rec = ekrecord.SingleChannelRecord()
        rec.load_intervals_from_list(np.array(intervals), np.array(amplitudes),
                                     np.array(flags))
        assert list(rec.periods.flags) == [0, ekrecord.FLAG_UNUSABLE, 0]

    def test_first_opening_starts_a_cluster(self):
        """Clause (1): no gap longer than tcrit is required before the first
        cluster of a record."""
        intervals = [20.0, 100.0, 19.0]
        amplitudes = [5.0, 0.0, 5.0]
        rec = ekrecord.SingleChannelRecord()
        rec.load_intervals_from_list(np.array(intervals), np.array(amplitudes))
        br = ekrecord.Bursts(rec.periods)
        br.slice_bursts(50.0)
        assert len(br.bursts) == 2

    def tearDown(self):
        self.intervals, self.amplitudes = None, None
        self.rec = None

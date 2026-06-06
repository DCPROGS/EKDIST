"""Functional tests for the full record workflow, updated for new API."""

import pytest
from ekdist.record import SingleChannelRecord


class TestFunctional:
    def setup_method(self):
        infile = str(pytest.importorskip("pathlib").Path(__file__).parent / "AChsim.scn")
        self.rec = SingleChannelRecord.from_scn(infile)
        self.rec.tres = 30e-6

    def test_record_initiated(self):
        assert self.rec._is_loaded

    def test_intervals_loaded(self):
        assert abs(len(self.rec.itint) - self.rec.header.n_intervals) <= 1

    def test_intervals_resolved(self):
        assert len(self.rec.itint) - 1 > len(self.rec.rtint)

    def test_periods_set(self):
        assert len(self.rec.periods.intervals) > 0

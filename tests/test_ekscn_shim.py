"""ekscn reads through dcio.

M2 of the record-layer plan. EKDIST carried a second implementation of the SCN
binary decoding, and two copies of a format parser is two places for a record
to be read differently. It was: the interval array is stored as 4-byte floats,
and under NEP 50 the millisecond-to-second multiply stayed in single precision
here, discarding most of the digits the file carries.
"""

import numpy as np
import pytest

from dcio.formats import scn as dcio_scn
from ekdist import ekrecord, ekscn

FILES = ["./tests/AChsim.scn", "./tests/181003S8.SCN"]


@pytest.mark.parametrize("path", FILES)
class TestAgreesWithDcio:

    def test_intervals_are_identical(self, path):
        """Not close -- identical. Both read the same bytes; any difference
        would be a decoding difference, not rounding."""
        header = ekscn.read_header(path)
        intervals, _, _ = ekscn.read_data(path, header)
        assert np.array_equal(intervals, dcio_scn.read(path).intervals)

    def test_flags_are_identical(self, path):
        header = ekscn.read_header(path)
        _, _, flags = ekscn.read_data(path, header)
        record = dcio_scn.read(path)
        assert np.array_equal(flags.astype(int), record.flags.astype(int))

    def test_amplitudes_round_trip_through_the_calibration(self, path):
        """read_data returns raw stored amplitudes, as it always has, so
        load_SCN_file's multiply by calfac2 stays correct. dcio applies the
        calibration on read, so it is divided back out."""
        header = ekscn.read_header(path)
        _, raw, _ = ekscn.read_data(path, header)
        record = dcio_scn.read(path)
        np.testing.assert_allclose(raw * record.header.calfac2,
                                   record.amplitudes, rtol=1e-12)

    def test_trailing_partial_interval_trimmed_the_same_way(self, path):
        """An experimental record is truncated back to a final shutting. Both
        readers do it, and they must agree on how many intervals remain."""
        header = ekscn.read_header(path)
        intervals, _, _ = ekscn.read_data(path, header)
        assert len(intervals) == len(dcio_scn.read(path).intervals)


class TestRecordStillLoads:

    @pytest.mark.parametrize("path", FILES)
    def test_load_scn_file_end_to_end(self, path):
        rec = ekrecord.SingleChannelRecord()
        rec.load_SCN_file(path)
        assert len(rec.itint) > 0
        assert len(rec.itint) == len(rec.iampl) == len(rec.iprop)

    def test_amplitudes_are_calibrated_exactly_once(self):
        """The file with a non-unit calfac2 is the one that would expose a
        double or missing application."""
        path = "./tests/181003S8.SCN"
        rec = ekrecord.SingleChannelRecord()
        rec.load_SCN_file(path)
        np.testing.assert_allclose(rec.iampl, dcio_scn.read(path).amplitudes,
                                   rtol=1e-12)

    def test_calfac2_is_not_unity_in_that_file(self):
        """Guards the test above: if this ever became 1.0, the check would
        pass while proving nothing."""
        assert dcio_scn.read("./tests/181003S8.SCN").header.calfac2 != 1.0

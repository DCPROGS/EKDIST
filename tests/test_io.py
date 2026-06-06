"""Tests for ekdist.io — SCN file reading via dcio."""

import numpy as np
import pytest

from ekdist.io import read_scn
from dcio.formats.scn import SCNRecord


class TestReadSCN:
    def test_returns_scn_record(self, scn_file):
        result = read_scn(scn_file)
        assert isinstance(result, SCNRecord)

    def test_interval_count_matches_header(self, scn_file):
        rec = read_scn(scn_file)
        # Simulated files: sentinel stripped → len may be header.n_intervals - 1
        assert abs(len(rec.intervals) - rec.header.n_intervals) <= 1

    def test_amplitudes_float(self, scn_file):
        rec = read_scn(scn_file)
        assert rec.amplitudes.dtype == float

    def test_intervals_positive(self, scn_file):
        rec = read_scn(scn_file)
        assert np.sum(rec.intervals > 0) > 0.9 * len(rec.intervals)

    def test_header_has_required_fields(self, scn_file):
        rec = read_scn(scn_file)
        assert rec.header.n_intervals > 0
        assert rec.header.version in (-103, 103, 104)
        assert rec.header.record_type in ("simulated", "experimental")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_scn(tmp_path / "no_such_file.scn")

    def test_accepts_path_object(self, scn_file):
        from pathlib import Path
        result = read_scn(Path(scn_file))
        assert result is not None

    def test_experimental_scn_loads(self, scn_file_exp):
        rec = read_scn(scn_file_exp)
        assert len(rec.intervals) > 0
        assert rec.header.record_type == "experimental"

    def test_flags_int8(self, scn_file):
        rec = read_scn(scn_file)
        assert rec.flags.dtype == np.int8

    def test_arrays_same_length(self, scn_file):
        rec = read_scn(scn_file)
        assert len(rec.intervals) == len(rec.amplitudes) == len(rec.flags)

"""Log-binned histogram, now built on dcio.analysis.histogram.

The binning arithmetic moved down to dcio (M3 of the record-layer plan). These
tests pin the behaviour EKDIST exposes through prepare_xlog_hist, including the
truncation bug that moving it fixed.
"""

import math

import numpy as np
import pytest

from ekdist.ekplot import prepare_xlog_hist


def _ek_old_edges(X, tres):
    """The bin edges EKDIST built before the move, for comparison.

    The upper limit is exp(ceil(ln(max))) -- a power of e, not of ten.
    """
    n = len(X)
    nbdec = 5 if n <= 300 else 8 if n <= 1000 else 10 if n <= 3000 else 12
    dx = math.exp(math.log(10.0) / float(nbdec))
    xend = math.exp(math.ceil(math.log(max(X))))
    nbin = int(math.log(xend / tres) / math.log(dx))
    return tres * np.array([dx ** i for i in range(nbin + 1)])


class TestPrepareXlogHist:

    def test_returns_staircase_lists(self):
        x = np.random.default_rng(0).exponential(0.005, 500) + 2.5e-5
        xout, yout = prepare_xlog_hist(x, 2.5e-5)
        assert isinstance(xout, list) and isinstance(yout, list)
        assert len(xout) == len(yout)

    def test_closes_to_zero_at_both_ends(self):
        x = np.random.default_rng(1).exponential(0.005, 500) + 2.5e-5
        _, yout = prepare_xlog_hist(x, 2.5e-5)
        assert yout[0] == 0 and yout[-1] == 0

    def test_starts_at_tres(self):
        x = np.random.default_rng(2).exponential(0.005, 500) + 2.5e-5
        xout, _ = prepare_xlog_hist(x, 2.5e-5)
        assert xout[0] == pytest.approx(2.5e-5)

    def test_counts_every_interval(self):
        """The regression the move fixed.

        The old upper limit could fall below max(X), and np.histogram then
        dropped those intervals with no warning. Over 400 random exponential
        samples that happened in 119 of them."""
        rng = np.random.default_rng(0)
        old_would_drop = 0
        for _ in range(150):
            n = int(rng.integers(50, 6000))
            x = rng.exponential(0.005, n) + 2.5e-5
            _, yout = prepare_xlog_hist(x, 2.5e-5)
            # each bar contributes its count twice
            assert sum(yout) // 2 == n
            if x.max() > _ek_old_edges(x, 2.5e-5)[-1]:
                old_would_drop += 1
        assert old_would_drop > 0, "sample no longer exercises the old bug"

    def test_bin_ratio_is_a_decade_root(self):
        x = np.random.default_rng(3).exponential(0.005, 2000) + 2.5e-5
        xout, _ = prepare_xlog_hist(x, 2.5e-5)
        edges = np.array(xout[::2])
        ratio = edges[1] / edges[0]
        assert ratio == pytest.approx(10.0 ** (1.0 / 10))   # 2000 -> 10/decade

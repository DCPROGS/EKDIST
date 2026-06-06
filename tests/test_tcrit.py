"""Tests for ekdist.tcrit.Tcrit."""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist.tcrit import Tcrit, _misclassified


class TestTcritConstruction:
    def test_requires_two_components(self):
        with pytest.raises(ValueError, match="at least 2"):
            Tcrit(np.array([0.01]), np.array([1.0]))

    def test_tcrits_keys(self):
        tau = np.array([0.01, 0.1])
        area = np.array([0.5, 0.5])
        tc = Tcrit(tau, area)
        assert set(tc.tcrits.keys()) == {"DC", "C&N", "Jackson"}

    def test_tcrits_length(self):
        """Number of tcrits per criterion = ncomps - 1."""
        tau = np.array([0.01, 0.1, 1.0])
        area = np.array([1/3, 1/3, 1/3])
        tc = Tcrit(tau, area)
        assert len(tc.tcrits["DC"]) == 2
        assert len(tc.tcrits["C&N"]) == 2
        assert len(tc.tcrits["Jackson"]) == 2


class TestTcritValues:
    def setup_method(self):
        """Reference: 2-component shut-time distribution from EKDIST example."""
        self.tau = np.array([0.01, 0.1])   # 10 ms and 100 ms
        self.area = np.array([0.7, 0.3])
        self.tc = Tcrit(self.tau, self.area)

    def test_dc_tcrit_between_components(self):
        """DC tcrit must lie between tau[0] and tau[1]."""
        dc = self.tc.tcrits["DC"][0]
        assert dc is not None
        assert self.tau[0] < dc < self.tau[1]

    def test_cn_tcrit_between_components(self):
        cn = self.tc.tcrits["C&N"][0]
        assert cn is not None
        assert self.tau[0] < cn < self.tau[1]

    def test_jackson_tcrit_between_components(self):
        jk = self.tc.tcrits["Jackson"][0]
        assert jk is not None
        assert self.tau[0] < jk < self.tau[1]

    def test_dc_equal_fraction_misclassified(self):
        """At DC tcrit, fraction misclassified must be equal for both sides."""
        dc = self.tc.tcrits["DC"][0]
        _, _, pf, ps = _misclassified(dc, self.tau, self.area, 1)
        assert abs(pf - ps) < 1e-6

    def test_cn_equal_number_misclassified(self):
        """At C&N tcrit, number misclassified must be equal for both sides."""
        cn = self.tc.tcrits["C&N"][0]
        enf, ens, _, _ = _misclassified(cn, self.tau, self.area, 1)
        assert abs(enf - ens) < 1e-6

    def test_three_component_two_tcrits(self):
        tau = np.array([0.001, 0.01, 0.1])
        area = np.array([0.5, 0.3, 0.2])
        tc = Tcrit(tau, area)
        dc = tc.tcrits["DC"]
        assert len(dc) == 2
        # First tcrit between comp 0 and 1
        assert dc[0] is not None and tau[0] < dc[0] < tau[1]
        # Second tcrit between comp 1 and 2
        assert dc[1] is not None and tau[1] < dc[1] < tau[2]


class TestTcritSummary:
    def test_summary_runs(self):
        tc = Tcrit(np.array([0.01, 0.1]), np.array([0.5, 0.5]))
        s = tc.summary()
        assert "DC" in s
        assert "C&N" in s

    def test_misclassified_summary_runs(self):
        tc = Tcrit(np.array([0.01, 0.1]), np.array([0.5, 0.5]))
        s = tc.misclassified_summary("DC")
        assert "misclassified" in s


class TestMisclassifiedFunction:
    def test_returns_four_values(self):
        tau = np.array([0.01, 0.1])
        area = np.array([0.5, 0.5])
        result = _misclassified(0.05, tau, area, 1)
        assert len(result) == 4

    def test_fractions_between_zero_one(self):
        tau = np.array([0.01, 0.1])
        area = np.array([0.5, 0.5])
        enf, ens, pf, ps = _misclassified(0.05, tau, area, 1)
        assert 0 <= pf <= 1
        assert 0 <= ps <= 1

    def test_at_very_short_tcrit_mostly_fast_misclassified(self):
        """At a very short tcrit, very few slow events are misclassified."""
        tau = np.array([0.001, 1.0])
        area = np.array([0.5, 0.5])
        _, ens, _, ps = _misclassified(0.0001, tau, area, 1)
        assert ps < 0.001  # almost no slow events misclassified

    def test_get_tcrits_from_exponential_pdf(self, dwell_times):
        from ekdist.fitting import ExponentialPDF
        pdf = ExponentialPDF([0.036, 1.1], [0.20])
        pdf.fit(dwell_times)
        pdf.get_tcrits()
        assert pdf.tcrits["DC"] is not None

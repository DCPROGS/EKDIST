"""Smoke tests for ekdist.plots — verify functions return Figure objects.

These are integration tests that exercise the full plotting pipeline.
They do not assert pixel-level correctness, only that functions complete
without error and return the expected type.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt

from ekdist import plots
from ekdist.fitting import ExponentialPDF


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


class TestStabilityPlots:
    def test_stability_intervals_returns_figure(self, loaded_record):
        fig = plots.stability_intervals(loaded_record)
        assert isinstance(fig, plt.Figure)

    def test_stability_amplitudes_returns_figure(self, loaded_record):
        fig = plots.stability_amplitudes(loaded_record)
        assert isinstance(fig, plt.Figure)

    def test_stability_intervals_with_flags(self, loaded_record):
        fig = plots.stability_intervals(
            loaded_record, show_open=True, show_shut=False, show_popen=True
        )
        assert isinstance(fig, plt.Figure)


class TestAmplitudeHistogram:
    def test_histogram_amplitudes_returns_figure(self, loaded_record):
        fc = 3000.0
        fig = plots.histogram_amplitudes(loaded_record, fc)
        assert isinstance(fig, plt.Figure)

    def test_histogram_amplitudes_with_gaussian_mixture(self, loaded_record):
        from ekdist.fitting import GaussianMixturePDF
        fc = 3000.0
        amps = np.abs(np.array(loaded_record.rampl))
        amps = amps[amps > 0]
        mu = float(np.mean(amps))
        pdf = GaussianMixturePDF([mu], [float(np.std(amps))])
        fig = plots.histogram_amplitudes(loaded_record, fc, pdf=pdf)
        assert isinstance(fig, plt.Figure)


class TestBurstHistogram:
    def test_burst_openings_returns_figure(self, loaded_record):
        from ekdist.bursts import Bursts
        bursts = Bursts.from_periods(loaded_record.periods, tcrit=5e-3)
        nops = bursts.list_n_openings()
        fig = plots.histogram_burst_openings(nops)
        assert isinstance(fig, plt.Figure)


class TestDwellTimeHistogram:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.X = rng.exponential(0.01, 500)
        self.tres = 1e-4

    def test_returns_figure(self):
        fig = plots.histogram_dwell_times(self.X, self.tres)
        assert isinstance(fig, plt.Figure)

    def test_with_pdf(self):
        pdf = ExponentialPDF([0.01])
        fig = plots.histogram_dwell_times(self.X, self.tres, pdf=pdf)
        assert isinstance(fig, plt.Figure)

    def test_with_tcrits(self):
        pdf = ExponentialPDF([0.005, 0.05], [0.6])
        pdf.fit(self.X)
        fig = plots.histogram_dwell_times(self.X, self.tres, pdf=pdf, tcrits=[0.02])
        assert isinstance(fig, plt.Figure)

    def test_prepare_xlog_hist_returns_lists(self):
        xout, yout = plots.prepare_xlog_hist(self.X, self.tres)
        assert isinstance(xout, list)
        assert isinstance(yout, list)
        assert len(xout) == len(yout)

    def test_on_real_open_intervals(self, loaded_record):
        opens = loaded_record.periods.open_intervals
        tres = loaded_record.tres
        pdf = ExponentialPDF([1e-3, 10e-3])
        pdf.fit(opens)
        fig = plots.histogram_dwell_times(opens, tres, pdf=pdf)
        assert isinstance(fig, plt.Figure)

    def test_with_gaussian_mixture_pdf_does_not_crash(self):
        """histogram_dwell_times must not crash when given a GaussianMixturePDF.

        Bug 5: _scale_factor accessed pdf.area and pdf.tau directly, which only
        exist on ExponentialPDF.  Any other PDF raised AttributeError.
        """
        from ekdist.fitting import GaussianMixturePDF
        rng = np.random.default_rng(3)
        X = np.abs(rng.normal(0.01, 0.002, 400)) + 1e-4
        pdf = GaussianMixturePDF([0.01], [0.002])
        fig = plots.histogram_dwell_times(X, self.tres, pdf=pdf)
        assert isinstance(fig, plt.Figure)

    def test_with_callable_pdf_does_not_crash(self):
        """A plain callable pdf(X) must also work."""
        from scipy.stats import expon
        X = np.random.exponential(0.01, 400)
        # Pass a lambda that has no .area or .tau attributes
        pdf_fn = lambda theta, x: expon.pdf(x, scale=0.01)
        # Wrap it to look like a PDF object with pdf() method but no tau/area
        class SimplePDF:
            theta = np.array([0.01])
            def pdf(self, theta, X):
                return expon.pdf(X, scale=theta[0])
        fig = plots.histogram_dwell_times(X, self.tres, pdf=SimplePDF())
        assert isinstance(fig, plt.Figure)


# =========================================================================== #
# Amplitude histogram — experimental file (real amplitude data)                 #
# =========================================================================== #

class TestAmplitudeHistogramExperimental:
    """Tests using glyr_experimental.scn which contains real per-opening amplitudes."""

    FC = 3000.0  # Hz

    def test_returns_figure(self, glyr_record):
        fig = plots.histogram_amplitudes(glyr_record, self.FC)
        assert isinstance(fig, plt.Figure)

    def test_auto_single_gaussian(self, glyr_record):
        """Without a pdf argument, a single Gaussian is auto-fitted."""
        fig = plots.histogram_amplitudes(glyr_record, self.FC)
        ax = fig.axes[0]
        # Title should contain µ= from the auto-fit branch
        assert "µ" in ax.get_title() or "=" in ax.get_title()

    def test_with_single_component_gaussian_mixture(self, glyr_record):
        from ekdist.fitting import GaussianMixturePDF
        import numpy as np
        amps = np.abs(np.asarray(glyr_record.iampl))
        amps = amps[amps > 0]
        pdf = GaussianMixturePDF([float(np.mean(amps))], [float(np.std(amps))])
        pdf.fit(amps)
        fig = plots.histogram_amplitudes(glyr_record, self.FC, pdf=pdf)
        assert isinstance(fig, plt.Figure)

    def test_with_two_component_gaussian_mixture(self, glyr_record):
        from ekdist.fitting import GaussianMixturePDF
        import numpy as np
        amps = np.abs(np.asarray(glyr_record.iampl))
        amps = amps[amps > 0]
        mu = float(np.mean(amps))
        sig = float(np.std(amps))
        pdf = GaussianMixturePDF([mu * 0.7, mu * 1.3], [sig, sig])
        pdf.fit(amps)
        fig = plots.histogram_amplitudes(glyr_record, self.FC, pdf=pdf)
        assert isinstance(fig, plt.Figure)

    def test_custom_n_risetimes(self, glyr_record):
        fig = plots.histogram_amplitudes(glyr_record, self.FC, n_risetimes=3.0)
        assert isinstance(fig, plt.Figure)

    def test_custom_nbins(self, glyr_record):
        fig = plots.histogram_amplitudes(glyr_record, self.FC, nbins=40)
        assert isinstance(fig, plt.Figure)

    def test_stability_amplitudes_returns_figure(self, glyr_record):
        fig = plots.stability_amplitudes(glyr_record)
        assert isinstance(fig, plt.Figure)


# =========================================================================== #
# stability_rolling                                                              #
# =========================================================================== #

class TestStabilityRolling:
    @pytest.fixture
    def open_shut(self):
        rng = np.random.default_rng(0)
        opens = rng.exponential(0.001, 300)
        shuts = rng.exponential(0.010, 300)
        return opens, shuts

    def test_returns_figure(self, open_shut):
        opens, shuts = open_shut
        fig = plots.stability_rolling(opens, shuts)
        assert isinstance(fig, plt.Figure)

    def test_has_three_axes(self, open_shut):
        opens, shuts = open_shut
        fig = plots.stability_rolling(opens, shuts)
        assert len(fig.axes) == 3

    def test_custom_window(self, open_shut):
        opens, shuts = open_shut
        fig = plots.stability_rolling(opens, shuts, window=50)
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(self, open_shut):
        opens, shuts = open_shut
        fig = plots.stability_rolling(opens, shuts, figsize=(8, 6))
        assert isinstance(fig, plt.Figure)

    def test_unequal_length_arrays(self):
        """len(opens) = len(shuts) + 1 is the normal single-channel case."""
        rng = np.random.default_rng(1)
        opens = rng.exponential(0.001, 201)
        shuts = rng.exponential(0.010, 200)
        fig = plots.stability_rolling(opens, shuts)
        assert isinstance(fig, plt.Figure)


# =========================================================================== #
# plot_serial_correlation                                                        #
# =========================================================================== #

class TestPlotSerialCorrelation:
    @pytest.fixture
    def sc_result(self):
        from ekdist.correlation import serial_correlation
        rng = np.random.default_rng(0)
        X = rng.exponential(0.01, 300)
        return serial_correlation(X, max_lag=5)

    def test_single_result_returns_figure(self, sc_result):
        fig = plots.plot_serial_correlation(sc_result)
        assert isinstance(fig, plt.Figure)

    def test_single_result_has_one_axis(self, sc_result):
        fig = plots.plot_serial_correlation(sc_result)
        assert len(fig.axes) == 1

    def test_two_results_returns_figure(self, sc_result):
        fig = plots.plot_serial_correlation(sc_result, sc_result)
        assert isinstance(fig, plt.Figure)

    def test_two_results_have_two_axes(self, sc_result):
        fig = plots.plot_serial_correlation(sc_result, sc_result)
        assert len(fig.axes) == 2

    def test_custom_labels(self, sc_result):
        fig = plots.plot_serial_correlation(sc_result, labels=["Open periods"])
        assert isinstance(fig, plt.Figure)

    def test_custom_colors(self, sc_result):
        fig = plots.plot_serial_correlation(
            sc_result, sc_result,
            labels=["Open", "Shut"], colors=["steelblue", "tomato"],
        )
        assert isinstance(fig, plt.Figure)

    def test_no_results_raises(self):
        with pytest.raises((ValueError, TypeError)):
            plots.plot_serial_correlation()


# =========================================================================== #
# plot_open_shut_scatter                                                         #
# =========================================================================== #

class TestPlotOpenShutScatter:
    @pytest.fixture
    def pairs(self):
        rng = np.random.default_rng(1)
        op = rng.exponential(0.001, 150)
        sh = rng.exponential(0.010, 150)
        return op, sh

    def test_returns_figure(self, pairs):
        op, sh = pairs
        fig = plots.plot_open_shut_scatter(op, sh)
        assert isinstance(fig, plt.Figure)

    def test_has_at_least_two_axes(self, pairs):
        op, sh = pairs
        fig = plots.plot_open_shut_scatter(op, sh)
        # scatter + 2D histogram + colorbar axes → at least 2
        assert len(fig.axes) >= 2

    def test_with_result_object(self, pairs):
        from ekdist.correlation import open_shut_correlation
        op, sh = pairs
        result = open_shut_correlation(op, sh)
        fig = plots.plot_open_shut_scatter(op, sh, result=result)
        assert isinstance(fig, plt.Figure)

    def test_without_result_object(self, pairs):
        op, sh = pairs
        fig = plots.plot_open_shut_scatter(op, sh, result=None)
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(self, pairs):
        op, sh = pairs
        fig = plots.plot_open_shut_scatter(op, sh, figsize=(8, 3))
        assert isinstance(fig, plt.Figure)

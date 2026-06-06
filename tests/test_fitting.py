"""Tests for ekdist.fitting — ExponentialPDF and GaussianMixturePDF."""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist.fitting import ExponentialPDF, GaussianMixturePDF


# =========================================================================== #
# ExponentialPDF                                                                #
# =========================================================================== #

class TestExponentialPDFConstruction:
    def test_uniform_area_default(self):
        pdf = ExponentialPDF([0.1, 1.0])
        npt.assert_almost_equal(pdf.area, [0.5, 0.5])

    def test_area_k_minus_1_derives_last(self):
        pdf = ExponentialPDF([0.1, 1.0], [0.3])
        npt.assert_almost_equal(pdf.area[-1], 0.7, decimal=10)

    def test_area_full_length_normalised(self):
        pdf = ExponentialPDF([0.1, 1.0], [0.4, 0.6])
        npt.assert_almost_equal(np.sum(pdf.area), 1.0, decimal=10)

    def test_negative_tau_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ExponentialPDF([-0.1, 1.0])

    def test_wrong_area_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            ExponentialPDF([0.1, 1.0], [0.3, 0.4, 0.3])  # length 3 for 2 components

    def test_pars_concatenation(self):
        tau = [0.1, 1.0]
        pdf = ExponentialPDF(tau)
        npt.assert_almost_equal(pdf.pars[:2], tau)

    def test_last_area_fixed(self):
        pdf = ExponentialPDF([0.1, 1.0])
        assert pdf.fixed[-1] is True

    def test_n_free_params(self):
        pdf = ExponentialPDF([0.1, 1.0])
        # 2 tau + 2 area - 1 fixed = 3 free
        assert len(pdf.theta) == 3


class TestExponentialPDFTheta:
    def test_set_theta_updates_tau_area(self):
        pdf = ExponentialPDF([0.1, 1.0])
        pdf.theta = np.array([0.2, 2.0, 0.3])
        npt.assert_almost_equal(pdf.pars, [0.2, 2.0, 0.3, 0.7])

    def test_fixed_param_preserved_on_set(self):
        pdf = ExponentialPDF([0.1, 1.0])
        pdf.fixed[1] = True  # fix tau[1]
        pdf.theta = np.array([0.2, 0.3])  # only tau[0] and area[0] free
        npt.assert_almost_equal(pdf.tau[1], 1.0)  # tau[1] unchanged

    def test_area_sums_to_one_after_set(self):
        pdf = ExponentialPDF([0.1, 1.0])
        pdf.theta = np.array([0.15, 0.9, 0.25])
        npt.assert_almost_equal(np.sum(pdf.area), 1.0, decimal=10)


class TestExponentialPDFDensity:
    def setup_method(self):
        self.pdf = ExponentialPDF([0.01, 0.1], [0.6, 0.4])

    def test_pdf_positive(self):
        X = np.array([0.001, 0.01, 0.1])
        y = self.pdf.pdf(self.pdf.theta, X)
        assert np.all(y > 0)

    def test_pdf_vectorised(self):
        X = np.linspace(0.001, 0.5, 100)
        y = self.pdf.pdf(self.pdf.theta, X)
        assert y.shape == X.shape

    def test_pdf_integrates_approximately_one(self):
        """∫ f(t) dt ≈ 1 over a wide range."""
        X = np.linspace(0.0, 2.0, 10000)
        y = self.pdf.pdf(self.pdf.theta, X)
        integral = np.trapezoid(y, X)
        assert abs(integral - 1.0) < 0.05


class TestExponentialPDFLogLikelihood:
    def test_ll_on_intervals_txt(self, dwell_times):
        tau, area = [0.036, 1.1], [0.20]
        pdf = ExponentialPDF(tau, area)
        theta_in = np.array(tau + area)
        ll = pdf.LL(theta_in, dwell_times)
        npt.assert_almost_equal(ll, 87.31806715582867, decimal=4)

    def test_ll_is_finite(self):
        pdf = ExponentialPDF([0.01, 0.1])
        X = np.random.exponential(0.05, 100)
        ll = pdf.LL(pdf.theta, X)
        assert np.isfinite(ll)

    def test_tres_defaults_to_zero(self):
        pdf = ExponentialPDF([0.01, 0.1])
        assert pdf._tres == 0.0

    def test_ll_with_tres_below_xmin_differs_from_no_tres(self):
        """LL should differ when tres < min(X) because the normalisation denominator changes."""
        rng = np.random.default_rng(0)
        X = rng.exponential(0.05, 300) + 0.01  # all values > 0.01
        pdf_no_tres = ExponentialPDF([0.05], [1.0])
        pdf_tres = ExponentialPDF([0.05], [1.0])
        pdf_tres._tres = 0.001  # tres < min(X) = ~0.01
        ll_no = pdf_no_tres.LL(pdf_no_tres.theta, X)
        ll_tr = pdf_tres.LL(pdf_tres.theta, X)
        assert ll_no != ll_tr, "LL must differ when tres != min(X)"

    def test_ll_tres_equals_xmin_matches_no_tres(self):
        """When tres == min(X) the two formulations are identical."""
        rng = np.random.default_rng(1)
        X = rng.exponential(0.05, 300)
        pdf_no = ExponentialPDF([0.05], [1.0])
        pdf_tr = ExponentialPDF([0.05], [1.0])
        pdf_tr._tres = float(np.min(X))
        ll_no = pdf_no.LL(pdf_no.theta, X)
        ll_tr = pdf_tr.LL(pdf_tr.theta, X)
        npt.assert_almost_equal(ll_no, ll_tr, decimal=10)

    def test_fit_sets_tres(self, dwell_times):
        pdf = ExponentialPDF([0.036, 1.1], [0.20])
        tres_val = 1e-4
        pdf.fit(dwell_times, tres=tres_val)
        assert pdf._tres == tres_val

    def test_fit_no_tres_keeps_zero(self, dwell_times):
        pdf = ExponentialPDF([0.036, 1.1], [0.20])
        pdf.fit(dwell_times)
        assert pdf._tres == 0.0


class TestExponentialPDFFit:
    def test_fit_reduces_ll(self, dwell_times):
        pdf = ExponentialPDF([0.036, 1.1], [0.20])
        ll_before = pdf.LL(pdf.theta, dwell_times)
        pdf.fit(dwell_times)
        ll_after = pdf.LL(pdf.theta, dwell_times)
        assert ll_after <= ll_before

    def test_fit_estimates_close_to_reference(self, dwell_times):
        """Compare to reference values from old test_errors.py."""
        pdf = ExponentialPDF([0.036, 1.1], [0.20])
        res = pdf.fit(dwell_times)
        npt.assert_almost_equal(res.x[0], 0.03700718, decimal=3)
        npt.assert_almost_equal(res.x[1], 1.07302608, decimal=2)

    def test_fit_areas_sum_to_one(self, dwell_times):
        pdf = ExponentialPDF([0.036, 1.1], [0.20])
        pdf.fit(dwell_times)
        npt.assert_almost_equal(np.sum(pdf.area), 1.0, decimal=8)

    def test_mean(self):
        pdf = ExponentialPDF([1.0, 10.0], [0.5, 0.5])
        npt.assert_almost_equal(pdf.mean(), 5.5, decimal=10)


# =========================================================================== #
# GaussianMixturePDF                                                           #
# =========================================================================== #

class TestGaussianMixturePDFConstruction:
    def test_uniform_areas_default(self):
        pdf = GaussianMixturePDF([5.0, 10.0], [1.0, 1.5])
        npt.assert_almost_equal(pdf.areas, [0.5, 0.5])

    def test_area_k_minus_1_derives_last(self):
        pdf = GaussianMixturePDF([5.0, 10.0], [1.0, 1.5], [0.3])
        npt.assert_almost_equal(pdf.areas[-1], 0.7, decimal=10)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GaussianMixturePDF([5.0], [-1.0])

    def test_mismatched_means_sigmas_raises(self):
        with pytest.raises(ValueError, match="same length"):
            GaussianMixturePDF([5.0, 10.0], [1.0])

    def test_last_area_fixed(self):
        pdf = GaussianMixturePDF([5.0, 10.0], [1.0, 1.5])
        assert pdf.fixed[-1] is True


class TestGaussianMixturePDFDensity:
    def setup_method(self):
        self.pdf = GaussianMixturePDF([5.0, 15.0], [1.0, 2.0], [0.4, 0.6])

    def test_pdf_positive(self):
        X = np.array([4.0, 5.0, 15.0, 16.0])
        y = self.pdf.pdf(self.pdf.theta, X)
        assert np.all(y > 0)

    def test_pdf_peaks_near_means(self):
        X = np.linspace(0, 25, 1000)
        y = self.pdf.pdf(self.pdf.theta, X)
        peak_idx = np.argmax(y[:500]), np.argmax(y[500:]) + 500
        assert abs(X[peak_idx[0]] - 5.0) < 1.0
        assert abs(X[peak_idx[1]] - 15.0) < 1.0

    def test_pdf_integrates_approximately_one(self):
        X = np.linspace(-10, 40, 10000)
        y = self.pdf.pdf(self.pdf.theta, X)
        integral = np.trapezoid(y, X)
        assert abs(integral - 1.0) < 0.01


class TestGaussianMixturePDFFit:
    def test_fit_runs(self):
        rng = np.random.default_rng(42)
        X = np.concatenate([
            rng.normal(5.0, 1.0, 200),
            rng.normal(15.0, 2.0, 300),
        ])
        pdf = GaussianMixturePDF([4.0, 14.0], [1.2, 2.2], [0.35])
        res = pdf.fit(X)
        assert res.success or res.fun < 1e6

    def test_fit_means_roughly_correct(self):
        rng = np.random.default_rng(42)
        X = np.concatenate([
            rng.normal(5.0, 1.0, 500),
            rng.normal(15.0, 2.0, 500),
        ])
        pdf = GaussianMixturePDF([4.5, 14.5], [1.0, 2.0])
        pdf.fit(X)
        means_sorted = np.sort(pdf.means)
        assert abs(means_sorted[0] - 5.0) < 0.5
        assert abs(means_sorted[1] - 15.0) < 0.5

    def test_areas_sum_to_one_after_fit(self):
        rng = np.random.default_rng(0)
        X = rng.normal(10.0, 1.5, 300)
        pdf = GaussianMixturePDF([9.0, 11.0], [1.0, 1.5])
        pdf.fit(X)
        npt.assert_almost_equal(np.sum(pdf.areas), 1.0, decimal=6)

    def test_ll_reduces_after_fit(self):
        rng = np.random.default_rng(1)
        X = np.concatenate([rng.normal(3.0, 0.5, 200), rng.normal(8.0, 1.0, 200)])
        pdf = GaussianMixturePDF([2.0, 7.0], [0.5, 1.0])
        ll_before = pdf.LL(pdf.theta, X)
        pdf.fit(X)
        ll_after = pdf.LL(pdf.theta, X)
        assert ll_after <= ll_before

    def test_mean(self):
        pdf = GaussianMixturePDF([5.0, 15.0], [1.0, 1.0], [0.5, 0.5])
        npt.assert_almost_equal(pdf.mean(), 10.0, decimal=10)


class TestGaussianMixturePDFVectorised:
    """Numerical regression tests for the vectorised pdf/LL implementation.

    Each test computes the expected result directly from scipy.stats.norm so
    the comparison is independent of which implementation path (loop or broadcast)
    is active in the code under test.
    """

    @staticmethod
    def _loop_pdf(means, sigmas, areas, X):
        from scipy.stats import norm
        y = np.zeros(len(X), dtype=float)
        for mu, sigma, area in zip(means, sigmas, areas):
            y += area * norm.pdf(X, mu, sigma)
        return y

    @staticmethod
    def _loop_ll(means, sigmas, areas, X):
        from scipy.stats import norm
        from ekdist._constants import LOG_LIKELIHOOD_MIN
        liks = np.zeros(len(X), dtype=float)
        for mu, sigma, area in zip(means, sigmas, areas):
            liks += area * norm.pdf(X, mu, sigma)
        liks = np.clip(liks, LOG_LIKELIHOOD_MIN, None)
        return float(-np.sum(np.log(liks)))

    def test_pdf_2comp_matches_loop(self):
        means, sigmas, areas = [5.0, 15.0], [1.0, 2.0], [0.4, 0.6]
        X = np.linspace(0.0, 25.0, 500)
        pdf = GaussianMixturePDF(means, sigmas, areas)
        result = pdf.pdf(pdf.theta, X)
        expected = self._loop_pdf(means, sigmas, areas, X)
        npt.assert_array_almost_equal(result, expected, decimal=12)

    def test_pdf_3comp_matches_loop(self):
        means, sigmas = [2.0, 8.0, 15.0], [0.5, 1.0, 2.0]
        areas = [0.2, 0.3, 0.5]
        X = np.linspace(-2.0, 25.0, 300)
        pdf = GaussianMixturePDF(means, sigmas, areas)
        result = pdf.pdf(pdf.theta, X)
        expected = self._loop_pdf(means, sigmas, areas, X)
        npt.assert_array_almost_equal(result, expected, decimal=12)

    def test_pdf_1comp_matches_loop(self):
        means, sigmas, areas = [10.0], [2.0], [1.0]
        X = np.linspace(0.0, 20.0, 200)
        pdf = GaussianMixturePDF(means, sigmas, areas)
        result = pdf.pdf(pdf.theta, X)
        expected = self._loop_pdf(means, sigmas, areas, X)
        npt.assert_array_almost_equal(result, expected, decimal=12)

    def test_pdf_output_shape_various_k(self):
        for k in [1, 2, 3, 5]:
            means = (np.arange(k, dtype=float) * 5.0).tolist()
            sigmas = [1.0] * k
            areas = [1.0 / k] * k
            pdf = GaussianMixturePDF(means, sigmas, areas)
            X = np.linspace(0.0, k * 5.0, 200)
            result = pdf.pdf(pdf.theta, X)
            assert result.shape == X.shape, f"shape mismatch for k={k}"

    def test_ll_2comp_matches_loop(self):
        rng = np.random.default_rng(7)
        X = np.concatenate([rng.normal(5.0, 1.0, 300), rng.normal(15.0, 2.0, 300)])
        means, sigmas, areas = [5.0, 15.0], [1.0, 2.0], [0.5, 0.5]
        pdf = GaussianMixturePDF(means, sigmas, areas)
        result = pdf.LL(pdf.theta, X)
        expected = self._loop_ll(means, sigmas, areas, X)
        npt.assert_almost_equal(result, expected, decimal=10)

    def test_ll_3comp_matches_loop(self):
        rng = np.random.default_rng(13)
        X = np.concatenate([
            rng.normal(2.0, 0.5, 200),
            rng.normal(8.0, 1.0, 200),
            rng.normal(15.0, 2.0, 200),
        ])
        means, sigmas, areas = [2.0, 8.0, 15.0], [0.5, 1.0, 2.0], [0.2, 0.3, 0.5]
        pdf = GaussianMixturePDF(means, sigmas, areas)
        result = pdf.LL(pdf.theta, X)
        expected = self._loop_ll(means, sigmas, areas, X)
        npt.assert_almost_equal(result, expected, decimal=10)

    def test_ll_negative_sigma_guard(self):
        """When the optimiser pushes sigma <= 0, LL must return a large penalty.

        theta layout for 2-comp: [mu0, mu1, sig0, sig1, area0] (last area fixed).
        Passing sig0 = -1.0 must trigger the guard.
        """
        pdf = GaussianMixturePDF([5.0, 15.0], [1.0, 2.0])
        X = np.linspace(0.0, 20.0, 100)
        # theta = [mu0, mu1, sig0, sig1, area0]; inject negative sig0
        bad_theta = np.array([5.0, 15.0, -1.0, 2.0, 0.5])
        result = pdf.LL(bad_theta, X)
        assert result == 1e10

"""Tests for ekdist.errors — ApproximateSD and LikelihoodIntervals."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import minimize

from ekdist.errors import ApproximateSD, LikelihoodIntervals
from ekdist.fitting import ExponentialPDF


@pytest.fixture(scope="module")
def fitted_state(dwell_times):
    """Fit a 2-component exponential to intervals.txt; return (res.x, pdf, asd)."""
    tau, area = [0.036, 1.1], [0.20]
    pdf = ExponentialPDF(tau, area)
    res = minimize(pdf.LL, pdf.theta, args=dwell_times, method="Nelder-Mead")
    pdf._set_theta(res.x)
    asd = ApproximateSD(res.x, pdf.LL, dwell_times)
    return res.x, pdf, asd, dwell_times


class TestApproximateSD:
    def test_hessian_shape(self, fitted_state):
        _, _, asd, _ = fitted_state
        n = 3  # 2 tau + 1 free area (3 free params)
        assert asd.hessian.shape == (n, n)

    def test_hessian_symmetric(self, fitted_state):
        _, _, asd, _ = fitted_state
        npt.assert_almost_equal(asd.hessian, asd.hessian.T, decimal=6)

    def test_hessian_values(self, fitted_state):
        _, _, asd, _ = fitted_state
        expected = np.array([
            [8475.49606292, -92.21425006, -626.06648917],
            [-92.21425006,  71.5478832,  -36.17166531],
            [-626.06648917, -36.17166531, 501.13966047],
        ])
        npt.assert_almost_equal(asd.hessian, expected, decimal=0)

    def test_covariance_values(self, fitted_state):
        _, _, asd, _ = fitted_state
        expected = np.array([
            [0.00013478, 0.00026864, 0.00018777],
            [0.00026864, 0.01504143, 0.00142128],
            [0.00018777, 0.00142128, 0.00233261],
        ])
        npt.assert_almost_equal(asd.covariance, expected, decimal=4)

    def test_sd_values(self, fitted_state):
        _, _, asd, _ = fitted_state
        expected = np.array([0.01160948, 0.1226435, 0.04829715])
        npt.assert_almost_equal(asd.sd, expected, decimal=3)

    def test_correlations_diagonal_one(self, fitted_state):
        _, _, asd, _ = fitted_state
        npt.assert_almost_equal(np.diag(asd.correlations), np.ones(3), decimal=8)

    def test_correlations_values(self, fitted_state):
        _, _, asd, _ = fitted_state
        expected = np.array([
            [1.0,        0.18867387, 0.33487997],
            [0.18867387, 1.0,        0.23994593],
            [0.33487997, 0.23994593, 1.0       ],
        ])
        npt.assert_almost_equal(asd.correlations, expected, decimal=3)

    def test_covariance_positive_semidefinite(self, fitted_state):
        _, _, asd, _ = fitted_state
        eigvals = np.linalg.eigvalsh(asd.covariance)
        assert np.all(eigvals >= -1e-10), "Covariance must be positive semi-definite"


class TestLikelihoodIntervals:
    def test_intervals_shape(self, fitted_state):
        theta, pdf, asd, data = fitted_state
        m = 2.0
        li = LikelihoodIntervals(theta, pdf, data, asd.sd, m)
        result = li.calculate()
        assert result.shape == (3, 2)

    def test_lower_less_than_mle(self, fitted_state):
        theta, pdf, asd, data = fitted_state
        li = LikelihoodIntervals(theta, pdf, data, asd.sd, m=2.0)
        result = li.calculate()
        for j in range(len(theta)):
            if result[j, 0] is not None:
                assert result[j, 0] < theta[j], f"Lower limit must be < MLE for param {j}"

    def test_upper_greater_than_mle(self, fitted_state):
        theta, pdf, asd, data = fitted_state
        li = LikelihoodIntervals(theta, pdf, data, asd.sd, m=2.0)
        result = li.calculate()
        for j in range(len(theta)):
            if result[j, 1] is not None:
                assert result[j, 1] > theta[j], f"Upper limit must be > MLE for param {j}"

    def test_interval_values(self, fitted_state):
        theta, pdf, asd, data = fitted_state
        li = LikelihoodIntervals(theta, pdf, data, asd.sd, m=2.0)
        result = li.calculate()
        expected = np.array([
            [0.019226384311366168, 0.07101933263063774],
            [0.8583999525995726,   1.3892163638906025],
            [0.10969760281850571,  0.3020372371637748],
        ])
        npt.assert_almost_equal(result, expected, decimal=2)

    def test_lcrit_below_lmax(self, fitted_state):
        theta, pdf, asd, data = fitted_state
        li = LikelihoodIntervals(theta, pdf, data, asd.sd, m=2.0)
        assert li.Lcrit < li.Lmax


class TestLikelihoodIntervalsFindLimit:
    """Unit tests for _find_limit edge cases."""

    def _make_li(self, fitted_state):
        theta, pdf, asd, data = fitted_state
        return LikelihoodIntervals(theta, pdf, data, asd.sd, m=2.0), theta

    def test_lower_limit_at_boundary_returns_boundary_not_none(self, fitted_state):
        """When LL is above Lcrit across the whole search range, the CI extends
        to the boundary.  _find_limit must return that boundary (0.0), not None."""
        li, theta = self._make_li(fitted_state)
        li._contour_LL = lambda x, idx: li.Lmax  # every point inside CI
        result = li._find_limit(0, 0.0, theta[0], factor=1.0)
        assert result == 0.0, f"Expected 0.0 (boundary), got {result!r}"

    def test_lower_limit_convergence_returns_value_not_none(self, fitted_state):
        """When bisection converges at a mid >= 0, return that value (never None).
        This tests that the dead `None if limit < 0` branch is gone."""
        li, theta = self._make_li(fitted_state)
        target = theta[0] * 0.5  # fake limit midway through search range
        # LL above Lcrit for x > target, below for x <= target
        li._contour_LL = lambda x, idx: (li.Lcrit + 0.001 if x > target else li.Lcrit - 1.0)
        result = li._find_limit(0, 0.0, theta[0], factor=1.0)
        assert result is not None
        assert result > 0.0

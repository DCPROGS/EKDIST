"""The absolute-increment rule for the Hessian.

``ApproximateSD`` has always sized one relative increment so that moving every
parameter together changes the function by 0.5% of its value. Colquhoun, Hatton
& Hawkes (2003), p. 702, describe a different rule: find the increment for
**each** parameter separately that changes the log likelihood by a specified
absolute amount, and where no such increment exists, omit that parameter's row
and column from the Hessian. ``delta_L`` selects that rule.

The tests use quadratics, whose Hessian is known exactly, so that a failure
means the estimator is wrong rather than that the reference values drifted.
"""

import numpy as np
import numpy.testing as npt
import pytest

from ekdist import errors


def quadratic(H, centre, offset=100.0):
    """A function with Hessian ``H``, minimised at ``centre``."""
    H = np.asarray(H, float)
    centre = np.asarray(centre, float)

    def f(theta, args=None):
        d = np.asarray(theta, float) - centre
        return offset + 0.5 * d @ H @ d

    return f


WELL_CONDITIONED = np.array([[4.0, 1.0], [1.0, 9.0]])
CENTRE = np.array([2.0, 3.0])


@pytest.mark.parametrize('kwargs', [{}, {'delta_L': 0.1}, {'delta_L': 2.0}])
def test_both_rules_recover_a_known_hessian(kwargs):
    asd = errors.ApproximateSD(CENTRE, quadratic(WELL_CONDITIONED, CENTRE),
                               None, **kwargs)
    npt.assert_allclose(asd.hessian, WELL_CONDITIONED, rtol=1e-6)
    npt.assert_allclose(
        asd.sd, np.sqrt(np.diag(np.linalg.inv(WELL_CONDITIONED))), rtol=1e-6)
    assert asd.dropped == ()


def test_absolute_rule_gives_the_right_correlation():
    covariance = np.linalg.inv(WELL_CONDITIONED)
    expected = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    asd = errors.ApproximateSD(CENTRE, quadratic(WELL_CONDITIONED, CENTRE),
                               None, delta_L=0.1)
    npt.assert_allclose(asd.correlations[0, 1], expected, rtol=1e-6)
    npt.assert_allclose(np.diag(asd.correlations), [1.0, 1.0], rtol=1e-9)


def test_each_increment_produces_the_requested_change():
    """That is what 'the increment needed to change L by delta_L' means."""
    delta_L = 0.1
    f = quadratic(WELL_CONDITIONED, CENTRE)
    asd = errors.ApproximateSD(CENTRE, f, None, delta_L=delta_L)
    L0 = f(CENTRE)
    for i, delta in enumerate(asd.deltas):
        moved = CENTRE.copy()
        moved[i] += delta
        change = f(moved) - L0
        # the search doubles or halves, so it lands within a factor of two
        assert delta_L <= change < 4 * delta_L, (i, change)


def test_increments_differ_between_parameters():
    """The point of the rule: a stiff parameter gets a smaller increment."""
    H = np.array([[1000.0, 0.0], [0.0, 1.0]])
    asd = errors.ApproximateSD(CENTRE, quadratic(H, CENTRE), None, delta_L=0.1)
    assert asd.deltas[0] < asd.deltas[1] / 10


# ------------------------------------------------- insensitive parameters

FLAT = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1e-14]])
CENTRE3 = np.array([2.0, 3.0, 5.0])


def test_insensitive_parameter_is_dropped():
    asd = errors.ApproximateSD(CENTRE3, quadratic(FLAT, CENTRE3), None,
                               delta_L=0.1)
    assert asd.dropped == (2,)
    assert asd.kept == (0, 1)
    assert asd.hessian.shape == (2, 2)
    assert all(isinstance(i, int) for i in asd.dropped)


def test_dropping_leaves_the_others_exact():
    """The point of omitting a row and column rather than giving up."""
    asd = errors.ApproximateSD(CENTRE3, quadratic(FLAT, CENTRE3), None,
                               delta_L=0.1)
    npt.assert_allclose(asd.hessian, FLAT[:2, :2], rtol=1e-6)
    covariance = np.linalg.inv(FLAT[:2, :2])
    npt.assert_allclose(asd.sd[:2], np.sqrt(np.diag(covariance)), rtol=1e-6)


def test_dropped_entries_are_nan_and_indexed_by_parameter():
    """sd[i] must still refer to theta[i], or a table of results misaligns."""
    asd = errors.ApproximateSD(CENTRE3, quadratic(FLAT, CENTRE3), None,
                               delta_L=0.1)
    assert asd.sd.shape == (3,)
    assert asd.correlations.shape == (3, 3)
    assert np.isnan(asd.sd[2]) and np.isnan(asd.deltas[2])
    assert np.all(np.isnan(asd.correlations[2, :]))
    assert np.all(np.isnan(asd.correlations[:, 2]))
    assert np.all(np.isfinite(asd.correlations[:2, :2]))


def test_the_flat_case_is_what_the_default_rule_cannot_do():
    """Motivation for the option, kept as a test so it stays true."""
    with pytest.raises(np.linalg.LinAlgError):
        errors.ApproximateSD(CENTRE3, quadratic(FLAT, CENTRE3), None)


def test_a_function_that_raises_is_treated_as_too_large_a_step():
    """Our HJC likelihood raises on a Q matrix it cannot handle."""
    f = quadratic(WELL_CONDITIONED, CENTRE)

    def fragile(theta, args=None):
        if abs(theta[0] - CENTRE[0]) > 0.05:
            raise ArithmeticError('too far')
        return f(theta, args)

    asd = errors.ApproximateSD(CENTRE, fragile, None, delta_L=1e-4)
    assert asd.deltas[0] <= 0.05
    assert np.isfinite(asd.sd[0])


def test_zero_valued_parameter_does_not_give_a_zero_increment():
    centre = np.array([0.0, 3.0])
    asd = errors.ApproximateSD(centre, quadratic(WELL_CONDITIONED, centre),
                               None, delta_L=0.1)
    assert asd.deltas[0] > 0
    npt.assert_allclose(asd.hessian, WELL_CONDITIONED, rtol=1e-6)

"""Backward-compatible tests for ExponentialPDF (originally from ekdist.exponentials).

Preserved and updated to use the new ekdist.fitting module.
"""

import numpy as np
import numpy.testing as npt

from ekdist.fitting import ExponentialPDF


def test_area_none():
    tau_in = [0.1, 1.0]
    pdf = ExponentialPDF(tau=tau_in)
    area = np.ones(2) / 2
    npt.assert_almost_equal(pdf.area, area)
    pars = np.concatenate((np.asarray(tau_in), area))
    npt.assert_almost_equal(pdf.pars, pars)
    fixed = [False, False, False, True]
    assert fixed == pdf.fixed
    theta = pars[:-1]
    npt.assert_almost_equal(pdf.theta, theta)


def test_insert_theta():
    tau_in = [0.1, 1.0]
    pdf = ExponentialPDF(tau=tau_in)
    theta_in = [0.2, 2.0, 0.3]
    pdf.theta = theta_in
    pars_in = [0.2, 2.0, 0.3, 0.7]
    npt.assert_almost_equal(pdf.pars, pars_in)


def test_fixed_pars():
    tau_in = [0.1, 1.0]
    pdf = ExponentialPDF(tau=tau_in)
    pdf.fixed[1] = True
    fixed = [False, True, False, True]
    assert fixed == pdf.fixed
    theta_in = [0.2, 0.3]
    pdf.theta = theta_in
    pars_in = [0.2, 1.0, 0.3, 0.7]
    npt.assert_almost_equal(pdf.pars, pars_in)
    npt.assert_almost_equal(pdf.theta, theta_in)

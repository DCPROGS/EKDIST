
import os, sys
import math
from scipy.optimize import minimize, bisect
import numpy as np
import numpy.testing as npt
from numpy import linalg as nplin

from ekdist import eklib
from ekdist import errors
from ekdist import exponentials

class TestErrorCalculation:
    def setup_method(self):
        self.infile = "./tests/intervals.txt"
        self.intervals = np.loadtxt(self.infile)
        self.tau, self.area = [0.036, 1.1], [0.20]
        self.theta = self.tau + self.area
        self.epdf = exponentials.ExponentialPDF(self.tau, self.area)
        self.res = minimize(self.epdf.LL, self.epdf.theta, 
                            args=self.intervals, 
                            method='Nelder-Mead')

        self.asd = errors.ApproximateSD(self.res.x, self.epdf.LL, self.intervals)
        #self.hess = asd.hess
        #self.hess = eklib.hessian(self.res.x, self.epdf.LL, self.intervals)
        #self.cov = eklib.covariance_matrix(self.theta, eklib.LLexp, self.intervals)
        #self.cov = asd.covariance
        #self.cov = nplin.inv(self.hess)
        #self.appSD = asd.sd #np.sqrt(self.cov.diagonal())
        #self.corr = asd.correlations #eklib.correlation_matrix(self.cov)
        m = 2.0 # corresponds roughly to 2 SD 
        self.likints = errors.LikelihoodIntervals(self.res.x, self.epdf, 
                                                  self.intervals, self.asd.sd, m)

    def test_infile_exists(self):
        assert os.path.isfile(self.infile)

    def test_interval_number(self):
        assert len(self.intervals)  == 125

    def test_expLogLik(self):
        npt.assert_almost_equal(self.epdf.LL(self.theta, self.intervals), 87.31806715582867)

    def test_estimates(self):
        npt.assert_almost_equal(self.res.x, np.array([0.03700718, 1.07302608, 0.19874548]))

    def test_hessian(self):
        npt.assert_almost_equal(self.asd.hessian, [[8475.49606292, -92.21425006, -626.06648917],
                                            [ -92.21425006,  71.5478832,   -36.17166531],
                                            [-626.06648917, -36.17166531,  501.13966047]])
    def test_covariance(self):
        npt.assert_almost_equal(self.asd.covariance, [[0.00013478, 0.00026864, 0.00018777],
                                           [0.00026864, 0.01504143, 0.00142128],
                                           [0.00018777, 0.00142128, 0.00233261]])

    def test_SD(self):
        npt.assert_almost_equal(self.asd.sd, [0.01160948, 0.1226435,  0.04829715])

    def test_correlations(self):
        npt.assert_almost_equal(self.asd.correlations, np.array([[1.        , 0.18867387, 0.33487997],
                                                     [0.18867387, 1.        , 0.23994593],
                                                     [0.33487997, 0.23994593, 1.        ]]))

    def test_likelihood_intervals(self):
        npt.assert_almost_equal(self.likints.calculate(), np.array([[0.019226384311366168, 0.07101933263063774],
                                                        [0.8583999525995726, 1.3892163638906025],
                                                        [0.10969760281850571, 0.3020372371637748]]))

    
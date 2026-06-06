"""Named constants used throughout ekdist.

All magic numbers that appear in algorithms are defined here with an explanation
of their origin or meaning.
"""

# Interval properties flag — bit 3 set means the interval is unusable/bad.
# From the SCAN file format specification (iprops field).
BAD_FLAG = 8

# Amplitude tolerance for deciding whether two successive open sub-levels are
# part of the same conductance sublevel during resolution imposition.
# Value from Colquhoun lab Fortran original.
AMPLITUDE_TOLERANCE = 1e-5

# Gaussian filter 10-90% rise time coefficient.
# Rise time T_r = FILTER_RISE_COEFF / fc, where fc is the -3 dB cut-off
# frequency (Hz).  Derived from the step-response of a Gaussian filter:
# T_r = sqrt(2 * ln(9)) / (2π * fc) ≈ 0.3321 / fc.
FILTER_RISE_COEFF = 0.3321

# Minimum value accepted inside a log-likelihood sum.  Guards against log(0).
LOG_LIKELIHOOD_MIN = 1e-37

# Penalty returned by LL when parameters are physically invalid (tau <= 0,
# area < 0).  Large enough to steer Nelder-Mead away without causing overflow.
LOG_LIKELIHOOD_PENALTY = 1e10

# Finite-difference step factor for Hessian estimation (as fraction of θ).
HESSIAN_STEP_FACTOR = 1e-4

# Number of bins per decade for log-scale dwell-time histograms, thresholded
# by sample size (from Colquhoun lab Fortran original).
_HIST_BINS_THRESHOLDS = [(0, 300, 5), (300, 1000, 8), (1000, 3000, 10)]
HIST_BINS_DEFAULT = 12

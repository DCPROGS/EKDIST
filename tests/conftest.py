"""Shared pytest fixtures for the ekdist test suite."""

from pathlib import Path

import numpy as np
import pytest

# Absolute path to the test data directory, regardless of where pytest is run from.
DATA_DIR = Path(__file__).parent


@pytest.fixture(scope="session")
def scn_file() -> Path:
    """Path to the AChsim.scn test file."""
    p = DATA_DIR / "AChsim.scn"
    assert p.exists(), f"Test SCN file not found: {p}"
    return p


@pytest.fixture(scope="session")
def scn_file_exp() -> Path:
    """Path to the experimental SCN test file."""
    p = DATA_DIR / "181003S8.SCN"
    assert p.exists(), f"Test SCN file not found: {p}"
    return p


@pytest.fixture(scope="session")
def glyr_file() -> Path:
    """Path to the glycine-receptor experimental SCN file with real amplitude data."""
    p = DATA_DIR / "glyr_experimental.scn"
    assert p.exists(), f"Test SCN file not found: {p}"
    return p


@pytest.fixture(scope="session")
def glyr_record(glyr_file):
    """SingleChannelRecord from glyr_experimental.scn with tres=30 µs."""
    from ekdist.record import SingleChannelRecord

    rec = SingleChannelRecord.from_scn(glyr_file)
    rec.tres = 30e-6
    return rec


@pytest.fixture(scope="session")
def intervals_file() -> Path:
    """Path to the intervals.txt test file."""
    p = DATA_DIR / "intervals.txt"
    assert p.exists(), f"Test intervals file not found: {p}"
    return p


@pytest.fixture(scope="session")
def dwell_times(intervals_file) -> np.ndarray:
    """125 synthetic dwell times from intervals.txt."""
    return np.loadtxt(intervals_file)


@pytest.fixture(scope="session")
def fitted_exp_pdf(dwell_times):
    """Pre-fitted 2-component ExponentialPDF on intervals.txt data."""
    from scipy.optimize import minimize
    from ekdist.fitting import ExponentialPDF

    tau, area = [0.036, 1.1], [0.20]
    pdf = ExponentialPDF(tau, area)
    res = minimize(pdf.LL, pdf.theta, args=dwell_times, method="Nelder-Mead")
    pdf._set_theta(res.x)
    return pdf, res


@pytest.fixture(scope="module")
def loaded_record(scn_file):
    """SingleChannelRecord loaded from AChsim.scn with tres=30 µs."""
    from ekdist.record import SingleChannelRecord

    rec = SingleChannelRecord.from_scn(scn_file)
    rec.tres = 30e-6
    return rec


# ------------------------------------------------------------------ #
# Synthetic interval arrays for unit tests                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def simple_intervals():
    """One open + one shut interval, both resolvable."""
    itint = np.array([0.1, 0.05])
    iampl = np.array([1.0, 0.0])
    iprop = np.array([0, 0])
    return itint, iampl, iprop


@pytest.fixture
def single_open_record():
    """Three intervals: long open (100 ms), long shut (50 ms).
    After resolution (tres=1 ms) → one open period of 100 ms.
    """
    itint = np.array([0.1, 0.05])
    iampl = np.array([1.0, 0.0])
    iprop = np.array([0, 0])
    return itint, iampl, iprop


@pytest.fixture
def two_open_record():
    """open-shut-open-shut: should yield two open periods."""
    itint = np.array([0.05, 0.02, 0.08, 0.03])
    iampl = np.array([1.0, 0.0, 1.0, 0.0])
    iprop = np.array([0, 0, 0, 0])
    return itint, iampl, iprop


@pytest.fixture
def short_shut_between_opens():
    """open-short_shut-open: short shut should be concatenated with opens."""
    itint = np.array([0.05, 0.0001, 0.08, 0.04])
    iampl = np.array([1.0, 0.0, 1.0, 0.0])
    iprop = np.array([0, 0, 0, 0])
    return itint, iampl, iprop

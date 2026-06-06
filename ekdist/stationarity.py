"""Stationarity tests for single-channel interval sequences.

Two classical tests for temporal stationarity of a sequence of dwell-time
intervals recorded in the order they occurred:

* :func:`runs_test` — Wald-Wolfowitz runs test on deviations from the median.
  Detects both positive autocorrelation (too few runs) and oscillation (too
  many runs).

* :func:`cox_lewis_test` — Cox-Lewis test for monotonic trend in event
  occurrence times.  U < 0.5 indicates intervals getting longer over time
  (declining channel activity); U > 0.5 indicates intervals shortening.

Both return a :class:`StationarityResult` dataclass with the test statistic
(Z-score), two-tailed p-value, sample size, and a plain-English verdict.

Usage::

    from ekdist.stationarity import runs_test, cox_lewis_test

    opens = rec.periods.open_intervals
    print(runs_test(opens))
    print(cox_lewis_test(opens))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm as _norm


@dataclass
class StationarityResult:
    """Result of a stationarity hypothesis test.

    Attributes
    ----------
    statistic:
        Z-score (test statistic).
    p_value:
        Two-tailed p-value under the null hypothesis of stationarity.
    n:
        Number of observations used.
    name:
        Human-readable test name.
    description:
        Plain-English verdict including direction of any detected trend.
    """

    statistic: float
    p_value: float
    n: int
    name: str
    description: str

    def __repr__(self) -> str:
        return (
            f"{self.name} (n={self.n})\n"
            f"  statistic = {self.statistic:.4f}\n"
            f"  p-value   = {self.p_value:.4f}\n"
            f"  {self.description}"
        )


def runs_test(X: np.ndarray) -> StationarityResult:
    """Wald-Wolfowitz runs test for randomness of interval sequence.

    Counts the number of maximal runs of consecutive values above or below
    the sample median.  Under the null hypothesis of stationarity, the run
    count follows approximately a Normal distribution.

    A significant positive Z (too many runs) indicates oscillation; a
    significant negative Z (too few runs) indicates a systematic trend or
    drift.

    Parameters
    ----------
    X:
        1-D array of interval durations (seconds) in recording order.

    Returns
    -------
    StationarityResult

    Raises
    ------
    ValueError
        If fewer than 10 observations are provided, or if all values lie on
        the same side of the median (variance formula is undefined).
    """
    X = np.asarray(X, dtype=float)
    if len(X) < 4:
        raise ValueError(
            f"runs_test requires at least 4 observations; got {len(X)}"
        )

    median = float(np.median(X))
    # Drop ties to avoid undefined run boundaries
    mask = X != median
    above = X[mask] > median   # True = above, False = below
    n = len(above)

    n1 = int(np.sum(above))    # count above median
    n2 = n - n1                # count below median

    if n1 == 0 or n2 == 0:
        raise ValueError(
            "All non-tie values are on the same side of the median; "
            "runs test is undefined."
        )

    # Count runs: a new run starts whenever the sign changes
    runs = int(1 + np.sum(above[1:] != above[:-1]))

    # Large-sample Normal approximation (n >= 10)
    mu_R = 2.0 * n1 * n2 / (n1 + n2) + 1.0
    var_R = (
        2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)
        / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    )

    Z = (runs - mu_R) / math.sqrt(var_R)
    p_value = 2.0 * float(_norm.sf(abs(Z)))

    if p_value < 0.05:
        verdict = "Non-stationary (p < 0.05): evidence of trend or systematic change."
    else:
        verdict = "Stationary (p >= 0.05): no significant trend detected."

    return StationarityResult(
        statistic=float(Z),
        p_value=float(p_value),
        n=len(X),
        name="Runs test",
        description=verdict,
    )


def cox_lewis_test(X: np.ndarray) -> StationarityResult:
    """Cox-Lewis trend test for monotonic change in interval durations.

    Under the null hypothesis that the intervals are identically distributed,
    the normalised event-time statistic U is approximately Uniform(0, 1):

        U = mean(t_k / t_n, k = 1 … n-1)

    where t_k = T_1 + … + T_k is the cumulative time of the k-th event and
    t_n is the total recording duration.  The Z-score is:

        Z = (U − 0.5) · √(12 · (n − 1))

    U < 0.5 (Z < 0) → intervals getting longer over time (activity declining).
    U > 0.5 (Z > 0) → intervals getting shorter over time (activity increasing).

    Parameters
    ----------
    X:
        1-D array of interval durations (seconds) in recording order.

    Returns
    -------
    StationarityResult

    Raises
    ------
    ValueError
        If fewer than 2 observations are provided, or if the total duration
        is zero.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        raise ValueError(
            f"cox_lewis_test requires at least 2 observations; got {n}"
        )

    cumtimes = np.cumsum(X)
    total = float(cumtimes[-1])

    if total == 0.0:
        raise ValueError(
            "Total interval duration is zero; cannot compute Cox-Lewis test."
        )

    # U = mean of (t_1/t_n, t_2/t_n, …, t_{n-1}/t_n)
    U = float(np.mean(cumtimes[:-1] / total))

    # Z ~ N(0, 1) under H0
    Z = (U - 0.5) * math.sqrt(12.0 * (n - 1))
    p_value = 2.0 * float(_norm.sf(abs(Z)))

    if p_value < 0.05:
        if U < 0.5:
            direction = "Intervals getting longer (activity declining)."
        else:
            direction = "Intervals getting shorter (activity increasing)."
        verdict = f"Non-stationary (p < 0.05): {direction}"
    else:
        verdict = "Stationary (p >= 0.05): no significant trend detected."

    return StationarityResult(
        statistic=float(Z),
        p_value=float(p_value),
        n=n,
        name="Cox-Lewis trend test",
        description=verdict,
    )

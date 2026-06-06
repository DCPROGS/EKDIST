"""Serial autocorrelation and open-shut interval correlation.

Two analyses ported from the Fortran CORCAL.FOR subroutine in the Colquhoun
lab EKDIST suite:

* :func:`serial_correlation` — Lagged autocorrelation coefficients r₁ … rₖ
  for a sequence of dwell-time intervals.  Outlying intervals are handled by
  splitting the sequence into sub-segments and pooling the segment-level
  Fisher Z-transforms into a single pooled r and normal deviate, following
  the 1995 revision of CORCAL.FOR.

* :func:`open_shut_correlation` — Pearson correlation between each open
  interval and its adjacent shut interval (or any two paired interval
  sequences).  Converts to a Fisher Z normal deviate for significance
  testing.  This is a standard Colquhoun lab diagnostic: a non-zero
  correlation between open and shut times is inconsistent with a simple
  two-state model.

Both return dataclass results with all intermediate values.

Usage::

    from ekdist.correlation import serial_correlation, open_shut_correlation

    opens = rec.periods.open_intervals
    shuts = rec.periods.shut_intervals[1:-1]   # interior shuts only

    sc = serial_correlation(opens, max_lag=10)
    print(sc.summary())

    osc = open_shut_correlation(opens, shuts)
    print(osc)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm as _norm


# =========================================================================== #
# Result dataclasses                                                            #
# =========================================================================== #

@dataclass
class CorrelationResult:
    """Result of a single correlation test.

    Attributes
    ----------
    r:
        Pearson (or Fisher-pooled) correlation coefficient.
    z:
        Fisher Z normal deviate (use for significance; not the Fisher Z
        transform itself — that is an intermediate).
    df:
        Degrees of freedom used for the normal deviate.
    n:
        Number of observations (pairs) used.
    p_value:
        Two-tailed p-value under H₀: r = 0.
    name:
        Human-readable test name.
    description:
        Plain-English verdict.
    """

    r: float
    z: float
    df: int
    n: int
    p_value: float
    name: str
    description: str

    def __repr__(self) -> str:
        return (
            f"{self.name} (n={self.n})\n"
            f"  r           = {self.r:.4f}\n"
            f"  normal dev. = {self.z:.4f}  (df={self.df})\n"
            f"  p-value     = {self.p_value:.4f}\n"
            f"  {self.description}"
        )


@dataclass
class SerialCorrelationResult:
    """Results of serial autocorrelation for lags 1 … max_lag.

    Attributes
    ----------
    lags:
        1-D integer array ``[1, 2, …, max_lag]``.
    r:
        Pooled correlation coefficient at each lag.
    z:
        Fisher Z normal deviate at each lag.
    df:
        Total degrees of freedom at each lag (sum across segments).
    n_segments:
        Number of sub-segments pooled at each lag (>1 when outliers present).
    n_total:
        Number of non-outlier intervals used for mean and variance.
    outlier_limit:
        Upper limit used (``inf`` when no outlier exclusion was requested).
    """

    lags: np.ndarray
    r: np.ndarray
    z: np.ndarray
    df: np.ndarray
    n_segments: np.ndarray
    n_total: int
    outlier_limit: float

    def summary(self) -> str:
        """Formatted table of correlation coefficients."""
        limit_str = "∞" if math.isinf(self.outlier_limit) else f"{self.outlier_limit:.4g}"
        lines = [
            f"Serial autocorrelation  (n={self.n_total}, outlier limit={limit_str})",
            f"{'Lag':>5}  {'r':>10}  {'deviate':>10}  {'segments':>9}  {'df':>8}",
        ]
        for i, lag in enumerate(self.lags):
            lines.append(
                f"{int(lag):5d}  {self.r[i]:10.4f}  {self.z[i]:10.4f}"
                f"  {self.n_segments[i]:9d}  {self.df[i]:8d}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# =========================================================================== #
# Serial autocorrelation                                                        #
# =========================================================================== #

def serial_correlation(
    intervals: np.ndarray,
    max_lag: int = 50,
    outlier_limit: float | None = None,
) -> SerialCorrelationResult:
    """Serial autocorrelation coefficients r₁ … r_max_lag for a dwell-time sequence.

    Implements the Colquhoun lab CORCAL algorithm (1995 revision): when an
    interval exceeding *outlier_limit* is encountered, the current sub-segment
    is closed, its Fisher Z-transform is recorded, and a fresh sub-segment
    begins after the outlier.  The per-segment Z values are pooled by a
    degrees-of-freedom weighted mean to give a single pooled r and normal
    deviate for each lag.

    The denominator for each within-segment Pearson numerator is the global
    sum-of-squared deviations ``SSY = Σ(yᵢ − ȳ)²`` computed from *all*
    non-outlier intervals (not segment-local variance).  This matches the
    Fortran original.

    Parameters
    ----------
    intervals:
        1-D array of dwell-time values in recording order (seconds, or any
        consistent unit).
    max_lag:
        Highest lag to compute.  Default 50.
    outlier_limit:
        If given, any window [i, i+lag] that contains a value above this
        threshold terminates the current sub-segment and starts a new one
        after the outlier.  Must be positive.  When ``None`` (default) no
        outlier handling is applied.

    Returns
    -------
    SerialCorrelationResult

    Raises
    ------
    ValueError
        If fewer than 10 intervals are provided, or if *outlier_limit* is not
        positive.
    """
    X = np.asarray(intervals, dtype=float)
    n_total = len(X)
    if n_total < 5:
        raise ValueError(
            f"serial_correlation requires at least 5 intervals; got {n_total}"
        )
    if outlier_limit is not None and outlier_limit <= 0.0:
        raise ValueError(
            f"outlier_limit must be positive; got {outlier_limit}"
        )

    yhi = float(outlier_limit) if outlier_limit is not None else math.inf

    # Global mean and SSY from all non-outlier values
    # (this is the fixed denominator used in CORCAL for all lags)
    mask = X <= yhi
    X_good = X[mask]
    n_good = int(mask.sum())
    if n_good < 4:
        raise ValueError(
            "Too few non-outlier values to compute autocorrelation "
            f"(need ≥ 4, got {n_good})"
        )
    ybar = float(X_good.mean())
    ssy = float(np.sum((X_good - ybar) ** 2))

    N = len(X)
    lags = np.arange(1, max_lag + 1, dtype=int)
    r_arr = np.zeros(max_lag)
    z_arr = np.zeros(max_lag)
    df_arr = np.zeros(max_lag, dtype=int)
    nseg_arr = np.zeros(max_lag, dtype=int)

    for k_idx, lag in enumerate(lags):
        seg_z: list[float] = []
        seg_df: list[int] = []

        i = 0
        while i + lag < N:
            # Start a new sub-segment
            n_seg = 0
            sxy = 0.0

            # Inner loop: accumulate pairs until outlier or end of data
            while i + lag < N:
                # All values in window [i, i+lag] must be below the limit
                window_ok = True
                for m in range(i, i + lag + 1):
                    if X[m] > yhi:
                        window_ok = False
                        break

                if window_ok:
                    sxy += (X[i] - ybar) * (X[i + lag] - ybar)
                    n_seg += 1
                    i += 1
                else:
                    # Outlier found: close current sub-segment
                    if n_seg > 3:
                        r_seg = sxy / ssy
                        r_seg = max(-1.0 + 1e-15, min(1.0 - 1e-15, r_seg))
                        z_seg = 0.5 * math.log((1.0 + r_seg) / (1.0 - r_seg))
                        seg_z.append(z_seg)
                        seg_df.append(n_seg - 3)
                    # Advance past the outlier and restart
                    i += 1
                    break  # back to outer while with fresh n_seg, sxy
            else:
                # Inner while exited because i+lag >= N (end of data)
                if n_seg > 3:
                    r_seg = sxy / ssy
                    r_seg = max(-1.0 + 1e-15, min(1.0 - 1e-15, r_seg))
                    z_seg = 0.5 * math.log((1.0 + r_seg) / (1.0 - r_seg))
                    seg_z.append(z_seg)
                    seg_df.append(n_seg - 3)
                break  # outer while — no more data for this lag

        # Pool Fisher Z values across sub-segments (df-weighted mean)
        if seg_z:
            total_df = sum(seg_df)
            if total_df > 0:
                zbar = sum(d * z for d, z in zip(seg_df, seg_z)) / total_df
                # Normal deviate = zbar / sqrt(var_z) = zbar * sqrt(total_df)
                dev = zbar * math.sqrt(float(total_df))
                rbar = math.tanh(zbar)
            else:
                zbar = dev = rbar = 0.0
        else:
            total_df = 0
            dev = rbar = 0.0

        r_arr[k_idx] = rbar
        z_arr[k_idx] = dev
        df_arr[k_idx] = total_df
        nseg_arr[k_idx] = len(seg_z)

    return SerialCorrelationResult(
        lags=lags,
        r=r_arr,
        z=z_arr,
        df=df_arr,
        n_segments=nseg_arr,
        n_total=n_good,
        outlier_limit=yhi,
    )


# =========================================================================== #
# Open-shut interval correlation                                                 #
# =========================================================================== #

def open_shut_correlation(
    open_intervals: np.ndarray,
    shut_intervals: np.ndarray,
) -> CorrelationResult:
    """Pearson correlation between paired open and shut interval durations.

    Computes the correlation between each open interval and its adjacent shut
    interval.  A non-zero value is inconsistent with a simple two-state
    (one-open, one-shut) Markov model and is a standard Colquhoun lab
    diagnostic for multi-state kinetics.

    Implements the ``IDTYPE=11`` branch of CORCAL.FOR: the Fisher Z-transform
    is used to obtain a normal deviate, and both the untransformed correlation
    and its significance are reported.

    Parameters
    ----------
    open_intervals:
        1-D array of open-interval durations in recording order (seconds).
    shut_intervals:
        1-D array of shut-interval durations, same length as *open_intervals*.
        Element ``shut_intervals[i]`` is the shut interval adjacent to
        ``open_intervals[i]``.

    Returns
    -------
    CorrelationResult

    Raises
    ------
    ValueError
        If the two arrays differ in length, or if fewer than 4 pairs are
        provided (minimum for a meaningful correlation).
    """
    op = np.asarray(open_intervals, dtype=float)
    sh = np.asarray(shut_intervals, dtype=float)

    if len(op) != len(sh):
        raise ValueError(
            f"open_intervals and shut_intervals must have the same length; "
            f"got {len(op)} and {len(sh)}"
        )
    n = len(op)
    if n < 4:
        raise ValueError(
            f"open_shut_correlation requires at least 4 pairs; got {n}"
        )

    # Pearson correlation — matches Fortran formula:
    # r = Σ(xi-x̄)(yi-ȳ) / ((n-1) * SD_x * SD_y) = np.corrcoef()[0,1]
    op_mean = float(op.mean())
    sh_mean = float(sh.mean())
    ss_op = float(np.sum((op - op_mean) ** 2))
    ss_sh = float(np.sum((sh - sh_mean) ** 2))
    sp = float(np.sum((op - op_mean) * (sh - sh_mean)))

    denom = math.sqrt(ss_op * ss_sh)
    if denom == 0.0:
        r = 0.0
    else:
        r = sp / denom

    # Clamp to avoid log(0) at perfect ±1 correlation
    r_clamped = max(-1.0 + 1e-15, min(1.0 - 1e-15, r))

    # Fisher Z-transform → normal deviate
    # df = n - 3;  dev = z_F * sqrt(df)  [= z_F / sqrt(1/df)]
    df = n - 3
    z_fisher = 0.5 * math.log((1.0 + r_clamped) / (1.0 - r_clamped))
    dev = z_fisher * math.sqrt(float(df)) if df > 0 else 0.0

    p_value = 2.0 * float(_norm.sf(abs(dev)))

    if p_value < 0.05:
        direction = "positive" if r > 0 else "negative"
        description = (
            f"Significant {direction} open-shut correlation (p < 0.05): "
            "inconsistent with a simple two-state model."
        )
    else:
        description = (
            "No significant open-shut correlation (p ≥ 0.05): "
            "consistent with a two-state model."
        )

    return CorrelationResult(
        r=r,
        z=dev,
        df=df,
        n=n,
        p_value=p_value,
        name="Open-shut interval correlation",
        description=description,
    )

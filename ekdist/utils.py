"""Utility functions for single-channel analysis.

All physical constants and filter coefficients are named explicitly; see
``_constants.py`` for their derivations.
"""

from __future__ import annotations

import numpy as np

from ekdist._constants import FILTER_RISE_COEFF


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Valid sliding-window mean via cumulative sum.

    Unlike :func:`moving_average`, this function returns only the *valid*
    portion of the convolution — no edge padding.  The output has length
    ``len(x) - w + 1``, with element ``i`` equal to the mean of
    ``x[i : i+w]``.

    Parameters
    ----------
    x:
        Input 1-D array.
    w:
        Window size (number of points).  Must satisfy ``1 <= w <= len(x)``.

    Returns
    -------
    np.ndarray
        Array of length ``len(x) - w + 1``.

    Raises
    ------
    ValueError
        If *w* is not in ``[1, len(x)]``.
    """
    x = np.asarray(x, dtype=float)
    if w < 1:
        raise ValueError(f"Window w must be >= 1; got {w}")
    if w > len(x):
        raise ValueError(
            f"Window w={w} exceeds sequence length {len(x)}"
        )
    cs = np.empty(len(x) + 1)
    cs[0] = 0.0
    np.cumsum(x, out=cs[1:])
    return (cs[w:] - cs[:-w]) / w


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Compute an *n*-point moving average via convolution.

    Parameters
    ----------
    x:
        Input 1-D array.
    n:
        Window size (number of points).

    Returns
    -------
    np.ndarray
        Array of the same length as *x*.
    """
    x = np.asarray(x, dtype=float)
    if n < 1:
        raise ValueError(f"Window size n must be >= 1; got {n}")
    weights = np.ones(n) / n
    a = np.convolve(x, weights, mode="full")[: len(x)]
    a[:n] = a[n]
    return a


def moving_average_open_shut_popen(
    open_intervals: np.ndarray,
    shut_intervals: np.ndarray,
    window: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute moving averages of open, shut, and Popen sequences.

    Parameters
    ----------
    open_intervals:
        Open period durations (seconds).
    shut_intervals:
        Shut period durations (seconds).
    window:
        Moving average window size.

    Returns
    -------
    opma, shma, poma : np.ndarray
        Moving-averaged open, shut, and Popen (all same length, trimmed to
        the overlap region of the two input sequences).
    """
    opma = moving_average(open_intervals, window)[window - 1 :]
    shma = moving_average(shut_intervals, window)[window - 1 :]
    poma = opma / (opma + shma)
    return opma, shma, poma


def filter_risetime(fc: float) -> float:
    """10-90% rise time for a Gaussian filter with -3 dB cut-off *fc* (Hz).

    Uses the relation T_r = FILTER_RISE_COEFF / fc, where
    FILTER_RISE_COEFF ≈ 0.3321 is derived from the Gaussian step-response.

    Parameters
    ----------
    fc:
        Filter cut-off frequency in Hz.

    Returns
    -------
    float
        Rise time in seconds.
    """
    if fc <= 0:
        raise ValueError(f"Filter cut-off frequency must be positive; got {fc}")
    return FILTER_RISE_COEFF / fc


def amplitudes_openings_longer_than(
    rtint: np.ndarray,
    rampl: np.ndarray,
    fc: float,
    n: float = 2.0,
) -> np.ndarray:
    """Extract amplitudes of openings longer than *n* filter rise-times.

    Parameters
    ----------
    rtint:
        Resolved interval durations (seconds).
    rampl:
        Resolved interval amplitudes (pA; 0 = shut).
    fc:
        Filter -3 dB cut-off frequency (Hz).
    n:
        Minimum length as a multiple of the filter rise time.

    Returns
    -------
    np.ndarray
        Absolute amplitudes of openings longer than ``n * filter_risetime(fc)``.
    """
    rtint = np.asarray(rtint, dtype=float)
    rampl = np.asarray(rampl, dtype=float)
    is_open = np.abs(rampl) > 0.0
    open_durations = rtint[is_open]
    open_amplitudes = rampl[is_open]
    threshold = n * filter_risetime(fc)
    long_enough = open_durations > threshold
    return np.abs(open_amplitudes[long_enough])

"""Temporal resolution imposition for idealised single-channel records.

The core algorithm is `impose_resolution`, a pure function that takes raw
interval arrays and returns resolution-filtered arrays.  Extracting it as a
pure function makes it independently testable and reusable.

Numerical regression tests against AChsim.scn (tres=30 µs) are in
tests/test_resolution.py::TestAChSimNumericalRegression (added 2026-06-04).
Values were pinned from our Python implementation; a DOSBox run of the
Fortran ekdist.exe to cross-validate remains a future TODO.
"""

from __future__ import annotations

import logging

import numpy as np

from ekdist._constants import AMPLITUDE_TOLERANCE, BAD_FLAG

logger = logging.getLogger(__name__)


def impose_resolution(
    itint: np.ndarray,
    iampl: np.ndarray,
    iprop: np.ndarray,
    tres: float,
    *,
    is_simulated: bool = False,
    badopen: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply temporal resolution to raw idealised single-channel intervals.

    Intervals shorter than *tres* are concatenated with adjacent intervals
    according to the rules described below.  The result always begins and ends
    with a resolvable open period.

    Rules
    -----
    (1) A concatenated **shut** period starts with a good resolvable shutting
        and ends when the first good resolvable opening is found.  Its length
        is the sum of all durations before that opening; amplitude = 0.

    (2) A concatenated **open** period starts with a good resolvable opening
        and ends when a resolvable interval with a different amplitude (or a
        shut) is found.  Its length is the sum of all concatenated durations;
        its amplitude is the time-weighted mean of all concatenated open
        intervals (for experimental records) or the amplitude of the first
        opening (for simulated records).

    The first interval in each concatenated group must be resolvable but may
    carry a bad flag, in which case the entire concatenated group is flagged bad.

    Parameters
    ----------
    itint:
        Interval durations in seconds.
    iampl:
        Amplitudes in pA (0 = shut).
    iprop:
        Integer property flags; flag >= BAD_FLAG means the interval is unusable.
    tres:
        Temporal resolution in seconds.  Intervals shorter than this are
        concatenated.
    is_simulated:
        Set True for simulated records (disables amplitude-change detection for
        open-open transitions and uses direct amplitude instead of mean).
    badopen:
        If > 0, any resolved open period longer than this value (in seconds) is
        marked bad (flag = BAD_FLAG) in the output.

    Returns
    -------
    rtint, rampl, rprop : np.ndarray
        Resolved intervals, amplitudes, and flags.  Arrays start and end with
        an open period.

    Raises
    ------
    ValueError
        If arrays have inconsistent lengths, if tres < 0, or if no resolvable
        usable interval exists.
    """
    itint = np.asarray(itint, dtype=float)
    iampl = np.asarray(iampl, dtype=float)
    iprop = np.asarray(iprop, dtype=int).copy()

    if not (len(itint) == len(iampl) == len(iprop)):
        raise ValueError("itint, iampl, and iprop must have the same length")
    if len(itint) == 0:
        raise ValueError("Interval arrays are empty")
    if tres < 0:
        raise ValueError(f"tres must be non-negative; got {tres}")

    # Mark negative durations unusable
    iprop[itint < 0.0] = BAD_FLAG

    # Find first resolvable AND usable interval to start from
    resolvable = itint > tres
    usable = iprop < BAD_FLAG
    valid_indices = np.where(resolvable & usable)[0]
    if len(valid_indices) == 0:
        raise ValueError("No resolvable usable interval found in the record")

    n = int(valid_indices[0])

    # ------------------------------------------------------------------ #
    # Initialise accumulation variables for the first good interval        #
    # ------------------------------------------------------------------ #
    ttemp: float = float(itint[n])
    otemp: int = int(iprop[n])
    isopen: bool = iampl[n] != 0

    if not isopen:
        atemp: float = 0.0
    elif is_simulated:
        atemp = float(iampl[n])
    else:
        atemp = float(iampl[n]) * float(itint[n])

    n += 1

    rtint_list: list[float] = []
    rampl_list: list[float] = []
    rprop_list: list[int] = []

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    while n < len(itint):
        t_n = float(itint[n])
        a_n = float(iampl[n])
        o_n = int(iprop[n])

        if t_n < tres:
            # ---- Unresolvable interval --------------------------------- #
            is_last = n == len(itint) - 1
            if is_last and a_n == 0 and isopen:
                # Special case: last interval is an unresolvable shut while
                # we are currently accumulating an open period.  Save the
                # open period now, then record this short shut as bad.
                rtint_list.append(ttemp)
                rampl_list.append(atemp / ttemp)
                rprop_list.append(otemp)
                isopen = False
                ttemp = t_n
                atemp = 0.0
                otemp = BAD_FLAG
            else:
                ttemp += t_n
                if o_n >= BAD_FLAG:
                    otemp = o_n
                if isopen:
                    atemp += a_n * t_n
        else:
            if a_n == 0:
                # ---- Resolvable shut ----------------------------------- #
                if not isopen:
                    # Previous was shut: extend shut period
                    ttemp += t_n
                    if o_n >= BAD_FLAG:
                        otemp = o_n
                else:
                    # Previous was open: save open, start shut
                    rtint_list.append(ttemp)
                    if is_simulated:
                        rampl_list.append(atemp)
                    else:
                        rampl_list.append(atemp / ttemp)
                    _append_open_flag(rtint_list, rprop_list, otemp, badopen)
                    ttemp = t_n
                    otemp = o_n
                    isopen = False
            else:
                # ---- Resolvable open ----------------------------------- #
                if not isopen:
                    # Previous was shut: save shut, start open
                    rtint_list.append(ttemp)
                    rampl_list.append(0.0)
                    rprop_list.append(otemp)
                    ttemp = t_n
                    otemp = o_n
                    atemp = a_n if is_simulated else a_n * t_n
                    isopen = True
                else:
                    # Previous was open
                    if is_simulated:
                        # For simulated records, always extend the open period
                        ttemp += t_n
                        if o_n >= BAD_FLAG:
                            otemp = o_n
                    elif abs((atemp / ttemp) - a_n) <= AMPLITUDE_TOLERANCE:
                        # Same amplitude sublevel: extend
                        ttemp += t_n
                        atemp += a_n * t_n
                        if o_n >= BAD_FLAG:
                            otemp = o_n
                    else:
                        # Different amplitude sublevel: save and restart
                        rtint_list.append(ttemp)
                        rampl_list.append(atemp / ttemp)
                        _append_open_flag(rtint_list, rprop_list, otemp, badopen)
                        ttemp = t_n
                        otemp = o_n
                        atemp = a_n * t_n
        n += 1

    # ------------------------------------------------------------------ #
    # Append the last accumulated period                                   #
    # ------------------------------------------------------------------ #
    if isopen:
        rtint_list.append(-1.0)  # sentinel: truncated/incomplete open
        if is_simulated:
            rampl_list.append(atemp)
        else:
            rampl_list.append(atemp / ttemp)
    else:
        rtint_list.append(ttemp)
        rampl_list.append(0.0)
    rprop_list.append(BAD_FLAG)

    rtint = np.array(rtint_list, dtype=float)
    rampl = np.array(rampl_list, dtype=float)
    rprop = np.array(rprop_list, dtype=int)

    # ------------------------------------------------------------------ #
    # Trim leading shuts and trailing bad/shut periods                     #
    # ------------------------------------------------------------------ #
    rtint, rampl, rprop = _trim_leading_shuts(rtint, rampl, rprop)
    rtint, rampl, rprop = _trim_trailing_bad_or_shut(rtint, rampl, rprop)

    if len(rtint) == 0:
        logger.warning("impose_resolution produced an empty record")

    return rtint, rampl, rprop


# --------------------------------------------------------------------------- #
# Private helpers                                                               #
# --------------------------------------------------------------------------- #

def _append_open_flag(
    rtint_list: list[float],
    rprop_list: list[int],
    otemp: int,
    badopen: float,
) -> None:
    """Append the property flag for the open period just saved."""
    if badopen > 0 and rtint_list[-1] > badopen:
        rprop_list.append(BAD_FLAG)
    else:
        rprop_list.append(otemp)


def _trim_leading_shuts(
    rtint: np.ndarray, rampl: np.ndarray, rprop: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove leading shut periods so the record starts with an open."""
    while len(rampl) > 0 and rampl[0] == 0.0:
        rtint = rtint[1:]
        rampl = rampl[1:]
        rprop = rprop[1:]
    return rtint, rampl, rprop


def _trim_trailing_bad_or_shut(
    rtint: np.ndarray, rampl: np.ndarray, rprop: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove trailing intervals that are bad (rtint < 0) or shut (rampl == 0)."""
    while len(rtint) > 0 and (rtint[-1] < 0.0 or rampl[-1] == 0.0):
        rtint = rtint[:-1]
        rampl = rampl[:-1]
        rprop = rprop[:-1]
    return rtint, rampl, rprop

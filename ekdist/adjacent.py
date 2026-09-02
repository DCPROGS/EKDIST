"""Statistics of open times and the shut times next to them.

Correlations between an opening and its neighbouring shuttings are a property
of the mechanism, not of the fitting, and they are what the conditional
displays of a single-channel record are for. This module measures them **from a
record**; the theoretical counterparts live in :mod:`scalcs.sccurves` and
:mod:`scalcs.scalcslib`, and the point of having both is to superimpose one on
the other.

Two measurements, matching Figs 6D and 6E of Colquhoun, Hatton & Hawkes (2003):

:func:`conditional_mean_open_time`
    the mean apparent open time for openings whose adjacent shut time falls in
    each of a list of ranges -- the diamonds of Fig. 6D.
:func:`dependency`
    the observed dependency of Magleby & Song (1992) on a log-log grid --
    Fig. 6E.

**Which neighbour.** An opening has a shut time on each side, and the theory
distinguishes them: ``scalcs`` returns the mean open time given the *previous*
gap and given the *next* gap as two different curves, and its ``HJC_dependency``
is the joint density of an open time and the shut time that **follows** it. So
every function here takes ``side``:

``'next'``
    the shut time that follows the opening. Compare with ``HJC_dependency``,
    and with the second return value of
    ``HJC_adjacent_mean_open_to_shut_time_pdf``.
``'previous'``
    the shut time before the opening. Compare with the first return value.
``'adjacent'``
    both, pooled, so an opening with a shut time on each side is counted twice.
    This is what a legend means by "adjacent".

Getting this wrong does not raise anything: it quietly compares a measurement
with the wrong curve.
"""

from __future__ import annotations

import numpy as np

SIDES = ("next", "previous", "adjacent")


def pairs(tints, ampls, side="next"):
    """Pair each opening with a neighbouring shut time.

    Parameters
    ----------
    tints, ampls : array_like
        An alternating record: interval durations, and amplitudes that are zero
        for a shut interval and non-zero for an open one. This is what
        ``scalcs.scsim.impose_resolution`` returns.
    side : {'next', 'previous', 'adjacent'}

    Returns
    -------
    topen, tshut : ndarray
        Equal length. Openings at either end of the record with no neighbour on
        the required side are dropped.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, not {side!r}")
    tints = np.asarray(tints, float)
    ampls = np.asarray(ampls, float)
    if tints.shape != ampls.shape:
        raise ValueError("tints and ampls must have the same length")

    is_open = ampls != 0.0
    topen, tshut = [], []
    if side in ("next", "adjacent"):
        take = is_open[:-1] & ~is_open[1:]
        topen.append(tints[:-1][take])
        tshut.append(tints[1:][take])
    if side in ("previous", "adjacent"):
        take = is_open[1:] & ~is_open[:-1]
        topen.append(tints[1:][take])
        tshut.append(tints[:-1][take])
    return np.concatenate(topen), np.concatenate(tshut)


def conditional_mean_open_time(topen, tshut, ranges):
    """Mean apparent open time for each range of adjacent shut time.

    The diamonds and their bars in Fig. 6D. Ranges must be given, not chosen
    here, because they have to be wide enough to hold enough observations and
    that is a judgement about the data.

    Parameters
    ----------
    topen, tshut : array_like
        Paired, as returned by :func:`pairs`.
    ranges : sequence of (float, float)
        Half-open shut-time ranges ``[lo, hi)``; ``hi`` may be ``inf``.

    Returns
    -------
    dict of ndarray
        ``n`` (openings in the range), ``shut_mean`` (mean of the shut times
        that defined it, which is where the paper plots the point),
        ``open_mean``, ``open_sd``, ``open_sem``. Empty ranges give ``nan``
        rather than being dropped, so the result lines up with ``ranges``.
    """
    topen = np.asarray(topen, float)
    tshut = np.asarray(tshut, float)
    n = np.zeros(len(ranges), int)
    shut_mean = np.full(len(ranges), np.nan)
    open_mean = np.full(len(ranges), np.nan)
    open_sd = np.full(len(ranges), np.nan)

    for i, (lo, hi) in enumerate(ranges):
        take = (tshut >= lo) & (tshut < hi)
        n[i] = int(take.sum())
        if n[i] == 0:
            continue
        shut_mean[i] = tshut[take].mean()
        open_mean[i] = topen[take].mean()
        if n[i] > 1:
            open_sd[i] = topen[take].std(ddof=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        open_sem = open_sd / np.sqrt(n)
    return dict(n=n, shut_mean=shut_mean, open_mean=open_mean,
                open_sd=open_sd, open_sem=open_sem)


def log_edges(values, tres=None, per_decade=8):
    """Log-spaced bin edges spanning ``values``, for :func:`dependency`."""
    values = np.asarray(values, float)
    lo = tres if tres else values[values > 0].min()
    hi = values.max()
    n = max(int(np.ceil(np.log10(hi / lo) * per_decade)), 2)
    return np.logspace(np.log10(lo), np.log10(hi), n + 1)


def dependency(topen, tshut, open_edges, shut_edges, min_count=10):
    """The observed dependency, on a grid of open and shut time.

    Dependency is defined (eqn (8) of the paper, after Magleby & Song 1992) as

        d(to, ts) = f(to, ts) / (fo(to) fs(ts)) - 1,

    zero where the two are independent, positive where openings of that length
    occur next to shuttings of that length more often than chance. Binned, the
    ratio of densities is the ratio of the joint probability of a cell to the
    product of its marginals, so with ``N`` observations in total,

        d = N_ij N / (N_i N_j) - 1.

    Parameters
    ----------
    topen, tshut : array_like
        Paired, as returned by :func:`pairs`.
    open_edges, shut_edges : array_like
        Bin edges; use :func:`log_edges`, since dwell times span decades.
    min_count : int
        Cells holding fewer than this are returned as ``nan``. A dependency
        estimated from three intervals is noise, and the published figures
        leave such regions blank.

    Returns
    -------
    dict
        ``dependency`` (2-D, open by shut, ``nan`` where sparse), ``counts``
        (the joint histogram), ``open_centres`` and ``shut_centres``
        (geometric), ``n`` (pairs in all, which is what the marginals are taken
        over) and ``inside`` (pairs that fell within the grid).
    """
    topen = np.asarray(topen, float)
    tshut = np.asarray(tshut, float)
    open_edges = np.asarray(open_edges, float)
    shut_edges = np.asarray(shut_edges, float)

    counts, _, _ = np.histogram2d(topen, tshut, bins=[open_edges, shut_edges])
    if counts.sum() == 0:
        raise ValueError("no pairs fell inside the grid")

    # The marginals must come from EVERY pair, not only from those inside the
    # grid, because the definition is a ratio of the joint density to the
    # product of the unconditional marginals. Taking the row and column sums of
    # the joint histogram instead silently renormalises to whatever rectangle
    # the grid happens to cover, which is wrong whenever the grid does not span
    # the whole distribution -- and a shut-time grid never does, since the long
    # shut times between activations run to seconds.
    total = len(topen)
    row = np.histogram(topen, bins=open_edges)[0][:, None].astype(float)
    col = np.histogram(tshut, bins=shut_edges)[0][None, :].astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        dep = counts * total / (row * col) - 1.0
    dep[counts < min_count] = np.nan

    centres = lambda e: np.sqrt(e[:-1] * e[1:])       # noqa: E731 (geometric)
    return dict(dependency=dep, counts=counts, n=total,
                inside=int(counts.sum()),
                open_centres=centres(open_edges),
                shut_centres=centres(shut_edges))

"""Global open probability (Popen) with bootstrap confidence intervals.

Two entry points:

* :func:`global_popen` — point estimate and bootstrap CI from a
  :class:`~ekdist.record.SingleChannelRecord`'s resolved period durations.
* :func:`burst_popen` — point estimate and bootstrap CI from a
  :class:`~ekdist.bursts.Bursts` object, resampling at the burst level.

Both return a :class:`PopenResult` dataclass.

Usage::

    from ekdist.popen import global_popen, burst_popen

    result = global_popen(rec, n_bootstrap=1000)
    print(result)

    bursts = Bursts.from_periods(rec.periods, tcrit=5e-3)
    result = burst_popen(bursts, n_bootstrap=1000)
    print(result)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PopenResult:
    """Result of a global Popen calculation with bootstrap confidence interval.

    Attributes
    ----------
    popen:
        Point estimate of the open probability.
    ci_lower:
        Lower bound of the bootstrap confidence interval.
    ci_upper:
        Upper bound of the bootstrap confidence interval.
    n_bootstrap:
        Number of bootstrap resamples used.
    confidence:
        Confidence level (e.g. 0.95 for 95% CI).
    total_open_time:
        Total open time used in the point estimate (seconds).
    total_time:
        Total time (open + shut) used in the point estimate (seconds).
    """

    popen: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    confidence: float
    total_open_time: float
    total_time: float

    def __repr__(self) -> str:
        pct = int(round(self.confidence * 100))
        return (
            f"Popen = {self.popen:.6f}  "
            f"[{pct}% CI: {self.ci_lower:.6f} – {self.ci_upper:.6f}]  "
            f"(n_bootstrap={self.n_bootstrap})"
        )


def global_popen(
    rec,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> PopenResult:
    """Compute global Popen from a record's resolved periods with bootstrap CI.

    The point estimate is::

        Popen = sum(open_period_durations) / (sum(open) + sum(shut))

    Bootstrap confidence intervals are computed by resampling open and shut
    period duration arrays independently (with replacement, same size as the
    original arrays) and recomputing Popen for each resample.

    Parameters
    ----------
    rec:
        A :class:`~ekdist.record.SingleChannelRecord` with ``tres`` set.
    n_bootstrap:
        Number of bootstrap resamples.  Must be >= 1.
    confidence:
        Confidence level for the CI (0 < confidence < 1).
    rng:
        NumPy random generator.  Pass a seeded generator for reproducibility.

    Returns
    -------
    PopenResult

    Raises
    ------
    ValueError
        If ``n_bootstrap < 1``, ``confidence`` is outside (0, 1), or the
        record has no shut intervals.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")

    opens = np.asarray(rec.periods.open_intervals, dtype=float)
    shuts = np.asarray(rec.periods.shut_intervals, dtype=float)

    if len(shuts) == 0:
        raise ValueError(
            "Record has no shut intervals; cannot compute global Popen."
        )

    total_open = float(np.sum(opens))
    total_shut = float(np.sum(shuts))
    total_time = total_open + total_shut
    popen = total_open / total_time

    if rng is None:
        rng = np.random.default_rng()

    # Bootstrap: resample each array independently, same size
    boot_popens = np.empty(n_bootstrap)
    n_open = len(opens)
    n_shut = len(shuts)
    for i in range(n_bootstrap):
        bo = np.sum(opens[rng.integers(0, n_open, size=n_open)])
        bs = np.sum(shuts[rng.integers(0, n_shut, size=n_shut)])
        boot_popens[i] = bo / (bo + bs)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_popens, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(boot_popens, 100.0 * (1.0 - alpha / 2.0)))

    return PopenResult(
        popen=popen,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        total_open_time=total_open,
        total_time=total_time,
    )


def burst_popen(
    bursts,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> PopenResult:
    """Compute mean burst Popen with bootstrap CI by resampling bursts.

    The point estimate is :meth:`~ekdist.bursts.Bursts.mean_popen` (Popen
    excluding the last opening, averaged over all bursts).  Bootstrap CIs are
    obtained by resampling the per-burst Popen values with replacement.

    Parameters
    ----------
    bursts:
        A :class:`~ekdist.bursts.Bursts` instance.
    n_bootstrap:
        Number of bootstrap resamples.  Must be >= 1.
    confidence:
        Confidence level for the CI (0 < confidence < 1).
    rng:
        NumPy random generator.  Pass a seeded generator for reproducibility.

    Returns
    -------
    PopenResult

    Raises
    ------
    ValueError
        If fewer than 2 bursts are available (CI is undefined), or if
        ``n_bootstrap < 1`` or ``confidence`` is outside (0, 1).
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if len(bursts) < 2:
        raise ValueError(
            f"burst_popen requires at least 2 bursts; got {len(bursts)}"
        )

    per_burst = np.array(bursts.list_popen(), dtype=float)
    # Exclude NaN bursts (degenerate single-interval bursts)
    per_burst = per_burst[~np.isnan(per_burst)]
    popen = float(np.mean(per_burst))

    if rng is None:
        rng = np.random.default_rng()

    n = len(per_burst)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = per_burst[rng.integers(0, n, size=n)]
        boot_means[i] = float(np.mean(sample))

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    # total_open_time and total_time are not meaningful for burst-level Popen
    return PopenResult(
        popen=popen,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        total_open_time=float("nan"),
        total_time=float("nan"),
    )

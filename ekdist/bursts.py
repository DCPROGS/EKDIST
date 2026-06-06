"""Burst analysis for single-channel records.

A burst is a group of openings separated by short shut times, flanked by
long (inter-burst) shut times.  Burst boundaries are defined by a critical
shut time *tcrit*: any shut period longer than *tcrit* ends the current burst
and starts a new one.

Usage::

    bursts = Bursts.from_periods(rec.periods, tcrit=5e-3)
    print(bursts)
    print("Mean Popen:", bursts.mean_popen())
    print("Number of openings per burst:", bursts.list_n_openings())
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Bursts:
    """Collection of bursts extracted from a :class:`~ekdist.record.Periods` object.

    Each burst is stored as a 1-D numpy array of alternating open/shut interval
    durations starting with an open interval (same convention as Periods)::

        burst[0::2]  → open intervals within the burst
        burst[1::2]  → shut intervals within the burst

    Parameters
    ----------
    burst_list:
        List of 1-D numpy arrays, one per burst.
    tcrit:
        Critical shut time used to define burst boundaries (seconds).
    """

    def __init__(self, burst_list: list[np.ndarray], tcrit: float) -> None:
        self.bursts: list[np.ndarray] = burst_list
        self.tcrit: float = float(tcrit)

    # ------------------------------------------------------------------ #
    # Constructor                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_periods(cls, periods: object, tcrit: float) -> "Bursts":
        """Slice a :class:`~ekdist.record.Periods` object into bursts.

        Definition:
        (1) Does not require a gap > tcrit before the first burst;
        (2) An unusable shut time is a valid end-of-burst marker;
        (3) Burst Popen is calculated without the last opening.

        Parameters
        ----------
        periods:
            A :class:`~ekdist.record.Periods` instance.
        tcrit:
            Critical shut time in seconds.  Shut intervals longer than this
            define burst boundaries.
        """
        t = np.asarray(periods.intervals, dtype=float)
        a = np.asarray(periods.amplitudes, dtype=float)

        long_shuts = np.where((t > tcrit) & (a == 0.0))[0]
        groups = np.split(t, long_shuts)
        burst_list = [groups[0]] + [np.delete(g, 0) for g in groups[1:]]
        # Remove empty bursts that can arise from consecutive long shuts
        burst_list = [b for b in burst_list if len(b) > 0]
        return cls(burst_list, tcrit)

    # ------------------------------------------------------------------ #
    # Per-burst metrics                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _open(burst: np.ndarray) -> np.ndarray:
        return burst[0::2]

    @staticmethod
    def _shut(burst: np.ndarray) -> np.ndarray:
        return burst[1::2]

    def total_open_time(self, burst: np.ndarray) -> float:
        """Total open time within *burst* (seconds)."""
        return float(np.sum(self._open(burst)))

    def total_shut_time(self, burst: np.ndarray) -> float:
        """Total shut time within *burst* (seconds, excludes flanking shuts)."""
        return float(np.sum(self._shut(burst)))

    def total_duration(self, burst: np.ndarray) -> float:
        """Total burst duration (open + shut) in seconds."""
        return float(np.sum(burst))

    def n_openings(self, burst: np.ndarray) -> int:
        """Number of openings in *burst*."""
        return len(self._open(burst))

    def popen(self, burst: np.ndarray) -> float:
        """Open probability including the last opening.

        Popen = total_open_time / total_duration.
        """
        return self.total_open_time(burst) / self.total_duration(burst)

    def popen_excl_last(self, burst: np.ndarray) -> float:
        """Open probability excluding the last opening.

        The last opening is removed and Popen is calculated from the remaining
        open time divided by the total time up to (but not including) the last
        opening.  For single-opening bursts returns 1.0 (degenerate case).
        """
        opens = self._open(burst)
        if len(opens) <= 1:
            return 1.0
        open_time_excl = float(np.sum(opens[:-1]))
        shut_time_total = self.total_shut_time(burst)
        denom = open_time_excl + shut_time_total
        return open_time_excl / denom if denom > 0 else 0.0

    def mean_open_time(self, burst: np.ndarray) -> float:
        """Mean open interval duration within *burst* (seconds)."""
        opens = self._open(burst)
        return float(np.mean(opens)) if len(opens) else 0.0

    def mean_shut_time(self, burst: np.ndarray) -> float:
        """Mean within-burst shut interval duration (seconds)."""
        shuts = self._shut(burst)
        return float(np.mean(shuts)) if len(shuts) else 0.0

    def longest_opening(self, burst: np.ndarray) -> float:
        """Duration of the longest single opening in *burst* (seconds)."""
        opens = self._open(burst)
        return float(np.max(opens)) if len(opens) else 0.0

    def longest_shut(self, burst: np.ndarray) -> float:
        """Duration of the longest within-burst shut interval (seconds)."""
        shuts = self._shut(burst)
        return float(np.max(shuts)) if len(shuts) else 0.0

    # ------------------------------------------------------------------ #
    # List methods (apply per-burst metric to all bursts)                  #
    # ------------------------------------------------------------------ #

    def list_popen(self) -> list[float]:
        """Popen (excluding last opening) for each burst."""
        return [self.popen_excl_last(b) for b in self.bursts]

    def list_popen_incl_last(self) -> list[float]:
        """Popen (including last opening) for each burst."""
        return [self.popen(b) for b in self.bursts]

    def list_duration(self) -> list[float]:
        """Total duration (seconds) for each burst."""
        return [self.total_duration(b) for b in self.bursts]

    def list_n_openings(self) -> list[int]:
        """Number of openings in each burst."""
        return [self.n_openings(b) for b in self.bursts]

    def list_total_open_time(self) -> list[float]:
        """Total open time (seconds) for each burst."""
        return [self.total_open_time(b) for b in self.bursts]

    def list_total_shut_time(self) -> list[float]:
        """Total within-burst shut time (seconds) for each burst (bursts with > 1 interval)."""
        return [self.total_shut_time(b) for b in self.bursts if len(b) > 1]

    def list_mean_open_time(self) -> list[float]:
        """Mean single-opening duration (seconds) for each burst."""
        return [self.mean_open_time(b) for b in self.bursts]

    def list_mean_shut_time(self) -> list[float]:
        """Mean within-burst shut duration (seconds) for bursts with > 1 interval."""
        return [self.mean_shut_time(b) for b in self.bursts if len(b) > 1]

    def list_longest_opening(self) -> list[float]:
        """Longest single opening (seconds) for each burst."""
        return [self.longest_opening(b) for b in self.bursts]

    def list_longest_shut(self) -> list[float]:
        """Longest within-burst shut (seconds) for bursts with > 1 interval."""
        return [self.longest_shut(b) for b in self.bursts if len(b) > 1]

    def bursts_with_n_openings(self, n: int) -> list[np.ndarray]:
        """Return all bursts that contain exactly *n* openings."""
        return [b for b in self.bursts if self.n_openings(b) == n]

    # ------------------------------------------------------------------ #
    # Aggregate statistics                                                  #
    # ------------------------------------------------------------------ #

    def mean_popen(self) -> float:
        """Grand mean Popen (excl. last opening), excluding NaN bursts."""
        popens = np.array(self.list_popen())
        valid = popens[~np.isnan(popens)]
        return float(np.mean(valid)) if len(valid) else float("nan")

    def mean_n_openings(self) -> float:
        """Mean number of openings per burst."""
        return float(np.mean(self.list_n_openings()))

    def mean_duration(self) -> float:
        """Mean burst duration (seconds)."""
        return float(np.mean(self.list_duration()))

    def mean_total_open_time(self) -> float:
        """Mean total open time per burst (seconds)."""
        return float(np.mean(self.list_total_open_time()))

    def mean_total_shut_time(self) -> float:
        """Mean total within-burst shut time per burst (seconds)."""
        vals = self.list_total_shut_time()
        return float(np.mean(vals)) if vals else float("nan")

    def mean_mean_open_time(self) -> float:
        """Mean of per-burst mean opening durations (seconds)."""
        return float(np.mean(self.list_mean_open_time()))

    # ------------------------------------------------------------------ #
    # Filtering                                                             #
    # ------------------------------------------------------------------ #

    def filter_min_openings(self, n: int) -> "Bursts":
        """Return a new :class:`Bursts` keeping only bursts with >= *n* openings."""
        kept = [b for b in self.bursts if self.n_openings(b) >= n]
        return Bursts(kept, self.tcrit)

    def filter_max_open_time(self, max_open: float) -> "Bursts":
        """Return a new :class:`Bursts` excluding bursts that contain an opening
        longer than *max_open* seconds.

        This replaces the buggy ``remove_bursts_with_long_open_times`` from
        the original code.
        """
        kept = [b for b in self.bursts if self.longest_opening(b) <= max_open]
        return Bursts(kept, self.tcrit)

    def filter_min_duration(self, min_dur: float) -> "Bursts":
        """Return a new :class:`Bursts` keeping bursts with total duration >= *min_dur*."""
        kept = [b for b in self.bursts if self.total_duration(b) >= min_dur]
        return Bursts(kept, self.tcrit)

    # ------------------------------------------------------------------ #
    # Representation                                                        #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.bursts)

    def __repr__(self) -> str:
        return (
            f"Bursts(tcrit={self.tcrit*1e3:.3f} ms, n={len(self.bursts)},\n"
            f"  mean_duration   = {self.mean_duration()*1e3:.4f} ms\n"
            f"  mean_Popen      = {self.mean_popen():.4f}\n"
            f"  mean_n_openings = {self.mean_n_openings():.2f})"
        )

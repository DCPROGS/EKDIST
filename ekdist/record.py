"""Single-channel record data model.

Provides ``SingleChannelRecord`` (loading + resolution) and ``Periods``
(grouping resolved intervals into open/shut periods).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from ekdist import io as _io
from ekdist.resolution import impose_resolution

logger = logging.getLogger(__name__)


class SingleChannelRecord:
    """Wrapper over a list of idealised single-channel intervals.

    Loading
    -------
    Use the class-method constructors::

        rec = SingleChannelRecord.from_scn("path/to/file.scn")
        rec = SingleChannelRecord.from_intervals(itint, iampl, iprop)

    Resolution
    ----------
    Set ``rec.tres`` (in seconds) to impose a temporal resolution.  This
    filters out unresolvable short intervals by concatenating them with their
    neighbours::

        rec.tres = 30e-6  # 30 µs resolution

    Accessing data
    --------------
    * ``rec.itint``, ``rec.iampl``, ``rec.iprop`` — raw intervals.
    * ``rec.rtint``, ``rec.rampl``, ``rec.rprop`` — resolved intervals.
    * ``rec.periods`` — :class:`Periods` object grouping resolved intervals.
    """

    def __init__(self) -> None:
        self._is_loaded: bool = False
        self._origin: str = ""
        self._is_simulated: bool = False
        self._header = None  # SCNHeader or None
        self._tres: float = 0.0
        self._badopen: float = -1.0

        # Raw intervals (set by loaders)
        self.itint: np.ndarray = np.array([])
        self.iampl: np.ndarray = np.array([])
        self.iprop: np.ndarray = np.array([], dtype=int)

        # Resolved intervals (set by _apply_resolution)
        self.rtint: np.ndarray = np.array([])
        self.rampl: np.ndarray = np.array([])
        self.rprop: np.ndarray = np.array([], dtype=int)

        self._periods: Periods | None = None

    # ------------------------------------------------------------------ #
    # Constructors                                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_scn(cls, path: str | Path, *, verbose: bool = False) -> "SingleChannelRecord":
        """Load from a SCAN binary (.SCN) file.

        Parameters
        ----------
        path:
            Path to the SCN file.
        verbose:
            Forward to the SCN header reader.
        """
        rec = cls()
        scn = _io.read_scn(path, verbose=verbose)
        rec.itint = scn.intervals
        rec.iampl = scn.amplitudes
        rec.iprop = scn.flags.astype(int)
        rec._header = scn.header
        rec._is_simulated = scn.header.record_type == "simulated"
        rec._origin = f"SCN file: {Path(path).name}"
        rec._is_loaded = True
        rec._apply_resolution()
        return rec

    @classmethod
    def from_scn_files(
        cls,
        paths: list,
        *,
        verbose: bool = False,
    ) -> "SingleChannelRecord":
        """Load and concatenate multiple SCN files into one record.

        Interval, amplitude, and flag arrays from each file are concatenated
        in order.  The header from the first file is used.

        Parameters
        ----------
        paths:
            List of paths to SCN files.  Must contain at least one entry.
        verbose:
            Forward to the SCN reader.

        Raises
        ------
        ValueError
            If *paths* is empty.
        FileNotFoundError
            If any path does not exist.
        """
        if not paths:
            raise ValueError("from_scn_files requires at least one path")

        path_objs = [Path(p) for p in paths]
        scns = [_io.read_scn(p, verbose=verbose) for p in path_objs]

        record_types = {s.header.record_type for s in scns}
        if len(record_types) > 1:
            logger.warning(
                "SCN files have mixed record types %s; using first file's header",
                record_types,
            )

        rec = cls()
        rec.itint = np.concatenate([s.intervals for s in scns])
        rec.iampl = np.concatenate([s.amplitudes for s in scns])
        rec.iprop = np.concatenate([s.flags.astype(int) for s in scns])
        rec._header = scns[0].header
        rec._is_simulated = all(s.header.record_type == "simulated" for s in scns)
        rec._origin = "SCN files: " + ", ".join(p.name for p in path_objs)
        rec._is_loaded = True
        rec._apply_resolution()
        return rec

    @classmethod
    def from_intervals(
        cls,
        itint: np.ndarray,
        iampl: np.ndarray,
        iprop: np.ndarray | None = None,
        *,
        is_simulated: bool = False,
    ) -> "SingleChannelRecord":
        """Load from pre-built interval arrays.

        Parameters
        ----------
        itint:
            Interval durations (seconds).
        iampl:
            Amplitudes (pA; 0 = shut).
        iprop:
            Property flags.  Defaults to all zeros (all usable).
        is_simulated:
            Set True for simulated records.
        """
        rec = cls()
        rec.itint = np.asarray(itint, dtype=float)
        rec.iampl = np.asarray(iampl, dtype=float)
        rec.iprop = (
            np.zeros(len(itint), dtype=int)
            if iprop is None
            else np.asarray(iprop, dtype=int)
        )
        rec._is_simulated = is_simulated
        rec._origin = "intervals loaded from array"
        rec._is_loaded = True
        rec._apply_resolution()
        return rec

    # ------------------------------------------------------------------ #
    # Resolution property                                                   #
    # ------------------------------------------------------------------ #

    @property
    def tres(self) -> float:
        """Temporal resolution in seconds."""
        return self._tres

    @tres.setter
    def tres(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"tres must be non-negative; got {value}")
        self._tres = float(value)
        self._apply_resolution()

    @property
    def badopen(self) -> float:
        """Open intervals longer than this (seconds) are flagged bad.  -1 = disabled."""
        return self._badopen

    @badopen.setter
    def badopen(self, value: float) -> None:
        self._badopen = float(value)
        self._apply_resolution()

    def _apply_resolution(self) -> None:
        if not self._is_loaded or len(self.itint) == 0:
            return
        try:
            self.rtint, self.rampl, self.rprop = impose_resolution(
                self.itint,
                self.iampl,
                self.iprop,
                self._tres,
                is_simulated=self._is_simulated,
                badopen=self._badopen,
            )
        except ValueError as exc:
            logger.warning("Resolution imposition failed: %s", exc)
            self.rtint = self.itint.copy()
            self.rampl = self.iampl.copy()
            self.rprop = self.iprop.copy()
        self._periods = Periods(self.rtint, self.rampl, self.rprop)

    # ------------------------------------------------------------------ #
    # Periods property                                                      #
    # ------------------------------------------------------------------ #

    @property
    def periods(self) -> "Periods":
        """Grouped open/shut periods after resolution imposition."""
        if self._periods is None:
            self._apply_resolution()
        return self._periods  # type: ignore[return-value]

    @property
    def header(self):
        """SCN file header (SCNHeader dataclass, or None if loaded from arrays)."""
        return self._header

    # ------------------------------------------------------------------ #
    # Representation                                                        #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        if not self._is_loaded:
            return "SingleChannelRecord (empty)"
        opens = self.periods.open_intervals
        shuts = self.periods.shut_intervals
        lines = [
            self._origin,
            f"  Raw intervals   : {len(self.itint)}",
            f"  Resolution (µs) : {self._tres * 1e6:.1f}",
            f"  Resolved        : {len(self.rtint)}",
            f"  Periods         : {len(self.periods.intervals)}",
            "",
            f"  Open periods  n={len(opens)}",
        ]
        if len(opens):
            lines += [
                f"    mean ± SD = {np.mean(opens)*1e3:.4f} ± {np.std(opens)*1e3:.4f} ms",
                f"    range     = {np.min(opens)*1e3:.4f} – {np.max(opens)*1e3:.4f} ms",
            ]
        lines += [f"  Shut periods  n={len(shuts)}"]
        if len(shuts):
            lines += [
                f"    mean ± SD = {np.mean(shuts)*1e3:.4f} ± {np.std(shuts)*1e3:.4f} ms",
                f"    range     = {np.min(shuts)*1e3:.4f} – {np.max(shuts)*1e3:.4f} ms",
            ]
        return "\n".join(lines)


class Periods:
    """Open and shut periods extracted from resolved single-channel intervals.

    After resolution imposition, consecutive resolved intervals of the same
    type (both open or both shut) are merged into a single period.  This handles
    sub-conductance levels: two successive openings at slightly different
    amplitudes are kept separate.

    The merged sequence always alternates open/shut starting with open, so::

        periods.intervals[0::2]  → open periods
        periods.intervals[1::2]  → shut periods
    """

    def __init__(
        self,
        rtint: np.ndarray,
        rampl: np.ndarray,
        rprop: np.ndarray | None = None,
    ) -> None:
        rprops = (
            np.zeros(len(rtint), dtype=int) if rprop is None else np.asarray(rprop, dtype=int)
        )
        self.intervals: list[float] = []
        self.amplitudes: list[float] = []
        self.flags: list[int] = []
        if len(rtint) > 0:
            self._build(rtint, rampl, rprops)

    def _build(
        self, rtint: np.ndarray, rampl: np.ndarray, rprop: np.ndarray
    ) -> None:
        from ekdist._constants import BAD_FLAG

        oint = float(rtint[0])
        oamp = float(rampl[0]) * float(rtint[0])  # time-weighted sum
        oopt = int(rprop[0])

        for t, a, o in zip(rtint[1:], rampl[1:], rprop[1:]):
            t, a, o = float(t), float(a), int(o)
            if o >= BAD_FLAG:
                oopt = BAD_FLAG
            both_open = math.fabs(a) > 0.0 and math.fabs(oamp) > 0.0
            both_shut = a == 0.0 and oamp == 0.0
            if both_open or both_shut:
                oint += t
                oamp += a * t
            else:
                self._append(oint, oamp, oopt)
                oint, oamp, oopt = t, a * t, o

        self._append(oint, oamp, oopt)

    def _append(self, oint: float, oamp: float, oopt: int) -> None:
        try:
            self.amplitudes.append(oamp / oint)
        except ZeroDivisionError:
            self.amplitudes.append(oamp)
        self.intervals.append(oint)
        self.flags.append(oopt)

    @property
    def open_intervals(self) -> np.ndarray:
        """Open period durations (seconds)."""
        return np.asarray(self.intervals[0::2])

    @property
    def shut_intervals(self) -> np.ndarray:
        """Shut period durations (seconds)."""
        return np.asarray(self.intervals[1::2])

    def open_intervals_in_range(self, low: float, high: float) -> np.ndarray:
        """Open periods with duration in [low, high] seconds."""
        ops = self.open_intervals
        return ops[(ops >= low) & (ops <= high)]

    def shut_intervals_in_range(self, low: float, high: float) -> np.ndarray:
        """Shut periods with duration in [low, high] seconds."""
        sh = self.shut_intervals
        return sh[(sh >= low) & (sh <= high)]

"""Critical time calculations for burst analysis.

The critical time *tcrit* between two exponential components of a shut-time
distribution is the shut interval duration that minimises misclassification
when dividing a record into bursts.

Three criteria are implemented (all from Colquhoun & Hawkes):

* **DC** (Colquhoun & Hawkes): equal *fraction* misclassified from each side.
* **C&N** (Clapham & Neher): equal *number* misclassified from each side.
* **Jackson**: minimise total number of misclassified events (rate-based criterion).

Usage::

    tc = Tcrit(tau, area)
    print(tc.summary())
    dc_tcrits = tc.tcrits['DC']
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import bisect

logger = logging.getLogger(__name__)


class Tcrit:
    """Critical time between exponential components using three criteria.

    Parameters
    ----------
    tau:
        Time constants of fitted exponential components (seconds).
    area:
        Component areas (relative weights, must sum to 1).
    """

    def __init__(self, tau: np.ndarray, area: np.ndarray) -> None:
        self.tau = np.asarray(tau, dtype=float)
        self.area = np.asarray(area, dtype=float)
        if len(self.tau) < 2:
            raise ValueError("Tcrit requires at least 2 exponential components")

        self.tcrits: dict[str, list[float | None]] = {}
        self._compute()

    def _compute(self) -> None:
        self.tcrits["DC"] = self._calculate_all(_tcrit_objective_DC)
        self.tcrits["C&N"] = self._calculate_all(_tcrit_objective_CN)
        self.tcrits["Jackson"] = self._calculate_all(_tcrit_objective_Jackson)

    def _calculate_all(
        self, objective
    ) -> list[float | None]:
        results: list[float | None] = []
        for i in range(len(self.tau) - 1):
            try:
                tcrit = bisect(
                    objective,
                    self.tau[i],
                    self.tau[i + 1],
                    args=(self.tau, self.area, i + 1),
                )
                results.append(float(tcrit))
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Bisection failed for components %d–%d: %s", i + 1, i + 2, exc
                )
                results.append(None)
        return results

    def misclassified(
        self, tcrit: float, comp: int
    ) -> tuple[float, float, float, float]:
        """Number and fraction misclassified at *tcrit* between components.

        Parameters
        ----------
        tcrit:
            Candidate critical time (seconds).
        comp:
            Component index: components 0…comp-1 are 'fast', comp…end are 'slow'.

        Returns
        -------
        enf, ens, pf, ps : float
            Number misclassified fast, number misclassified slow,
            fraction misclassified fast, fraction misclassified slow.
        """
        return _misclassified(tcrit, self.tau, self.area, comp)

    def summary(self) -> str:
        """Formatted table of tcrit values."""
        n = len(self.tau)
        lines = ["\nSUMMARY of tcrit values (ms)", "Components\t\tDC\t\tC&N\t\tJackson"]
        for i in range(n - 1):
            def _fmt(v):
                return f"{v*1000:.4g}" if v is not None else "failed"

            dc = _fmt(self.tcrits["DC"][i])
            cn = _fmt(self.tcrits["C&N"][i])
            ja = _fmt(self.tcrits["Jackson"][i])
            lines.append(f"{i+1} to {i+2}\t\t\t{dc}\t\t{cn}\t\t{ja}")
        return "\n".join(lines)

    def misclassified_summary(self, criterion: str = "DC") -> str:
        """Formatted misclassification stats for all component pairs."""
        lines = []
        for i, tcrit in enumerate(self.tcrits[criterion]):
            if tcrit is None:
                lines.append(f"Components {i+1}–{i+2}: bisection failed")
                continue
            enf, ens, pf, ps = self.misclassified(tcrit, i + 1)
            lines.append(
                f"Components {i+1}–{i+2}  tcrit = {tcrit*1e3:.4g} ms\n"
                f"  % misclassified: fast = {pf*100:.4g}  slow = {ps*100:.4g}\n"
                f"  # misclassified (per 100): fast = {enf*100:.4g}  slow = {ens*100:.4g}\n"
                f"  Total misclassified (per 100) = {(enf+ens)*100:.4g}"
            )
        return "\n".join(lines)


# =========================================================================== #
# Pure functions (usable without constructing Tcrit)                           #
# =========================================================================== #

def _misclassified(
    tcrit: float,
    tau: np.ndarray,
    area: np.ndarray,
    comp: int,
) -> tuple[float, float, float, float]:
    """Calculate misclassified events at *tcrit* between components [:comp] and [comp:]."""
    tfast, tslow = tau[:comp], tau[comp:]
    afast, aslow = area[:comp], area[comp:]

    enf = float(np.sum(afast * np.exp(-tcrit / tfast)))
    ens = float(np.sum(aslow * (1.0 - np.exp(-tcrit / tslow))))
    pf = enf / float(np.sum(afast))
    ps = ens / float(np.sum(aslow))
    return enf, ens, pf, ps


def _tcrit_objective_DC(
    tcrit: float, tau: np.ndarray, area: np.ndarray, comp: int
) -> float:
    """Objective: equal fraction misclassified (DC criterion). Root = tcrit."""
    _, _, pf, ps = _misclassified(tcrit, tau, area, comp)
    return ps - pf


def _tcrit_objective_CN(
    tcrit: float, tau: np.ndarray, area: np.ndarray, comp: int
) -> float:
    """Objective: equal number misclassified (Clapham-Neher criterion). Root = tcrit."""
    enf, ens, _, _ = _misclassified(tcrit, tau, area, comp)
    return ens - enf


def _tcrit_objective_Jackson(
    tcrit: float, tau: np.ndarray, area: np.ndarray, comp: int
) -> float:
    """Objective: minimum total misclassified (Jackson criterion). Root = tcrit."""
    tfast, tslow = tau[:comp], tau[comp:]
    afast, aslow = area[:comp], area[comp:]
    enf = float(np.sum((afast / tfast) * np.exp(-tcrit / tfast)))
    ens = float(np.sum((aslow / tslow) * np.exp(-tcrit / tslow)))
    return enf - ens

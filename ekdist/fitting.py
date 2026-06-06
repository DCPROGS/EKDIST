"""Probability distribution fitting for dwell-time and amplitude data.

Provides:

* :class:`ExponentialPDF` — multi-component exponential mixture for
  dwell-time distributions.
* :class:`GaussianMixturePDF` — multi-component Gaussian mixture for
  amplitude distributions.

Both classes expose a common interface::

    pdf.theta            # free parameter vector
    pdf.pdf(theta, X)    # evaluate density at points X
    pdf.LL(theta, X)     # negative log-likelihood
    pdf.fit(X)           # Nelder-Mead MLE fit in-place; returns OptimizeResult
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import norm as _norm

from ekdist._constants import LOG_LIKELIHOOD_MIN, LOG_LIKELIHOOD_PENALTY

logger = logging.getLogger(__name__)


# =========================================================================== #
# Exponential mixture                                                           #
# =========================================================================== #

class ExponentialPDF:
    """Multi-component exponential PDF fitted to dwell-time data.

    Parameters
    ----------
    tau:
        Initial time constants (seconds).  Length k gives a k-component mixture.
    area:
        Initial component areas (relative weights, sum to 1).  Several forms
        are accepted:

        * ``None`` → uniform (1/k each)
        * length k-1 → last area is derived as ``1 - sum(area)``
        * length k → last area is normalised so they sum to 1

    Examples
    --------
    >>> pdf = ExponentialPDF([0.001, 0.01], [0.5])
    >>> res = pdf.fit(data)
    >>> print(pdf.tau, pdf.area)
    """

    def __init__(
        self,
        tau: list[float] | np.ndarray,
        area: list[float] | np.ndarray | None = None,
    ) -> None:
        tau_arr = np.asarray(tau, dtype=float)
        if np.any(tau_arr <= 0):
            raise ValueError("All time constants tau must be positive")

        k = len(tau_arr)
        if area is None:
            area_arr = np.ones(k) / k
        else:
            area_arr = np.asarray(area, dtype=float)
            if len(area_arr) == k - 1:
                area_arr = np.append(area_arr, 1.0 - np.sum(area_arr))
            elif len(area_arr) == k:
                area_arr = area_arr.copy()
                area_arr[-1] = 1.0 - np.sum(area_arr[:-1])
            else:
                raise ValueError(
                    f"area must have length {k} or {k-1}; got {len(area_arr)}"
                )

        self.tau: np.ndarray = tau_arr
        self.area: np.ndarray = area_arr
        self.ncomps: int = k
        self.eqname: str = "exponential pdf"
        self._tres: float = 0.0  # lower integration bound; 0 = use min(X)

        # pars = [tau_0 … tau_{k-1}, area_0 … area_{k-1}]
        self.pars: np.ndarray = np.concatenate([self.tau, self.area])
        # Last area is always derived from the others
        self.fixed: list[bool] = [False] * (2 * k)
        self.fixed[-1] = True
        self.names: list[str] = ["tau"] * k + ["area"] * k

        self.tcrits: np.ndarray = np.empty((3, max(k - 1, 1)))

    # ------------------------------------------------------------------ #
    # theta property (free parameters only)                                #
    # ------------------------------------------------------------------ #

    def _get_theta(self) -> np.ndarray:
        return self.pars[np.where(~np.array(self.fixed))[0]]

    def _set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        # Re-insert fixed parameters
        for idx in np.nonzero(self.fixed)[0]:
            theta = np.insert(theta, idx, self.pars[idx])
        self.tau, self.area = np.split(theta, [self.ncomps])
        self.area[-1] = 1.0 - np.sum(self.area[:-1])
        self.pars = np.concatenate([self.tau, self.area])

    theta = property(_get_theta, _set_theta)

    # ------------------------------------------------------------------ #
    # Density and log-likelihood                                            #
    # ------------------------------------------------------------------ #

    def pdf(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate the PDF at points *X* for parameter vector *theta*.

        Fully vectorised.

        Parameters
        ----------
        theta:
            Free parameter vector (as returned by ``self.theta``).
        X:
            Data points (seconds).

        Returns
        -------
        np.ndarray
            PDF values at each point in X.
        """
        self._set_theta(theta)
        X = np.asarray(X, dtype=float)
        # shape: (ncomps, len(X)) → broadcast over components
        return np.sum(
            (self.area[:, None] / self.tau[:, None]) * np.exp(-X[None, :] / self.tau[:, None]),
            axis=0,
        )

    def LL(self, theta: np.ndarray, X: np.ndarray) -> float:
        """Negative log-likelihood (minimise this to fit).

        Uses the scaled likelihood to account for observations outside the
        observable range [min(X), max(X)].

        Parameters
        ----------
        theta:
            Free parameter vector.
        X:
            Dwell-time observations (seconds).

        Returns
        -------
        float
            Negative log-likelihood.
        """
        self._set_theta(theta)

        # Guard: optimizer may probe tau <= 0 or area < 0 during Nelder-Mead
        if np.any(self.tau <= 0.0) or np.any(self.area < 0.0):
            return LOG_LIKELIHOOD_PENALTY

        X = np.asarray(X, dtype=float)
        xmin = self._tres if self._tres > 0.0 else float(np.min(X))
        xmax = float(np.max(X))

        # Normalisation: probability mass in [xmin, xmax]
        d = np.sum(self.area * (np.exp(-xmin / self.tau) - np.exp(-xmax / self.tau)))
        if d < LOG_LIKELIHOOD_MIN:
            logger.warning("LL normalisation factor d = %g; possible numerical issue", d)
            d = LOG_LIKELIHOOD_MIN

        # Vectorised log-likelihood sum
        f = np.sum(
            (self.area[:, None] / self.tau[:, None]) * np.exp(-X[None, :] / self.tau[:, None]),
            axis=0,
        )
        f = np.clip(f, LOG_LIKELIHOOD_MIN, None)
        return float(-np.sum(np.log(f)) + len(X) * np.log(d))

    # ------------------------------------------------------------------ #
    # Fitting                                                               #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, *, tres: float = 0.0, **minimize_kwargs) -> OptimizeResult:
        """Fit by maximum likelihood using the Nelder-Mead simplex.

        Updates ``self.tau`` and ``self.area`` in-place.

        Parameters
        ----------
        X:
            Dwell-time observations (seconds).
        tres:
            Temporal resolution (seconds).  Used as the lower integration
            bound in the normalised log-likelihood.  Should match the value
            used when imposing resolution on the record.  If 0 (default),
            ``min(X)`` is used (old behaviour, but statistically biased when
            ``tres < min(X)``).
        **minimize_kwargs:
            Forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        OptimizeResult
        """
        self._tres = float(tres)
        X = np.asarray(X, dtype=float)
        logger.info("Start LL = %.6f", self.LL(self.theta, X))
        kwargs = {"method": "Nelder-Mead"}
        kwargs.update(minimize_kwargs)
        res = minimize(self.LL, self.theta, args=X, **kwargs)
        self._set_theta(res.x)
        logger.info("Final LL = %.6f  (%s)", res.fun, res.message)
        return res

    # ------------------------------------------------------------------ #
    # Derived quantities                                                    #
    # ------------------------------------------------------------------ #

    def mean(self) -> float:
        """Overall mean of the distribution (seconds)."""
        return float(np.sum(self.area * self.tau))

    def predicted_counts(self, X: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """Predicted number of events per component.

        Parameters
        ----------
        X:
            Observed data.

        Returns
        -------
        en : np.ndarray
            Predicted count per component.
        enout : list of float
            [predicted below xmin, predicted above xmax].
        """
        X = np.asarray(X, dtype=float)
        p1 = np.sum(self.area * np.exp(-np.min(X) / self.tau))
        p2 = np.sum(self.area * np.exp(-np.max(X) / self.tau))
        antrue = len(X) / (p1 - p2)
        en = antrue * self.area
        enout = [antrue * (1.0 - p1), antrue * p2]
        return en, enout

    def summary(self, X: np.ndarray) -> str:
        """Return a formatted fit summary string."""
        en, enout = self.predicted_counts(X)
        lines = []
        for i, (t, a, n) in enumerate(zip(self.tau, self.area, en)):
            lines.append(
                f"Component {i+1}: tau = {t*1e3:.4f} ms  (rate = {1/t:.2f} /s)  "
                f"area = {a:.4f}  predicted n = {n:.1f}"
            )
        lines.append(f"Overall mean = {self.mean()*1e3:.4f} ms")
        lines.append(f"Predicted true n = {sum(en):.1f}  fitted n = {len(X)}")
        lines.append(
            f"Predicted n below t_min = {enout[0]:.2f}  above t_max = {enout[1]:.2f}"
        )
        return "\n".join(lines)

    def get_tcrits(self, verbose: bool = False) -> None:
        """Populate ``self.tcrits`` with DC, C&N, and Jackson critical times.

        Parameters
        ----------
        verbose:
            If True, print a formatted misclassification report and summary
            table to stdout.
        """
        from ekdist.tcrit import Tcrit

        tc = Tcrit(self.tau, self.area)
        self.tcrits = tc.tcrits
        if verbose:
            for criterion, label in [
                ("DC",      "Equal % misclassified (DC criterion)"),
                ("C&N",     "Equal # misclassified (Clapham & Neher criterion)"),
                ("Jackson", "Minimum total # misclassified (Jackson et al criterion)"),
            ]:
                print(label)
                print(tc.misclassified_summary(criterion))
                print()
            print(tc.summary())


# =========================================================================== #
# Gaussian mixture                                                              #
# =========================================================================== #

class GaussianMixturePDF:
    """Multi-component Gaussian mixture PDF for amplitude distribution fitting.

    Parameters
    ----------
    means:
        Initial component means (pA).
    sigmas:
        Initial component standard deviations (pA; must all be positive).
    areas:
        Initial component areas (relative weights, sum to 1).  Accepts same
        forms as :class:`ExponentialPDF`.

    Examples
    --------
    >>> pdf = GaussianMixturePDF([10.0, 20.0], [1.0, 1.5])
    >>> res = pdf.fit(amplitudes)
    >>> print(pdf.means, pdf.sigmas, pdf.areas)
    """

    def __init__(
        self,
        means: list[float] | np.ndarray,
        sigmas: list[float] | np.ndarray,
        areas: list[float] | np.ndarray | None = None,
    ) -> None:
        means_arr = np.asarray(means, dtype=float)
        sigmas_arr = np.asarray(sigmas, dtype=float)
        if len(means_arr) != len(sigmas_arr):
            raise ValueError("means and sigmas must have the same length")
        if np.any(sigmas_arr <= 0):
            raise ValueError("All sigmas must be positive")

        k = len(means_arr)
        if areas is None:
            areas_arr = np.ones(k) / k
        else:
            areas_arr = np.asarray(areas, dtype=float)
            if len(areas_arr) == k - 1:
                areas_arr = np.append(areas_arr, 1.0 - np.sum(areas_arr))
            elif len(areas_arr) == k:
                areas_arr = areas_arr.copy()
                areas_arr[-1] = 1.0 - np.sum(areas_arr[:-1])
            else:
                raise ValueError(
                    f"areas must have length {k} or {k-1}; got {len(areas_arr)}"
                )

        self.means: np.ndarray = means_arr
        self.sigmas: np.ndarray = sigmas_arr
        self.areas: np.ndarray = areas_arr
        self.ncomps: int = k
        self.eqname: str = "gaussian mixture pdf"

        # pars = [means..., sigmas..., areas...]
        self.pars: np.ndarray = np.concatenate([self.means, self.sigmas, self.areas])
        self.fixed: list[bool] = [False] * (3 * k)
        self.fixed[-1] = True  # last area derived

    # ------------------------------------------------------------------ #
    # theta property                                                        #
    # ------------------------------------------------------------------ #

    def _get_theta(self) -> np.ndarray:
        return self.pars[np.where(~np.array(self.fixed))[0]]

    def _set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        for idx in np.nonzero(self.fixed)[0]:
            theta = np.insert(theta, idx, self.pars[idx])
        k = self.ncomps
        self.means, self.sigmas, self.areas = np.split(theta, [k, 2 * k])
        self.areas[-1] = 1.0 - np.sum(self.areas[:-1])
        self.pars = np.concatenate([self.means, self.sigmas, self.areas])

    theta = property(_get_theta, _set_theta)

    # ------------------------------------------------------------------ #
    # Density and log-likelihood                                            #
    # ------------------------------------------------------------------ #

    def pdf(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate the Gaussian mixture density at points *X*.

        Fully vectorised over components and data points.
        """
        self._set_theta(theta)
        X = np.asarray(X, dtype=float)
        # shape (k, n): broadcast means/sigmas/areas over all data points at once
        contrib = self.areas[:, None] * _norm.pdf(
            X[None, :], self.means[:, None], self.sigmas[:, None]
        )
        return np.sum(contrib, axis=0)

    def LL(self, theta: np.ndarray, X: np.ndarray) -> float:
        """Negative log-likelihood for a Gaussian mixture."""
        self._set_theta(theta)
        X = np.asarray(X, dtype=float)
        # Guard against negative sigmas produced by the optimiser
        if np.any(self.sigmas <= 0):
            return 1e10
        contrib = self.areas[:, None] * _norm.pdf(
            X[None, :], self.means[:, None], self.sigmas[:, None]
        )
        liks = np.clip(np.sum(contrib, axis=0), LOG_LIKELIHOOD_MIN, None)
        return float(-np.sum(np.log(liks)))

    # ------------------------------------------------------------------ #
    # Fitting                                                               #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, **minimize_kwargs) -> OptimizeResult:
        """Fit by maximum likelihood using Nelder-Mead.

        Updates ``self.means``, ``self.sigmas``, ``self.areas`` in-place.
        """
        X = np.asarray(X, dtype=float)
        logger.info("Start LL = %.6f", self.LL(self.theta, X))
        kwargs = {"method": "Nelder-Mead"}
        kwargs.update(minimize_kwargs)
        res = minimize(self.LL, self.theta, args=X, **kwargs)
        self._set_theta(res.x)
        logger.info("Final LL = %.6f  (%s)", res.fun, res.message)
        return res

    def mean(self) -> float:
        """Overall mean amplitude (pA)."""
        return float(np.sum(self.areas * self.means))

    def summary(self) -> str:
        """Formatted summary of fitted parameters."""
        lines = []
        for i, (mu, sigma, area) in enumerate(zip(self.means, self.sigmas, self.areas)):
            lines.append(
                f"Component {i+1}: mean = {mu:.4f} pA  sigma = {sigma:.4f} pA  "
                f"area = {area:.4f}"
            )
        lines.append(f"Overall mean = {self.mean():.4f} pA")
        return "\n".join(lines)

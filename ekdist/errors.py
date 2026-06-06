"""Parameter error estimation via Hessian inversion and likelihood contours.

Provides:

* :class:`ApproximateSD` — approximate standard deviations from the inverse
  Hessian of the log-likelihood at the MLE.
* :class:`LikelihoodIntervals` — confidence intervals by bisection on the
  likelihood surface.
"""

from __future__ import annotations

import copy
import logging
import math

import numpy as np
from numpy import linalg as nplin
from scipy import optimize

from ekdist._constants import HESSIAN_STEP_FACTOR

logger = logging.getLogger(__name__)


class ApproximateSD:
    """Approximate standard deviations from the inverse Hessian at the MLE.

    The Hessian of the *negative* log-likelihood is estimated by finite
    differences.  Its inverse is the covariance matrix of the parameters
    at the MLE (under the Cramér-Rao bound assumption).

    Parameters
    ----------
    theta:
        MLE parameter vector (free parameters only).
    func:
        Callable ``func(theta, data)`` returning the *negative* log-likelihood.
    arg:
        Data array passed as the second argument to *func*.
    delta_step:
        Finite-difference step as a fraction of each parameter value.
        Adaptive tuning adjusts this automatically.

    Attributes
    ----------
    hessian : np.ndarray
        Estimated Hessian matrix.
    covariance : np.ndarray
        Inverse of the Hessian (covariance matrix).
    sd : np.ndarray
        Square root of diagonal of covariance (approximate SDs).
    correlations : np.ndarray
        Pearson correlation matrix.
    """

    def __init__(
        self,
        theta: np.ndarray,
        func,
        arg: np.ndarray,
        delta_step: float = HESSIAN_STEP_FACTOR,
    ) -> None:
        theta = np.asarray(theta, dtype=float)
        self.hessian = self._hessian_matrix(theta, func, arg, delta_step)

        # Check conditioning before inverting
        cond = np.linalg.cond(self.hessian)
        if cond > 1e12:
            logger.warning(
                "Hessian is poorly conditioned (cond=%.2e); SDs may be unreliable", cond
            )

        try:
            self.covariance = nplin.inv(self.hessian)
        except nplin.LinAlgError as exc:
            raise ValueError("Hessian is singular; cannot compute covariance") from exc

        diag = self.covariance.diagonal()
        if np.any(diag < 0):
            logger.warning(
                "Negative diagonal elements in covariance matrix; "
                "Hessian may not be positive-definite at this point"
            )
        self.sd = np.sqrt(np.abs(diag))
        self.correlations = self._correlation_matrix(self.covariance)

    def _hessian_matrix(
        self,
        theta: np.ndarray,
        LLfunc,
        args: np.ndarray,
        delta_step: float,
    ) -> np.ndarray:
        hess = np.zeros((theta.size, theta.size))
        deltas = self._optimal_deltas(theta, LLfunc, args, delta_step)
        L0 = LLfunc(theta, args)

        # Diagonal elements
        for i in range(theta.size):
            di = deltas[i]
            tp = theta.copy(); tp[i] += di
            tn = theta.copy(); tn[i] -= di
            hess[i, i] = (LLfunc(tp, args) - 2.0 * L0 + LLfunc(tn, args)) / di**2

        # Off-diagonal elements
        for i in range(theta.size):
            for j in range(i + 1, theta.size):
                di, dj = deltas[i], deltas[j]
                coe1 = theta.copy(); coe1[i] += di; coe1[j] += dj
                coe2 = theta.copy(); coe2[i] += di; coe2[j] -= dj
                coe3 = theta.copy(); coe3[i] -= di; coe3[j] += dj
                coe4 = theta.copy(); coe4[i] -= di; coe4[j] -= dj
                val = (
                    LLfunc(coe1, args) - LLfunc(coe2, args)
                    - LLfunc(coe3, args) + LLfunc(coe4, args)
                ) / (4.0 * di * dj)
                hess[i, j] = val
                hess[j, i] = val

        return hess

    def _tune_deltas(
        self,
        theta: np.ndarray,
        func,
        args: np.ndarray,
        Lcrit: float,
        deltas: np.ndarray,
        increase: bool,
    ) -> np.ndarray:
        factor, scale = (1, 2.0) if increase else (-1, 0.5)
        count = 0
        while factor * func(theta + deltas, args) < factor * Lcrit and count < 100:
            deltas = deltas * scale
            count += 1
        return deltas

    def _optimal_deltas(
        self,
        theta: np.ndarray,
        LLfunc,
        args: np.ndarray,
        step_factor: float,
    ) -> np.ndarray:
        L0 = LLfunc(theta, args)
        Lcrit = L0 + math.fabs(L0 * 0.005)
        deltas = step_factor * np.abs(theta)
        # Avoid zero deltas for parameters near zero
        deltas = np.where(deltas == 0, step_factor, deltas)
        L = LLfunc(theta + deltas, args)
        if L < Lcrit:
            deltas = self._tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=True)
        elif L > Lcrit:
            deltas = self._tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=False)
        return deltas

    @staticmethod
    def _correlation_matrix(covar: np.ndarray) -> np.ndarray:
        n = len(covar)
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Use abs to guard against negative diagonal elements when
                # the Hessian is not positive-definite at this parameter point.
                denom = math.sqrt(abs(covar[i, i] * covar[j, j]))
                corr[i, j] = covar[i, j] / denom if denom > 0 else 0.0
        return corr


class LikelihoodIntervals:
    """Likelihood-based confidence intervals by bisection on the LL surface.

    For each free parameter, holds all other parameters at their MLE values
    and finds the parameter values where the log-likelihood drops by *m* units
    (corresponding roughly to *m* standard deviations for large samples).

    Parameters
    ----------
    theta:
        MLE free parameter vector.
    pdf:
        Fitted distribution object (must expose ``LL(theta, data)`` and
        ``fixed`` / ``pars`` attributes following the :class:`ExponentialPDF`
        convention).
    data:
        Observed data.
    SD:
        Approximate standard deviations from :class:`ApproximateSD`.
    m:
        Likelihood drop defining the interval (e.g., m=2 ≈ 2 SD).
    """

    def __init__(
        self,
        theta: np.ndarray,
        pdf,
        data: np.ndarray,
        SD: np.ndarray,
        m: float = 2.0,
    ) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.pdf = pdf
        self.data = np.asarray(data, dtype=float)
        self.SD = np.asarray(SD, dtype=float)
        self.m = float(m)
        self.Lmax = -pdf.LL(self.theta, self.data)
        self.clim = math.sqrt(2.0 * m)
        self.Lcrit = self.Lmax - m

    def calculate(self) -> np.ndarray:
        """Compute lower and upper likelihood limits for each free parameter.

        Returns
        -------
        np.ndarray, shape (n_params, 2)
            ``result[i] = [lower_limit, upper_limit]`` for parameter *i*.
        """
        logger.info("Calculating likelihood intervals…")
        limits = []
        for j in range(len(self.theta)):
            val = self.theta[j]
            sd = self.SD[j]
            xhi1, xhi2 = val, val + 5.0 * self.clim * sd
            xlo1, xlo2 = max(0.0, val - 2.0 * self.clim * sd), val
            logger.debug(
                "Parameter %d = %.4f  lower search [%.4f, %.4f]  upper [%.4f, %.4f]",
                j, val, xlo1, xlo2, xhi1, xhi2,
            )
            lo = self._find_limit(j, xlo1, xlo2, factor=1.0)
            hi = self._find_limit(j, xhi1, xhi2, factor=-1.0)
            limits.append([lo, hi])
        return np.array(limits)

    def _find_limit(
        self, index: int, low: float, high: float, factor: float
    ) -> float | None:
        # For lower-limit searches: if the LL at the boundary is already above
        # Lcrit, the CI extends all the way to the boundary — return it directly
        # rather than running 100 iterations and silently returning None.
        if factor > 0:
            L_low = self._contour_LL(low, index)
            if math.isfinite(L_low) and L_low >= self.Lcrit:
                return low

        for _ in range(100):
            mid = (low + high) / 2.0
            L = self._contour_LL(mid, index)
            if math.fabs(self.Lcrit - L) <= 0.01:
                return mid  # mid is always >= 0 given search range
            if factor * L < factor * self.Lcrit:
                low = mid
            else:
                high = mid
        logger.warning("Likelihood interval bisection did not converge for parameter %d", index)
        return None

    def _contour_LL(self, x: float, num: int) -> float:
        """LL with parameter *num* fixed at *x*, all others optimised."""
        func = copy.deepcopy(self.pdf)
        func.fixed[num] = True
        func.pars[num] = x
        theta_free = func.theta
        res = optimize.minimize(func.LL, theta_free, args=self.data, method="Nelder-Mead")
        return -res.fun

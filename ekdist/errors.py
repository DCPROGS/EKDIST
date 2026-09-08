import math
import copy
import numpy as np
from numpy import linalg as nplin
from scipy import optimize

class LikelihoodIntervals:
    def __init__(self, theta, pdf, data, SD, m):
        self.SD = SD
        self.m = m
        self.pdf = pdf
        self.theta = theta
        self.data = data
        self.Lmax = -pdf.LL(self.theta, self.data)
        self.clim = math.sqrt(2. * m)
        self.Lcrit = self.Lmax - m
        #self.Llimits = self.calculate(theta, pdf, arg)
        
    def calculate(self): #, theta, pdf, arg):
        print('calculating likelihood intervals...')
        Llimits = []
        i = 0
        for j in range(len(self.theta)):
            xhigh1, xhigh2 = self.theta[i], self.theta[i] + 5 * self.clim * self.SD[i]
            xlow1, xlow2 = self.theta[i] - 2 * self.clim * self.SD[i], xhigh1
            if xlow1 < 0: xlow1 = 0.0

            print('\nCalculating Lik limits for parameter- {0} = {1:.3f}'.
                  format(self.pdf.names[j], self.theta[i]))
            print('\tInitial guesses for lower limit: {0:.3f} and {1:.3f}'.
                  format(xlow1, xhigh1))
            print('\tInitial guesses for higher limit: {0:.3f} and {1:.3f}'.
                  format(xlow2, xhigh2))

            xlowlim = self.__get_limit(j, xlow1, xhigh1, factor=1.)
            xhighlim = self.__get_limit(j, xlow2, xhigh2, factor=-1.)
            Llimits.append([xlowlim, xhighlim])
            i += 1
        return Llimits

    def __get_limit(self, index, low, high, factor=1.):
        limit = None
        found = False
        iter = 0
        while not found and iter < 100:
            L = self.__lik_contour(((low + high) / 2), index, 
                                self.theta, self.pdf, self.data) 
            if math.fabs(self.Lcrit - L) > 0.01:
                low, high = self.__adjust_guesses(low, high, L, factor)
            else:
                limit, found = self.__finalize_limit(low, high)
            iter += 1
        return limit

    def __finalize_limit(self, low, high):
        limit = (low + high) / 2
        if limit < 0: limit = None
        #print ('limit found: ', limit)
        return limit, True

    def __adjust_guesses(self, low, high, L, factor=1.):
        if L * factor < self.Lcrit * factor:
            low = (low + high) / 2
        else:
            high = (low + high) / 2
        return low, high

    def __lik_contour(self, x, num, theta, func, data):
        functemp = copy.deepcopy(func)
        functemp.fixed[num] = True
        functemp.pars[num] = x
        theta = functemp.theta
        result = optimize.minimize(functemp.LL, theta, args=data, method='Nelder-Mead')
        return -result.fun

class ApproximateSD:
    """Approximate standard deviations and correlations from the Hessian.

    The Hessian of the function being minimised is estimated by finite
    differences, inverted to give the covariance matrix, and from that the
    standard deviation of each estimate and the correlation between each pair.
    ``func`` is the quantity that was *minimised* -- minus the log likelihood --
    so it increases as the parameters move away from the fitted values.

    Everything turns on the increments used for the finite differences. Two
    rules are available.

    **The default**, unchanged since this class was written, sizes a single
    relative increment so that moving *every* parameter together by that
    relative amount changes ``func`` by 0.5% of its own value.

    **The absolute rule**, selected by giving ``delta_L``, is the one described
    by Colquhoun, Hatton & Hawkes (2003), p. 702: the increment is found
    separately **for each parameter**, and is the one that changes ``func`` by
    ``delta_L`` -- an absolute amount, not a fraction of the value. Where no
    such increment can be found, because the fit is insensitive to that
    parameter, its row and column are omitted from the Hessian before it is
    inverted, exactly as the paper describes, and its index is listed in
    :attr:`dropped`.

    Which rule to use matters when the likelihood surface is strongly
    anisotropic. Moving all the parameters together, as the default does, moves
    along the diagonal ridge that such a surface has, where the function changes
    slowly; the increment that results can be far too large for the directions
    across the ridge.

    Parameters
    ----------
    theta : ndarray
        The fitted parameters.
    func : callable
        ``func(theta, arg) -> float``, the quantity minimised.
    arg : object
        Passed through to ``func``.
    delta_step : float
        Relative size of the first increment tried, before tuning.
    delta_L : float, optional
        Selects the absolute rule, and is the change in ``func`` each increment
        is tuned to produce. Note that the paper's 0.1 "log units" are natural
        logarithms; a likelihood in log10 wants 0.1/ln(10) = 0.0434.
    max_relative_delta : float
        An increment larger than this multiple of its own parameter counts as
        not found, and the parameter is dropped. Absolute rule only.

    Attributes
    ----------
    hessian, covariance : ndarray
        Square, over the parameters actually used -- smaller than ``theta`` if
        any were dropped.
    kept, dropped : tuple of int
        Indices into ``theta``.
    sd, correlations : ndarray
        Full size, indexed by parameter, with ``nan`` for dropped parameters,
        so that ``sd[i]`` always refers to ``theta[i]``.
    deltas : ndarray
        The increments used; ``nan`` for dropped parameters.
    """

    def __init__(self, theta, func, arg, delta_step=0.0001,
                 delta_L=None, max_relative_delta=100.0):
        theta = np.asarray(theta, float)
        if delta_L is None:
            self.deltas = self.__optimal_deltas(theta, func, arg, delta_step)
            self.dropped = ()
        else:
            self.deltas = self.__absolute_deltas(
                theta, func, arg, delta_step, delta_L, max_relative_delta)
            self.dropped = tuple(
                int(i) for i in np.flatnonzero(~np.isfinite(self.deltas)))
        self.kept = tuple(i for i in range(theta.size) if i not in self.dropped)

        self.hessian = self.hessian_matrix(theta, func, arg, delta_step,
                                           deltas=self.deltas, keep=self.kept)
        self.covariance = nplin.inv(self.hessian)

        self.sd = np.full(theta.size, np.nan)
        self.sd[list(self.kept)] = np.sqrt(self.covariance.diagonal())
        reduced = self.correlation_matrix(self.covariance)
        self.correlations = np.full((theta.size, theta.size), np.nan)
        for a, i in enumerate(self.kept):
            for b, j in enumerate(self.kept):
                self.correlations[i, j] = reduced[a, b]

    def hessian_matrix(self, theta, LLfunc, args, delta_step=0.0001,
                       deltas=None, keep=None):
        """Second derivatives of ``LLfunc`` by central differences.

        ``deltas`` and ``keep`` are supplied by :meth:`__init__`; called
        directly with neither, this behaves as it always has.
        """
        theta = np.asarray(theta, float)
        if deltas is None:
            deltas = self.__optimal_deltas(theta, LLfunc, args, delta_step)
        if keep is None:
            keep = tuple(range(theta.size))
        if len(keep) != theta.size:
            return self.__hessian_subset(theta, LLfunc, args, deltas, keep)

        hess = np.zeros((theta.size, theta.size))
        # Diagonal elements of Hessian
        coe11 = np.array([theta.copy(), ] * theta.size) + np.diag(deltas)
        coe33 = np.array([theta.copy(), ] * theta.size) - np.diag(deltas)
        for i in range(theta.size):
            hess[i, i] = ((LLfunc(coe11[i], args) - 
                2.0 * LLfunc(theta, args) +
                LLfunc(coe33[i], args)) / (deltas[i]  ** 2))
        # Non diagonal elements of Hessian
        for i in range(theta.size):
            for j in range(theta.size):
                coe1, coe2, coe3, coe4 = theta.copy(), theta.copy(), theta.copy(), theta.copy()
                if i != j:                
                    coe1[i] += deltas[i]
                    coe1[j] += deltas[j]
                    coe2[i] += deltas[i]
                    coe2[j] -= deltas[j]
                    coe3[i] -= deltas[i]
                    coe3[j] += deltas[j]
                    coe4[i] -= deltas[i]
                    coe4[j] -= deltas[j]
                    hess[i, j] = ((
                        LLfunc(coe1, args) -
                        LLfunc(coe2, args) -
                        LLfunc(coe3, args) +
                        LLfunc(coe4, args)) /
                        (4 * deltas[i] * deltas[j]))
        return hess

    def __hessian_subset(self, theta, LLfunc, args, deltas, keep):
        """The Hessian over ``keep`` only, the rest omitted as in the paper."""
        keep = list(keep)
        n = len(keep)
        hess = np.zeros((n, n))
        L0 = LLfunc(theta, args)
        for a, i in enumerate(keep):
            up, down = theta.copy(), theta.copy()
            up[i] += deltas[i]
            down[i] -= deltas[i]
            hess[a, a] = ((LLfunc(up, args) - 2.0 * L0 + LLfunc(down, args))
                          / deltas[i] ** 2)
        for a, i in enumerate(keep):
            for b, j in enumerate(keep):
                if a >= b:
                    continue
                pp, pm, mp, mm = (theta.copy(), theta.copy(),
                                  theta.copy(), theta.copy())
                pp[i] += deltas[i]; pp[j] += deltas[j]
                pm[i] += deltas[i]; pm[j] -= deltas[j]
                mp[i] -= deltas[i]; mp[j] += deltas[j]
                mm[i] -= deltas[i]; mm[j] -= deltas[j]
                value = ((LLfunc(pp, args) - LLfunc(pm, args)
                          - LLfunc(mp, args) + LLfunc(mm, args))
                         / (4 * deltas[i] * deltas[j]))
                hess[a, b] = hess[b, a] = value
        return hess

    def __change_in(self, theta, func, args, index, delta):
        """How much ``func`` rises when parameter ``index`` moves by ``delta``.

        ``inf`` if it cannot be evaluated there, which counts as too large a
        step and makes the caller try a smaller one.
        """
        trial = theta.copy()
        trial[index] += delta
        try:
            value = func(trial, args)
        except (ArithmeticError, ValueError, RuntimeError):
            return np.inf
        if not np.isfinite(value):
            return np.inf
        return value - self.__L0

    def __absolute_deltas(self, theta, func, args, step_factor, delta_L,
                          max_relative):
        """One increment per parameter, each changing ``func`` by ``delta_L``.

        The rule of Colquhoun, Hatton & Hawkes (2003), p. 702. Returns ``nan``
        for a parameter whose increment could not be found, which is how a
        parameter the fit is insensitive to gets dropped.
        """
        self.__L0 = func(theta, args)
        deltas = np.full(theta.size, np.nan)
        for i in range(theta.size):
            scale = abs(theta[i]) if theta[i] != 0.0 else 1.0
            biggest = max_relative * scale
            delta = step_factor * scale
            change = self.__change_in(theta, func, args, i, delta)
            if change < delta_L:                        # too small: grow it
                while change < delta_L and delta < biggest:
                    delta *= 2.0
                    change = self.__change_in(theta, func, args, i, delta)
            else:                                       # too big: shrink it
                smallest = 1e-12 * scale
                while change > delta_L and delta > smallest:
                    delta /= 2.0
                    change = self.__change_in(theta, func, args, i, delta)
                delta *= 2.0                            # the last one that was
                change = self.__change_in(theta, func, args, i, delta)
            if np.isfinite(change) and change >= delta_L and delta <= biggest:
                deltas[i] = delta
        return deltas

    def __tune_deltas(self, theta, func, args, Lcrit, deltas, increase=True):
        factor = [1, 2] if increase else [-1, 0.5]
        count = 0
        while factor[0] * func(theta + deltas, args) < factor[0] * Lcrit and count < 100:
            deltas *= factor[1]
            count += 1
        return deltas

    def __optimal_deltas(self, theta, LLfunc, args, step_factor=0.0001):
        Lcrit = LLfunc(theta, args) + math.fabs(LLfunc(theta, args) * 0.005)
        deltas = step_factor * theta
        L = LLfunc(theta + deltas, args)
        if L < Lcrit:
            deltas = self.__tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=True)
        elif L > Lcrit:
            deltas = self.__tune_deltas(theta, LLfunc, args, Lcrit, deltas, increase=False)
        return deltas

    def correlation_matrix(self, covar):
        correl = np.zeros((len(covar),len(covar)))
        for i1 in range(len(covar)):
            for j1 in range(len(covar)):
                correl[i1,j1] = (covar[i1,j1] / 
                    np.sqrt(np.multiply(covar[i1,i1],covar[j1,j1])))
        return correl


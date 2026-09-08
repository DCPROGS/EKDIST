"""Two populations in one set of estimates, and how to separate them.

Fit a mechanism to many simulated records and the estimates do not always form
one cloud. Some records take the optimiser to a second maximum -- typically one
where most events are missed -- and the results come back as two groups an
order of magnitude apart. Any summary that ignores this is meaningless: the
mean of a set like that describes neither group.

Two ways of finding the split, for two shapes of problem:

:func:`mixture2`
    fits two Gaussian components in log space, for two groups that overlap or
    nearly do. Reports the proportion in each and their geometric means.
:func:`gap_split`
    finds the largest multiplicative step between neighbouring sorted values,
    for a handful of runaways cleanly separated from the body.

**Why not just pick a threshold.** Choosing a cut by eye and counting either
side makes the answer a property of the cut. Both functions here determine the
split from the data, so an exclusion can be *stated* -- "16 values above a
55-fold gap" -- rather than assumed, and someone else can check it.

**Why log space.** Rate constants are positive and their errors are
multiplicative, so a group of them is roughly log-normal, not normal. Fitting
in logs keeps every fitted mean positive and makes each group closer to the
Gaussian the mixture assumes. It also means :attr:`Mixture.means` are geometric
means.

These were written for the reproduction of Colquhoun, Hatton & Hawkes (2003)
*J Physiol* **547**:699-728, whose Fig. 2 is exactly this situation: 1000 fits
of the same mechanism falling into a group near the true rates and a second
group where alpha2 comes out near 17 000 instead of 2000. Their examples appear
below because a concrete case is easier to read than an abstract one, but
nothing here is specific to that paper or to ion channels.

Usage::

    from ekdist import mixtures

    m = mixtures.mixture2(np.column_stack([alpha2, beta2]))
    print(m.means, m.proportion(1))         # per cent, with its binomial SE
    good = m.cluster(alpha2, 0)

    keep, ratio, cut = mixtures.gap_split(beta1b)
    if cut is not None:
        print(f"excluded {(~keep).sum()} values above a {ratio:.0f}-fold gap")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def proportion(k, n):
    """A proportion and its binomial standard error, **both per cent**."""
    p = k / n
    return 100.0 * p, 100.0 * float(np.sqrt(p * (1.0 - p) / n))


@dataclass(frozen=True)
class Mixture:
    """A two-component Gaussian mixture, fitted in log space by :func:`mixture2`.

    Attributes
    ----------
    weights : ndarray, shape (2,)
        Mixing proportions, summing to 1, ordered so that component 0 is the
        one with the smaller first coordinate.
    means : ndarray, shape (2, d)
        Component means in the **original** units, obtained by exponentiating
        the log-space means, so they are geometric means.
    labels : ndarray of int, shape (n,)
        Each point assigned to its most probable component.
    responsibilities : ndarray, shape (n, 2)
        The posterior probability of each component for each point. Use these
        rather than ``labels`` when the two groups overlap enough that a hard
        assignment throws away what the fit knows.
    loglik : float
        Mean log-likelihood per point at the last EM step.
    n_iter : int
    converged : bool
        Whether the change in ``loglik`` fell below the tolerance before the
        iteration cap. A fit that did not converge is not necessarily wrong,
        but it should not be quoted without looking at it.
    """

    weights: np.ndarray
    means: np.ndarray
    labels: np.ndarray
    responsibilities: np.ndarray
    loglik: float
    n_iter: int
    converged: bool

    def cluster(self, x, which):
        """The members of ``x`` assigned to component ``which``.

        ``x`` is indexed by the same order as the data that was fitted, so it
        can be any per-point quantity, not only a fitted coordinate.
        """
        return np.asarray(x)[self.labels == which]

    def proportion(self, which=1):
        """Proportion in one component, per cent, with its binomial SE.

        Taken from the hard assignment rather than from the fitted weight, so
        that it is a count of points and its error is the error of a count.
        The two agree well within that error when the groups are separated;
        when they disagree, the groups overlap and neither number means much on
        its own.
        """
        return proportion(int((self.labels == which).sum()), self.labels.size)


def mixture2(X, n_iter=500, tol=1e-8):
    """Fit two Gaussian components to positive data, in log space.

    Parameters
    ----------
    X : array_like, shape (n,) or (n, d)
        Strictly positive. Several coordinates at once are better than one:
        two groups that overlap in every single parameter can still separate
        cleanly in two, and the correlations within each group help.
    n_iter, tol : int, float
        EM iteration cap, and the change in mean log-likelihood per point that
        counts as converged.

    Returns
    -------
    Mixture

    Notes
    -----
    Started from a split at the median of the first coordinate, which for a
    genuinely bimodal set lies inside the gap, so no random restarts are
    needed and the result is deterministic.

    **A set that is not bimodal will still return two components.** Nothing
    here tests whether two is the right number. The separation of the means and
    the proportion are what say whether the answer means anything: two
    components fitted to one population typically land within a factor of two
    of each other, where a real split is an order of magnitude.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < 4:
        raise ValueError("need at least four points to fit two components")
    if not np.all(X > 0):
        raise ValueError("mixture2 fits in log space; all values must be > 0")

    Y = np.log(X)
    n, d = Y.shape
    # a split at the median of the first coordinate, which lies in the gap
    hard = (Y[:, 0] > np.median(Y[:, 0])).astype(int)
    resp = np.zeros((n, 2))
    resp[np.arange(n), hard] = 1.0

    ridge = 1e-9 * np.eye(d)          # keeps a near-degenerate component safe
    prev = -np.inf
    ll = -np.inf
    converged = False
    it = 0
    w = mu = None
    for it in range(1, int(n_iter) + 1):
        # M step
        nk = resp.sum(axis=0) + 1e-300
        w = nk / n
        mu = (resp.T @ Y) / nk[:, None]
        cov = np.empty((2, d, d))
        for k in range(2):
            D = Y - mu[k]
            cov[k] = (D * resp[:, k, None]).T @ D / nk[k] + ridge

        # E step
        logp = np.empty((n, 2))
        for k in range(2):
            sign, logdet = np.linalg.slogdet(cov[k])
            if sign <= 0:
                raise np.linalg.LinAlgError("degenerate mixture component")
            D = Y - mu[k]
            m = np.einsum("ij,jk,ik->i", D, np.linalg.inv(cov[k]), D)
            logp[:, k] = np.log(w[k]) - 0.5 * (d * np.log(2 * np.pi)
                                               + logdet + m)
        top = logp.max(axis=1, keepdims=True)
        lse = top[:, 0] + np.log(np.exp(logp - top).sum(axis=1))
        resp = np.exp(logp - lse[:, None])

        ll = float(lse.mean())
        if abs(ll - prev) < tol:
            converged = True
            break
        prev = ll

    order = np.argsort(mu[:, 0])       # component 0 = the smaller first coord
    w, mu, resp = w[order], mu[order], resp[:, order]
    return Mixture(weights=w, means=np.exp(mu),
                   labels=resp.argmax(axis=1), responsibilities=resp,
                   loglik=ll, n_iter=it, converged=converged)


def gap_split(x, min_ratio=5.0):
    """Find a high outlier population separated by a multiplicative gap.

    For the case a mixture is the wrong tool for: a few values far above the
    rest with nothing in between. In one of the CHH 2003 scenarios, 234
    estimates of beta1b lie below 671 and 16 above 37 180, with nothing
    between -- a 55-fold step. Excluding them at a round number would be an
    arbitrary cut; this finds the **largest ratio between neighbouring sorted
    values** and, if it exceeds ``min_ratio``, splits there.

    Parameters
    ----------
    x : array_like
        Strictly positive.
    min_ratio : float
        The smallest gap that counts as a separation. 5 is well above the
        largest step inside a unimodal set of a few hundred values, and well
        below the 55-fold step above. Raise it if the body of the distribution
        is itself broad.

    Returns
    -------
    body : ndarray of bool
        True for the members below the gap. **All True when no gap is found**,
        so that the caller can apply the mask unconditionally.
    ratio : float
        The largest neighbouring ratio, whether or not it was taken -- so that
        "no gap" can be reported with a number rather than as a bare negative.
    cut : float or None
        The last value below the gap, or None when no gap was found.

    Notes
    -----
    Only the largest gap is considered, and only a high tail is separated. A
    set with two gaps, or with outliers at both ends, needs looking at rather
    than a function.
    """
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(x > 0):
        raise ValueError("gap_split needs positive values")
    s = np.sort(x)
    ratios = s[1:] / s[:-1]
    i = int(np.argmax(ratios))
    ratio = float(ratios[i])
    if ratio < min_ratio:
        return np.ones(x.shape, dtype=bool), ratio, None
    cut = float(s[i])
    return x <= cut, ratio, cut

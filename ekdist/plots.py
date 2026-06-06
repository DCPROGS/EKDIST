"""Visualization for single-channel analysis.

All functions return a ``matplotlib.figure.Figure`` so callers can save or
further customise the output.  No ``print()`` calls are made; use the
``logging`` module for diagnostic output.

Functions
---------
stability_intervals        Moving-average stability plot (open/shut/Popen).
stability_rolling          3-panel rolling-mean stability plot (open/shut/Popen).
stability_amplitudes       Scatter plot of open amplitudes over time.
histogram_amplitudes       Amplitude histogram with single/multi-Gaussian fit.
histogram_burst_openings   Distribution of openings per burst.
histogram_dwell_times      Log-x / sqrt-y dwell-time histogram with PDF overlay.
plot_serial_correlation    Autocorrelation coefficients vs lag with significance band.
plot_open_shut_scatter     Open-shut interval scatter + 2-D density histogram.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

import matplotlib.colors as _mcolors

from ekdist import utils
from ekdist._constants import (
    HIST_BINS_DEFAULT,
    _HIST_BINS_THRESHOLDS,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
# Stability plots                                                               #
# =========================================================================== #

def stability_intervals(
    rec,
    *,
    show_open: bool = True,
    show_shut: bool = True,
    show_popen: bool = True,
    window: int = 50,
    figsize: tuple = (8, 3),
) -> plt.Figure:
    """Moving-average stability plot of open, shut, and Popen over time.

    Parameters
    ----------
    rec:
        A :class:`~ekdist.record.SingleChannelRecord` instance.
    show_open, show_shut, show_popen:
        Toggle each trace.
    window:
        Moving-average window (number of intervals).
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    opens = rec.periods.open_intervals[:-1]  # drop last (incomplete)
    shuts = rec.periods.shut_intervals
    opma, shma, poma = utils.moving_average_open_shut_popen(opens, shuts, window)
    x = np.arange(len(opma))

    fig, ax = plt.subplots(figsize=figsize)
    if show_open:
        ax.semilogy(x, opma, "g", label="Open periods")
    if show_shut:
        ax.semilogy(x, shma, "r", label="Shut periods")
    if show_popen:
        ax.semilogy(x, poma, "b", label="Popen")
    ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=3, borderaxespad=0.0)
    ax.set_xlabel("Interval number")
    ax.set_ylabel("Duration (s) / Popen")
    return fig


def stability_amplitudes(
    rec,
    *,
    window: int = 1,
    figsize: tuple = (8, 3),
) -> plt.Figure:
    """Scatter/moving-average plot of resolved open amplitudes over time.

    Parameters
    ----------
    rec:
        A :class:`~ekdist.record.SingleChannelRecord` instance.
    window:
        Moving-average window (1 = no smoothing).
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rampl = np.asarray(rec.rampl)
    open_amps = rampl[np.abs(rampl) > 0.0]
    amps = utils.moving_average(open_amps, window)
    logger.info("Average open amplitude = %.4f pA", float(np.mean(amps)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(amps, ".g", markersize=2)
    ax.set_ylabel("Amplitude (pA)")
    ax.set_xlabel("Interval number")
    return fig


# =========================================================================== #
# Amplitude histogram                                                           #
# =========================================================================== #

def histogram_amplitudes(
    rec,
    fc: float,
    *,
    n_risetimes: float = 2.0,
    nbins: int = 20,
    pdf=None,
    figsize: tuple = (6, 4),
) -> plt.Figure:
    """Amplitude histogram for openings longer than *n_risetimes* rise-times.

    If *pdf* is a :class:`~ekdist.fitting.GaussianMixturePDF` (or a fitted
    :class:`~scipy.stats.norm`-like object exposing ``.pdf``), it is overlaid
    on the histogram.  Otherwise a single Gaussian is fitted automatically.

    Parameters
    ----------
    rec:
        A :class:`~ekdist.record.SingleChannelRecord` instance.
    fc:
        Filter -3 dB cut-off frequency (Hz).
    n_risetimes:
        Minimum opening length as a multiple of the filter rise time.
    nbins:
        Number of histogram bins.
    pdf:
        Optional pre-fitted PDF object (must expose ``.pdf(theta, X)`` or
        be callable as ``pdf(X)``).
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    long_amps = utils.amplitudes_openings_longer_than(
        np.asarray(rec.rtint), np.asarray(rec.rampl), fc, n_risetimes
    )
    logger.info(
        "Amplitude range: %.4f – %.4f pA  (n=%d)",
        float(np.min(long_amps)), float(np.max(long_amps)), len(long_amps),
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(long_amps, nbins, density=True, alpha=0.6, color="steelblue")

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 300)

    if pdf is None:
        # Single Gaussian auto-fit
        mu, sigma = float(np.mean(long_amps)), float(np.std(long_amps))
        ax.plot(x, _norm.pdf(x, mu, sigma), "k-", linewidth=2, label=f"Gaussian μ={mu:.2f}")
        ax.set_title(f"μ = {mu:.3f} pA   σ = {sigma:.3f} pA")
    else:
        try:
            y = pdf.pdf(pdf.theta, x)
        except AttributeError:
            y = pdf(x)
        ax.plot(x, y, "k-", linewidth=2, label="Fit")
        # Individual components for GaussianMixturePDF
        if hasattr(pdf, "means"):
            for mu, sigma, area in zip(pdf.means, pdf.sigmas, pdf.areas):
                ax.plot(x, area * _norm.pdf(x, mu, sigma), "--", linewidth=1)

    ax.set_xlim([0.0, 1.2 * float(np.max(long_amps))])
    ax.set_xlabel("Amplitude (pA)")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


# =========================================================================== #
# Burst histogram                                                               #
# =========================================================================== #

def histogram_burst_openings(
    n_openings: list[int] | np.ndarray,
    *,
    figsize: tuple = (10, 3),
) -> plt.Figure:
    """Histogram of number of openings per burst.

    Parameters
    ----------
    n_openings:
        List or array of opening counts per burst.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_openings = np.asarray(n_openings)
    bins = np.arange(0.5, float(np.max(n_openings)) + 1.5, 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(n_openings, bins=bins, histtype="step", color="steelblue")
    ax.set_xlabel("Number of openings per burst")
    ax.set_ylabel("Count")
    return fig


# =========================================================================== #
# Dwell-time histogram                                                          #
# =========================================================================== #

def _bins_per_decade(n: int) -> int:
    for lo, hi, nbdec in _HIST_BINS_THRESHOLDS:
        if lo < n <= hi:
            return nbdec
    return HIST_BINS_DEFAULT


def _bin_width(X: np.ndarray) -> float:
    return math.exp(math.log(10.0) / float(_bins_per_decade(len(X))))


def _scale_factor(X: np.ndarray, pdf, tres: float) -> float:
    bw = _bin_width(X)
    # Fraction of the distribution with t >= tres.
    # Analytical formula for ExponentialPDF; numerical integration for anything else.
    if hasattr(pdf, "area") and hasattr(pdf, "tau"):
        norm = float(np.sum(pdf.area * np.exp(-tres / pdf.tau)))
    else:
        from scipy.integrate import quad
        norm, _ = quad(
            lambda t: float(pdf.pdf(pdf.theta, np.array([t]))[0]),
            tres,
            np.inf,
            limit=200,
        )
    return len(X) * math.log10(bw) * math.log(10.0) / norm


def prepare_xlog_hist(X: np.ndarray, tres: float) -> tuple[list, list]:
    """Compute x and y arrays for a log-x dwell-time histogram.

    Parameters
    ----------
    X:
        Dwell-time observations (seconds).
    tres:
        Temporal resolution (histogram starts here).

    Returns
    -------
    xout, yout : list
        x and y values suitable for plotting with ``ax.semilogx``.
    """
    X = np.asarray(X, dtype=float)
    dx = _bin_width(X)
    xend = math.exp(math.ceil(math.log(float(np.max(X)))))
    nbin = int(math.log(xend / tres) / math.log(dx))
    bins = tres * np.array([dx ** i for i in range(nbin + 1)])
    hist, edges = np.histogram(X, bins=bins)
    xout = [v for pair in zip(edges, edges) for v in pair]
    yout = [0] + [v for pair in zip(hist, hist) for v in pair] + [0]
    return xout, yout


def histogram_dwell_times(
    X: np.ndarray,
    tres: float,
    *,
    pdf=None,
    tcrits: list | np.ndarray | None = None,
    xlabel: str = "Dwell times (s)",
    title: str | None = None,
    figsize: tuple = (5, 4),
) -> plt.Figure:
    """Log-x / sqrt-y dwell-time histogram with optional PDF overlay.

    Parameters
    ----------
    X:
        Dwell-time observations (seconds).
    tres:
        Temporal resolution (seconds); also sets the left edge of the histogram.
    pdf:
        Optional fitted :class:`~ekdist.fitting.ExponentialPDF`.  If None, a
        single-component exponential with mean = mean(X) is shown.
    tcrits:
        Optional list of critical times (seconds) to draw as vertical lines.
    xlabel:
        x-axis label.
    title:
        Optional figure title.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from ekdist.fitting import ExponentialPDF

    X = np.asarray(X, dtype=float)
    xout, yout = prepare_xlog_hist(X, tres)

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogx(xout, np.sqrt(yout), color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("√frequency")
    if title is not None:
        ax.set_title(title)

    if pdf is None:
        pdf = ExponentialPDF([float(np.mean(X))], [1.0])

    t = np.logspace(math.log10(tres), math.log10(2.0 * float(np.max(X))), 512)
    scale = _scale_factor(X, pdf, tres)
    ax.plot(t, np.sqrt(scale * t * pdf.pdf(pdf.theta, t)), "-b", linewidth=1.5)
    # Per-component overlay is only meaningful for ExponentialPDF
    if hasattr(pdf, "tau") and hasattr(pdf, "area"):
        for ta, ar in zip(pdf.tau, pdf.area):
            ax.plot(
                t,
                np.sqrt(scale * t * (ar / ta) * np.exp(-t / ta)),
                "--b",
                linewidth=0.8,
                alpha=0.7,
            )

    if tcrits is not None:
        for tc in np.asarray(tcrits):
            ax.axvline(x=tc, color="green", linestyle="--", alpha=0.8)

    return fig


# =========================================================================== #
# Rolling-mean stability plots                                                   #
# =========================================================================== #

def stability_rolling(
    open_intervals: np.ndarray,
    shut_intervals: np.ndarray,
    *,
    window: int = 100,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """Three-panel rolling-mean stability plot: open periods, shut periods, Popen.

    Plots valid sliding-window means (no edge-padding artefacts) against
    sequential interval index.  A horizontal dashed line marks the overall
    mean of each quantity.  The shut-period panel uses a log y-axis to
    accommodate the wide dynamic range typical of single-channel recordings.

    Parameters
    ----------
    open_intervals:
        Open period durations in recording order (seconds).
    shut_intervals:
        Shut period durations in recording order (seconds).
    window:
        Number of intervals in each rolling window.  Default 100.
    figsize:
        Figure size passed to ``plt.subplots``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    opens = np.asarray(open_intervals, dtype=float)
    shuts = np.asarray(shut_intervals, dtype=float)

    rm_open = utils.rolling_mean(opens, window) * 1e3   # ms
    rm_shut = utils.rolling_mean(shuts, window) * 1e3   # ms

    # Popen: pair open[i] with shut[i] (use the shorter of the two arrays)
    n_p   = min(len(opens), len(shuts))
    op_p  = opens[:n_p]
    sh_p  = shuts[:n_p]
    total = utils.rolling_mean(op_p + sh_p, window)
    rm_po = utils.rolling_mean(op_p, window) / total

    # x positions: index of the centre interval for each window
    open_mid = np.arange(window // 2, window // 2 + len(rm_open))
    shut_mid = np.arange(window // 2, window // 2 + len(rm_shut))
    po_mid   = np.arange(window // 2, window // 2 + len(rm_po))

    mean_open_ms = float(np.mean(opens)) * 1e3
    mean_shut_ms = float(np.mean(shuts)) * 1e3
    mean_popen   = float(np.sum(op_p)) / float(np.sum(op_p + sh_p))
    x_max = max(len(opens), len(shuts))

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    ax0 = axes[0]
    ax0.plot(open_mid, rm_open, color="steelblue", lw=0.9, alpha=0.85)
    ax0.axhline(mean_open_ms, color="steelblue", lw=1.2, ls="--", alpha=0.6,
                label=f"mean = {mean_open_ms:.3f} ms")
    ax0.set_ylabel("Mean open period (ms)")
    ax0.set_title(f"Rolling mean open period  (window = {window} intervals)")
    ax0.legend(fontsize=9, loc="upper right")
    ax0.set_xlim(0, x_max)
    ax0.grid(axis="x", lw=0.4, alpha=0.4)

    ax1 = axes[1]
    ax1.plot(shut_mid, rm_shut, color="tomato", lw=0.9, alpha=0.85)
    ax1.axhline(mean_shut_ms, color="tomato", lw=1.2, ls="--", alpha=0.6,
                label=f"mean = {mean_shut_ms:.0f} ms")
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean shut period (ms, log)")
    ax1.set_title(f"Rolling mean shut period  (window = {window} intervals)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(0, x_max)
    ax1.grid(axis="x", lw=0.4, alpha=0.4)

    ax2 = axes[2]
    ax2.plot(po_mid, rm_po, color="seagreen", lw=0.9, alpha=0.85)
    ax2.axhline(mean_popen, color="seagreen", lw=1.2, ls="--", alpha=0.6,
                label=f"mean = {mean_popen:.5f}")
    ax2.set_ylabel("Rolling Popen")
    ax2.set_xlabel("Interval index (recording order)")
    ax2.set_title(f"Rolling Popen  (window = {window} intervals)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_xlim(0, x_max)
    ax2.grid(axis="x", lw=0.4, alpha=0.4)

    return fig


# =========================================================================== #
# Correlation plots                                                              #
# =========================================================================== #

_DEFAULT_CORR_COLORS = ["steelblue", "tomato", "seagreen", "darkorange"]


def plot_serial_correlation(
    *results,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Plot serial autocorrelation coefficients versus lag.

    Accepts one or two :class:`~ekdist.correlation.SerialCorrelationResult`
    objects and produces a figure with one panel per result.  Each panel shows
    the pooled r values as scatter points, a grey significance band
    (±1.96 / √df, pointwise), and filled markers for significant lags.

    Parameters
    ----------
    *results:
        One or two :class:`~ekdist.correlation.SerialCorrelationResult`
        instances returned by :func:`~ekdist.correlation.serial_correlation`.
    labels:
        Axis title for each panel.  Defaults to ``["Series 1", "Series 2"]``.
    colors:
        Marker colour for each panel.
    figsize:
        Figure size.  Defaults to ``(6, 4)`` for one panel or ``(11, 4)``
        for two.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not results:
        raise ValueError("At least one SerialCorrelationResult must be supplied")

    n = len(results)
    if labels is None:
        labels = [f"Series {i+1}" for i in range(n)]
    if colors is None:
        colors = _DEFAULT_CORR_COLORS[:n]
    if figsize is None:
        figsize = (6, 4) if n == 1 else (11, 4)

    fig, axes_raw = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes_raw[0]

    for ax, sc, label, color in zip(axes, results, labels, colors):
        lags = sc.lags
        r    = sc.r
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.where(sc.df > 0, 1.96 / np.sqrt(sc.df.astype(float)), np.nan)

        ax.axhline(0, color="k", lw=0.8, ls="--", zorder=1)
        ax.fill_between(lags, -se, se, alpha=0.15, color="grey",
                        label="±1.96 / √df (pointwise)")

        sig_mask = np.abs(r) > se
        ax.scatter(lags[~sig_mask], r[~sig_mask],
                   s=50, color=color, alpha=0.6, zorder=3, label="not significant")
        ax.scatter(lags[sig_mask], r[sig_mask],
                   s=70, color=color, zorder=4, edgecolors="k", linewidths=0.8,
                   label="p < 0.05")

        ax.set_xlabel("Lag k")
        ax.set_ylabel("Pooled rₖ")
        ax.set_title(f"Serial autocorrelation — {label}")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0.5, len(lags) + 0.5)

    return fig


def plot_open_shut_scatter(
    open_intervals: np.ndarray,
    shut_intervals: np.ndarray,
    *,
    result=None,
    figsize: tuple = (11, 4),
) -> plt.Figure:
    """Open-interval vs following-shut-interval scatter and 2-D density histogram.

    Left panel: log-log scatter of open (x) versus shut (y) durations.
    Right panel: 2-D histogram on log₁₀ axes with logarithmic colour scale.

    Parameters
    ----------
    open_intervals:
        Open period durations (seconds).
    shut_intervals:
        Shut period durations (seconds), same length as *open_intervals*.
    result:
        Optional :class:`~ekdist.correlation.CorrelationResult` from
        :func:`~ekdist.correlation.open_shut_correlation`.  When supplied,
        the r value and p-value are shown in the plot titles.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    op = np.asarray(open_intervals, dtype=float)
    sh = np.asarray(shut_intervals, dtype=float)

    r_str = ""
    if result is not None:
        p_str = f"p = {result.p_value:.3f}" if result.p_value >= 0.001 else "p < 0.001"
        r_str = f"  (r = {result.r:.3f},  dev = {result.z:.2f},  {p_str})"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: log-log scatter ---
    scatter_title = f"Open-shut scatter"
    if result is not None:
        scatter_title += f"  r = {result.r:.3f}"
    ax1.scatter(op * 1e3, sh * 1e3, s=3, alpha=0.25, color="steelblue", rasterized=True)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Open interval (ms)")
    ax1.set_ylabel("Following shut interval (ms)")
    ax1.set_title(scatter_title)

    # --- Right: 2-D histogram (log₁₀ bins) ---
    log_op = np.log10(op * 1e3)
    log_sh = np.log10(sh * 1e3)
    h, xedges, yedges = np.histogram2d(log_op, log_sh, bins=40)
    im = ax2.pcolormesh(
        xedges, yedges, h.T,
        norm=_mcolors.LogNorm(vmin=1),
        cmap="Blues",
    )
    fig.colorbar(im, ax=ax2, label="count")
    ax2.set_xlabel("log₁₀ open (ms)")
    ax2.set_ylabel("log₁₀ following shut (ms)")
    density_title = "Open-shut joint distribution"
    if result is not None:
        density_title += f"\n{r_str.strip()}"
    ax2.set_title(density_title, fontsize=10)

    return fig

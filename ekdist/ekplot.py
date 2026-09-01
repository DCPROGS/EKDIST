import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib import ticker

from dcio.analysis.histogram import (
    bins_per_decade, log_bin_histogram, staircase,
)

from ekdist import eklib
from ekdist import exponentials

def stability_intervals(rec, open=True, shut=True, popen=True, window=50):
    opma, shma, poma = eklib.moving_average_open_shut_Popen(
                       rec.periods.get_open_intervals()[:-1], 
                       rec.periods.get_shut_intervals(), 
                       window=window)
    x = np.linspace(0, np.prod(opma.shape), num=np.prod(opma.shape), endpoint=True)
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    if open:
        ax.semilogy(x, opma, 'g', label='Open periods')
    if shut:
        ax.semilogy(x, shma, 'r', label='Shut periods')
    if popen:
        ax.semilogy(x, poma, 'b', label='Popen')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3,
                        borderaxespad=0.)
    ax.set_xlabel('Interval number')
    return fig
    
def stability_amplitudes(rec, window=1):
    all_resolved_opamp = np.array(rec.rampl)[np.where( np.fabs(np.asarray(rec.rampl)) > 0.0)]
    amps = eklib.moving_average(all_resolved_opamp, window)
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.plot(amps, '.g')
    #ax.set_ylim([0, 1.2 * max(amps)])
    ax.set_ylabel('Amplitude, pA')
    ax.set_xlabel('Interval number')
    print('Average open amplitude = ', np.average(amps))
    return fig

def histogram_fitted_amplitudes(rec, fc, n=2, nbins=20, gauss=True):
    long_opamp = eklib.amplitudes_openings_longer_Tr(rec, fc, n)
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    ax.hist(long_opamp, nbins, density=True, alpha=0.6, color='g')
    if gauss:
        mu, std = norm.fit(long_opamp)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        ax.set_title("Fit results: mu = %.2f,  std = %.2f" % (mu, std))
    ax.set_xlim([0, 1.2 * max(long_opamp)])
    ax.set_xlabel('Amplitude, pA')
    ax.set_ylabel('Frequency')
    print('Range of amplitudes: {0:.3f} - {1:.3f}'.
          format(min(long_opamp), max(long_opamp)))
    return fig   

def burst_number_of_openings(nops):
    """ nops- list of number of openings  """
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    bins = np.arange(0., max(nops)+1, 1)
    ax.hist(nops, bins, histtype='step')
    ax.set_xlim([0, 1 + max(bins)])
    ax.set_xlabel('# of openings / burst')
    ax.set_ylabel('Frequency')
    return fig

###############################################################################
# Dwell time histograms: x-log / y-sqrt
def __histogram_bins_per_decade(X):
    return bins_per_decade(len(X))

def __bin_width(X):
    return 10.0 ** (1.0 / bins_per_decade(len(X)))

def __exponential_scale_factor(X, pdf, tres):
    """Normalise a fitted exponential mixture onto the histogram.

    Not the same quantity as HJCFIT's ideal_pdf_scale_factor, which
    renormalises an ideal pdf onto the resolved intervals from the Q matrix.
    This one scales a mixture already fitted to these data, so it stays here.
    """
    return (len(X) * math.log10(__bin_width(X)) * math.log(10) *
            (1 / np.sum(pdf.area * np.exp(-tres / pdf.tau))))

def prepare_xlog_hist(X, tres):
    """ Prepare x-log histogram.

    Binning is dcio.analysis.histogram; this is the staircase it produces.

    The bin edges used to be built here, and the upper limit was computed as
    exp(ceil(log(max(X)))) -- a power of e, where the whole scheme is bins per
    decade. That limit can fall below the longest interval, and np.histogram
    then drops the tail of the distribution silently: across 400 randomly drawn
    exponential samples it happened in 119 of them.

    Parameters
    ----------
    X :  1-D array or sequence of scalar
    tres : float
        Temporal resolution, shortest resolvable time interval. It is
        histogram's starting point.

    Returns
    -------
    xout, yout :  list of scalar
        x and y values to plot histogram.
    """
    counts, edges, _ = log_bin_histogram(X, tres)
    xout, yout = staircase(edges, counts)
    return list(xout), list(yout)

def histogram_xlog_ysqrt_data(X, tres, pdf=None, tcrit=None, xlabel='Dwell times'):
    """ Plot dwell time histogram in log x and square root y. """
    xout, yout= prepare_xlog_hist(X, tres)
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.semilogx(xout, np.sqrt(yout))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('sqrt(frequency)')
    t = np.logspace(math.log10(tres), math.log10(2 * max(X)), 512)
    if pdf is None:
        pdf = exponentials.ExponentialPDF([np.mean(X)], [1.0])
    scale = __exponential_scale_factor(X, pdf, tres)
    ax.plot(t, np.sqrt(scale * t * pdf.exp(pdf.theta, t)), '-b')
    for ta, ar in zip(pdf.tau, pdf.area):
        ax.plot(t, np.sqrt(scale * t * (ar / ta) * np.exp(-t / ta)), '--b')

    if tcrit is not None:
        for tc in np.asarray(tcrit):
            ax.axvline(x=tc, color='g')

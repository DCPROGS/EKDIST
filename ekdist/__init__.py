"""ekdist — single-channel dwell-time distribution analysis.

Public API
----------

    from ekdist.record import SingleChannelRecord, Periods
    from ekdist.bursts import Bursts
    from ekdist.fitting import ExponentialPDF, GaussianMixturePDF
    from ekdist.tcrit import Tcrit
    from ekdist.errors import ApproximateSD, LikelihoodIntervals
    from ekdist.stationarity import runs_test, cox_lewis_test
    from ekdist.correlation import serial_correlation, open_shut_correlation
    from ekdist.popen import global_popen, burst_popen
    from ekdist import plots

Typical workflow::

    rec = SingleChannelRecord.from_scn("my_record.scn")
    rec.tres = 30e-6            # impose 30 µs resolution

    opens = rec.periods.open_intervals
    print(runs_test(opens))
    print(cox_lewis_test(opens))

    pdf = ExponentialPDF([1e-3, 10e-3])
    pdf.fit(opens, tres=rec.tres)
    pdf.get_tcrits()

    bursts = Bursts.from_periods(rec.periods, tcrit=pdf.tcrits['DC'][0])
    print(global_popen(rec))
    print(burst_popen(bursts))

    fig = plots.histogram_dwell_times(opens, tres=rec.tres, pdf=pdf)
    fig.savefig("open_dwell_times.pdf")
"""

__version__ = "1.0.0"

from ekdist.record import SingleChannelRecord, Periods
from ekdist.bursts import Bursts
from ekdist.fitting import ExponentialPDF, GaussianMixturePDF
from ekdist.tcrit import Tcrit
from ekdist.errors import ApproximateSD, LikelihoodIntervals
from ekdist.stationarity import runs_test, cox_lewis_test, StationarityResult
from ekdist.correlation import (
    serial_correlation,
    open_shut_correlation,
    CorrelationResult,
    SerialCorrelationResult,
)
from ekdist.popen import global_popen, burst_popen, PopenResult

__all__ = [
    "SingleChannelRecord",
    "Periods",
    "Bursts",
    "ExponentialPDF",
    "GaussianMixturePDF",
    "Tcrit",
    "ApproximateSD",
    "LikelihoodIntervals",
    "runs_test",
    "cox_lewis_test",
    "StationarityResult",
    "serial_correlation",
    "open_shut_correlation",
    "CorrelationResult",
    "SerialCorrelationResult",
    "global_popen",
    "burst_popen",
    "PopenResult",
]

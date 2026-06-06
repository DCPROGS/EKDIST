# ekdist — Single-Channel Dwell-Time Analysis

Python rewrite of the DCProgs **EKDIST** Fortran program (Colquhoun lab).

Idealising a single-channel patch-clamp recording produces a list of interval
durations and amplitudes — one entry per opening or closing event.  `ekdist`
takes that list and provides the complete downstream analysis chain:

* Dead-time (temporal resolution) imposition
* Amplitude histogram with Gaussian mixture fitting
* Temporal stationarity testing before fitting
* Maximum-likelihood fitting of exponential dwell-time distributions
* Critical-time (*tcrit*) calculation for burst definition
* Burst grouping and within-burst open-probability estimation
* Global Popen with bootstrap confidence intervals
* Serial autocorrelation and open-shut interval correlation
* A Streamlit interactive app mirroring the analysis notebook


---

## Table of contents

1. [Installation](#installation)
2. [Usage modes](#usage-modes)
   - [Python API](#python-api)
   - [Jupyter notebooks](#jupyter-notebooks)
   - [Streamlit interactive app](#streamlit-interactive-app)
3. [Module architecture](#module-architecture)
4. [Algorithms and design choices](#algorithms-and-design-choices)
5. [API reference](#api-reference)
6. [Running the tests](#running-the-tests)
7. [Conversion status, TODO list, and known bugs](#conversion-status-todo-list-and-known-bugs)

---

## Installation

```bash
# From the EKDIST root (editable install — source changes take effect immediately)
pip install -e .

# Or with all developer extras (pytest, ruff, mypy, jupyterlab):
pip install -e ".[dev]"
```

The `dcprogs` conda environment already has `ekdist` installed in editable mode
and includes all runtime and development dependencies.

**Runtime requirements** (from `pyproject.toml`):

| Package | Minimum version |
|---------|----------------|
| Python  | 3.11 |
| numpy   | 2.0 |
| scipy   | 1.13 |
| matplotlib | 3.9 |
| dcio    | 0.1 (internal DCProgs package) |

---

## Usage modes

### Python API

The typical analysis sequence from a script or notebook:

```python
from ekdist import (
    SingleChannelRecord,
    ExponentialPDF, GaussianMixturePDF,
    Bursts,
    runs_test, cox_lewis_test,
    serial_correlation, open_shut_correlation,
    global_popen, burst_popen,
    plots,
)

# 1. Load and impose dead-time resolution
rec = SingleChannelRecord.from_scn("path/to/my_record.scn")
rec.tres = 30e-6          # 30 µs temporal resolution

# 2. Visualise stationarity
fig = plots.stability_rolling(
    rec.periods.open_intervals,
    rec.periods.shut_intervals,
    window=100,
)
fig.savefig("stationarity.pdf")

# 3. Stationarity tests (before fitting)
opens = rec.periods.open_intervals
print(runs_test(opens))
print(cox_lewis_test(opens))

# 4. Fit open-time distribution
pdf_open = ExponentialPDF([1e-3, 10e-3])   # two components, initial guesses
pdf_open.fit(opens, tres=rec.tres)
pdf_open.get_tcrits(verbose=True)          # prints DC / C&N / Jackson table
print(pdf_open.summary(opens))

# 5. Fit shut-time distribution and find critical times
shuts = rec.periods.shut_intervals
pdf_shut = ExponentialPDF([1e-3, 10e-3, 100e-3])
pdf_shut.fit(shuts, tres=rec.tres)
pdf_shut.get_tcrits()

# Use the Jackson tcrit between the last two components to define bursts
tcrit = pdf_shut.tcrits["Jackson"][-1]

# 6. Group into bursts
bursts = Bursts.from_periods(rec.periods, tcrit=tcrit)
print(bursts)

# 7. Popen
print(global_popen(rec, n_bootstrap=1000))
print(burst_popen(bursts, n_bootstrap=1000))

# 8. Correlation
sc_open = serial_correlation(opens, max_lag=20)
print(sc_open.summary())
osc = open_shut_correlation(opens, rec.periods.shut_intervals[1:-1])
print(osc)

# 9. Amplitude distribution (experimental records only)
amps_pdf = GaussianMixturePDF([10.0, 20.0], [1.0, 1.5])
amps_pdf.fit(abs(rec.rampl[rec.rampl != 0]))
fig = plots.histogram_amplitudes(rec, fc=3000.0, pdf=amps_pdf)
fig.savefig("amplitudes.pdf")
```

Loading multiple SCN files into a single concatenated record:

```python
rec = SingleChannelRecord.from_scn_files(["file1.scn", "file2.scn", "file3.scn"])
rec.tres = 30e-6
```

Loading from pre-built interval arrays (useful for simulated data):

```python
import numpy as np
rec = SingleChannelRecord.from_intervals(itint, iampl, iprop, is_simulated=True)
rec.tres = 30e-6
```

---

### Jupyter notebooks

Notebooks live in the `notebooks/` directory:

| Notebook | Purpose |
|----------|---------|
| `EKDIST_example.ipynb` | Complete worked example on AChR simulated record |
| `Fit_intervals.ipynb`  | Focused demo of `ExponentialPDF` fitting and tcrit calculation |

Launch:

```bash
# activate dcprogs env first
conda activate dcprogs
cd E:\dcprogs\EKDIST
jupyter lab
```

---

### Streamlit interactive app

`EKDIST-app.bat` (double-click from Explorer, or run from any terminal):

```
E:\dcprogs\EKDIST\EKDIST-app.bat
```

The app opens at **http://localhost:8501** and provides:

| Tab | Content |
|-----|---------|
| 📁 Record | File upload, `tres` parameter, record summary statistics |
| 📈 Stationarity | Stability plots; runs test and Cox-Lewis test results |
| 📡 Amplitudes | Amplitude stability scatter; amplitude histogram with Gaussian mixture fit |
| 🔓 Open distribution | MLE fit; parameter table; histogram with PDF overlay |
| 🔒 Shut distribution | MLE fit; tcrit table (DC / C&N / Jackson); histogram with coloured tcrit lines |
| ⚡ Bursts | tcrit selector (editable); burst summary; Popen; burst-opening histogram |
| 🎯 Popen | Global Popen with bootstrap CI; burst Popen; comparison table |
| 🔗 Correlation | Serial autocorrelation (open and shut); open-shut scatter and 2-D density |

**Caching strategy**: fits are cached in `st.session_state` keyed by
`"{filename}::{tres_us}"`.  Changing the file or `tres` invalidates the entire
chain; changing a plot parameter (bin count, lag, etc.) does not refit.
Expensive calculations (MLE, bootstrap) only run when you click **Fit** or
**Compute**.

**Simulated-record detection**: the Amplitudes tab is deactivated when all open
amplitudes are identical within floating-point tolerance (`std < 1e-10`).
This is the case for records generated by SCALCS/SCSIM simulations.

---

## Module architecture

```
ekdist/
├── __init__.py          Public API re-exports; version string
├── _constants.py        All magic numbers with documented origins
├── io.py                Thin adapter: ekdist.io.read_scn → dcio SCN reader
├── record.py            SingleChannelRecord, Periods — data model + resolution
├── resolution.py        impose_resolution() — pure function, independently testable
├── fitting.py           ExponentialPDF, GaussianMixturePDF — MLE fitters
├── tcrit.py             Tcrit — DC / C&N / Jackson critical-time calculation
├── errors.py            ApproximateSD, LikelihoodIntervals — parameter uncertainty
├── stationarity.py      runs_test, cox_lewis_test — stationarity tests
├── popen.py             global_popen, burst_popen, PopenResult
├── correlation.py       serial_correlation, open_shut_correlation
├── bursts.py            Bursts — grouping and per-burst metrics
├── plots.py             All visualisation functions
└── utils.py             moving_average, rolling_mean, filter rise-time helpers

app.py                   Streamlit application (8 tabs)
notebooks/               Jupyter notebooks
tests/                   pytest test suite (16 test files, ~150 tests)
EKDIST-app.bat           Windows launcher for the Streamlit app
```

### Data flow

```
SCN file
    ↓  ekdist.io.read_scn (→ dcio)
    ↓  SingleChannelRecord.from_scn
    │    .itint / .iampl / .iprop   (raw intervals)
    ↓  rec.tres = value → impose_resolution()
    │    .rtint / .rampl / .rprop   (resolved intervals)
    ↓  Periods(rtint, rampl, rprop)
    │    .open_intervals             (open period durations)
    │    .shut_intervals             (shut period durations)
    ↓
  ┌─────────────────────────────────────────────┐
  │ Analysis branch A: amplitude                 │
  │   GaussianMixturePDF.fit(|rampl[rampl≠0]|)  │
  │   plots.histogram_amplitudes(rec, fc)        │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ Analysis branch B: dwell-time distributions  │
  │   ExponentialPDF.fit(open_intervals)         │
  │   ExponentialPDF.fit(shut_intervals)         │
  │     → tcrit via Tcrit (DC / C&N / Jackson)  │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ Analysis branch C: bursts                    │
  │   Bursts.from_periods(periods, tcrit)        │
  │   burst_popen(bursts)                        │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ Analysis branch D: Popen                     │
  │   global_popen(rec)                          │
  │   burst_popen(bursts)                        │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ Analysis branch E: correlation               │
  │   serial_correlation(opens)                  │
  │   serial_correlation(shuts)                  │
  │   open_shut_correlation(opens, shuts)        │
  └─────────────────────────────────────────────┘
```

---

## Algorithms and design choices

### Dead-time (temporal resolution) imposition

**Source**: `ekdist/resolution.py` → `impose_resolution()`

**Algorithm** (faithfully ported from Fortran EKDIST):

1. Scan the raw interval list to find the first resolvable AND usable interval
   (duration > `tres` and `iprop < BAD_FLAG`).
2. Accumulate consecutive intervals of the same type (open or shut):
   - **Unresolvable intervals** (`t < tres`): merged into the current accumulator;
     their amplitude is included in the time-weighted mean if open.
   - **Resolvable shut after an open**: close the open period, start a new shut.
   - **Resolvable open after a shut**: close the shut period, start a new open.
   - **Resolvable open after an open** (sublevel change): only start a new open
     period if the running time-weighted amplitude differs by more than
     `AMPLITUDE_TOLERANCE = 1e-5 pA` (experimental records).  For simulated
     records, successive opens at the same conductance are always merged.
3. Trim leading shut periods and trailing bad/incomplete intervals.
4. The last accumulated open period gets duration `-1.0` (sentinel for
   "truncated / incomplete last opening").  It is removed in the trim step so
   the resolved array always starts and ends with a complete open period.

**Known bug (Bug 7)**: the `-1.0` sentinel for the truncated last opening is
an undocumented implementation detail inherited from the Fortran original.
Any code that processes `rtint` without checking for `-1.0` entries may
silently mis-report the last open duration.  The trim step in
`_trim_trailing_bad_or_shut` removes it before it reaches user code, so this
is only a concern if `impose_resolution()` is called directly.

**Design choice**: `impose_resolution` is a pure function (no class, no side
effects) so it can be tested in complete isolation and reused by other
packages (e.g., SCALCS).

---

### Exponential mixture MLE

**Source**: `ekdist/fitting.py` → `ExponentialPDF`

**Model**: a *k*-component exponential mixture

    f(t) = Σᵢ (aᵢ/τᵢ) · exp(−t/τᵢ)

subject to Σaᵢ = 1.  The last area is always derived from the others so the
optimiser works with *k* free tau values and *k−1* free area values.

**Likelihood**: truncated to the observable range `[tres, max(X)]`:

    LL(θ) = Σⱼ log f(tⱼ) − n · log Σᵢ aᵢ[exp(−tres/τᵢ) − exp(−tmax/τᵢ)]

This is the correct truncated MLE.  Using `min(X)` instead of `tres` as the
lower bound (which the original Fortran does) is statistically biased; the
`tres` keyword argument to `.fit()` fixes this.

**Optimiser**: Nelder-Mead simplex via `scipy.optimize.minimize`.  No gradient
needed; Nelder-Mead is robust to the discontinuous penalty returned when
`tau <= 0` or `area < 0` is probed.

**Initial guesses**: supplied by the user.  The Streamlit app uses
logarithmically spaced values between `min(X)` and `max(X)`.
A heuristic auto-guesser based on log-histogram peaks is a **planned TODO**.

---

### Critical time (*tcrit*) — burst definition

**Source**: `ekdist/tcrit.py` → `Tcrit`

For a k-component shut-time distribution with `k ≥ 2` there are `k−1`
adjacent-component pairs.  For each pair `(i, i+1)` a tcrit is found by
bisection between `τᵢ` and `τᵢ₊₁` using one of three criteria:

| Criterion | Condition solved | Reference |
|-----------|-----------------|-----------|
| **DC** | Fraction misclassified fast = fraction misclassified slow | Colquhoun & Hawkes |
| **C&N** | Number misclassified fast = number misclassified slow | Clapham & Neher |
| **Jackson** | d(total misclassified)/dt = 0; i.e., rate of fast tail = rate of slow head | Jackson et al. |

`ExponentialPDF.get_tcrits()` calls `Tcrit` internally and stores the result as
`pdf.tcrits`, a dict `{"DC": [...], "C&N": [...], "Jackson": [...]}` with one
entry per adjacent pair.

**Known bug (Bug 9)**: `Tcrit` assumes `tau` is sorted ascending on input.
If tau values are in wrong order after fitting (can happen with Nelder-Mead),
bisection may fail silently (returns `None` for that pair).  A guard that
sorts `tau` and `area` together before bisection is a planned fix.

---

### Stationarity tests

**Source**: `ekdist/stationarity.py`

Both tests should be run on raw interval sequences *before* fitting.
Non-stationarity invalidates MLE assumptions.

**Wald-Wolfowitz runs test** (`runs_test`):
Counts maximal runs of consecutive values above/below the sample median.
Under H₀ (stationarity), the run count is approximately Normal.
- Z ≪ 0 → too few runs → systematic drift or rundown
- Z ≫ 0 → too many runs → oscillation between states
- p-value is two-tailed; α = 0.05 is the conventional threshold.

**Cox-Lewis trend test** (`cox_lewis_test`):
Normalised event-time statistic U = mean(t_k / t_n) for k = 1 … n−1.
Under H₀, U ~ Uniform(0, 1); Z = (U − 0.5)√(12(n−1)) ~ N(0,1).
- Z < 0 → intervals getting longer over time (declining channel activity)
- Z > 0 → intervals getting shorter (increasing activity / potentiation)

---

### Serial autocorrelation

**Source**: `ekdist/correlation.py` → `serial_correlation`

Ports the 1995 revision of `CORCAL.FOR`.  For each lag k (1 … max_lag):

1. Compute a global mean `ȳ` and total sum-of-squares `SSY` from all non-outlier
   values (fixed denominator, matching Fortran).
2. Scan pairs `(X[i], X[i+k])`.  When a value exceeds `outlier_limit`, close
   the current sub-segment, record its Fisher Z-transform, and restart.
3. Pool per-segment Fisher Z values by a df-weighted mean:
   `z̄ = Σ(dfⱼ · zⱼ) / Σdfⱼ`; normal deviate = z̄ · √(Σdfⱼ).

This gives `SerialCorrelationResult` with pooled `r` and normal deviate at each
lag.  A significant deviate at lag k indicates that intervals k apart are
correlated — a sign of multi-state kinetics or non-stationarity.

---

### Parameter uncertainty

**Source**: `ekdist/errors.py`

* **`ApproximateSD`**: estimates the Hessian of the negative log-likelihood at
  the MLE by central finite differences, then inverts it to get the covariance
  matrix.  Step sizes are auto-tuned so the LL changes by ≈ 0.5% at the
  perturbed point.  SD = √(diag of covariance).

* **`LikelihoodIntervals`**: profile likelihood intervals.  For each free
  parameter, holds all others at their MLE and bisects to find where
  ΔLL = m (default m = 2, corresponding to ≈ 2 SD for large samples).

Both classes work on any PDF object that exposes `LL(theta, data)` and the
`fixed` / `pars` attributes (same interface as `ExponentialPDF`).

---

### Popen with bootstrap CI

**Source**: `ekdist/popen.py`

**Global Popen**: `Σ(open durations) / Σ(all durations)`, with CI from
non-parametric bootstrap — open and shut duration arrays resampled
independently with replacement.

**Burst Popen**: mean of per-burst Popen values (open time excluding the last
opening, divided by total burst time), with CI from resampling the per-burst
Popen values.

---

## API reference

### `ekdist.record`

#### `SingleChannelRecord`

```python
# Constructors
rec = SingleChannelRecord.from_scn(path, *, verbose=False)
rec = SingleChannelRecord.from_scn_files([path1, path2, ...], *, verbose=False)
rec = SingleChannelRecord.from_intervals(itint, iampl, iprop=None, *, is_simulated=False)

# Properties
rec.tres                # float — temporal resolution (s); setting it reruns imposition
rec.badopen             # float — openings longer than this (s) are flagged bad; -1 = disabled
rec.itint               # np.ndarray — raw interval durations (s)
rec.iampl               # np.ndarray — raw amplitudes (pA; 0 = shut)
rec.iprop               # np.ndarray(int) — raw property flags
rec.rtint               # np.ndarray — resolved interval durations (s)
rec.rampl               # np.ndarray — resolved amplitudes (pA)
rec.rprop               # np.ndarray(int) — resolved property flags
rec.periods             # Periods — grouped open/shut periods
rec.header              # SCNHeader | None — file header metadata
```

#### `Periods`

```python
periods.open_intervals            # np.ndarray — open period durations (s)
periods.shut_intervals            # np.ndarray — shut period durations (s)
periods.open_intervals_in_range(lo, hi)   # filtered open periods
periods.shut_intervals_in_range(lo, hi)   # filtered shut periods
periods.intervals                 # list[float] — all periods alternating open/shut
periods.amplitudes                # list[float] — time-weighted mean amplitudes
periods.flags                     # list[int]
```

---

### `ekdist.fitting`

#### `ExponentialPDF(tau, area=None)`

```python
pdf = ExponentialPDF([1e-3, 10e-3])          # 2 components
pdf = ExponentialPDF([1e-3, 10e-3], [0.6])   # explicit initial area

pdf.theta              # np.ndarray — free parameter vector [tau..., area... (last derived)]
pdf.tau                # np.ndarray — current time constants (s)
pdf.area               # np.ndarray — current component areas (sum to 1)
pdf.ncomps             # int — number of components
pdf.tcrits             # dict[str, list[float|None]] — populated by get_tcrits()

pdf.fit(X, tres=0.0)   # Nelder-Mead MLE; updates tau/area in-place; returns OptimizeResult
pdf.pdf(theta, X)      # evaluate density at X for parameter vector theta
pdf.LL(theta, X)       # negative log-likelihood (minimise this)
pdf.mean()             # float — overall mean (s)
pdf.predicted_counts(X)# (en, enout) — counts per component and outside range
pdf.summary(X)         # str — formatted fit summary
pdf.get_tcrits(verbose=False)  # populate self.tcrits; print table if verbose
```

#### `GaussianMixturePDF(means, sigmas, areas=None)`

```python
pdf = GaussianMixturePDF([10.0], [1.5])          # 1 component
pdf = GaussianMixturePDF([8.0, 14.0], [1.0, 1.2])  # 2 components

pdf.means              # np.ndarray — component means (pA)
pdf.sigmas             # np.ndarray — component standard deviations (pA)
pdf.areas              # np.ndarray — component areas (sum to 1)
pdf.ncomps             # int

pdf.fit(X)             # Nelder-Mead MLE; updates means/sigmas/areas in-place
pdf.pdf(theta, X)      # evaluate density
pdf.LL(theta, X)       # negative log-likelihood
pdf.mean()             # float — overall mean amplitude (pA)
pdf.summary()          # str — formatted parameter summary
```

---

### `ekdist.tcrit`

#### `Tcrit(tau, area)`

```python
tc = Tcrit(tau, area)

tc.tcrits              # dict[str, list[float|None]]  keys: "DC", "C&N", "Jackson"
tc.misclassified(tcrit, comp)   # (enf, ens, pf, ps) — misclassification at tcrit
tc.summary()           # str — formatted table of all tcrit values
tc.misclassified_summary(criterion)  # str — full misclassification report
```

Accessed indirectly via `ExponentialPDF.get_tcrits()`.

---

### `ekdist.bursts`

#### `Bursts`

```python
bursts = Bursts.from_periods(periods, tcrit)   # primary constructor

len(bursts)                   # number of bursts
bursts.tcrit                  # float — tcrit used (s)

# Per-burst metrics (take a burst array)
bursts.n_openings(burst)
bursts.total_open_time(burst)
bursts.total_shut_time(burst)
bursts.total_duration(burst)
bursts.popen(burst)           # including last opening
bursts.popen_excl_last(burst) # excluding last opening (standard)
bursts.mean_open_time(burst)
bursts.mean_shut_time(burst)
bursts.longest_opening(burst)
bursts.longest_shut(burst)

# List methods (return list over all bursts)
bursts.list_popen()           # Popen excl. last, per burst
bursts.list_duration()
bursts.list_n_openings()
bursts.list_total_open_time()
bursts.list_total_shut_time()
bursts.list_mean_open_time()
bursts.list_mean_shut_time()
bursts.list_longest_opening()
bursts.list_longest_shut()
bursts.bursts_with_n_openings(n)

# Aggregate statistics
bursts.mean_popen()
bursts.mean_n_openings()
bursts.mean_duration()
bursts.mean_total_open_time()
bursts.mean_total_shut_time()
bursts.mean_mean_open_time()

# Filtering
bursts.filter_min_openings(n)
bursts.filter_max_open_time(max_open)
bursts.filter_min_duration(min_dur)
```

---

### `ekdist.popen`

```python
result = global_popen(rec, *, n_bootstrap=1000, confidence=0.95, rng=None)
result = burst_popen(bursts, *, n_bootstrap=1000, confidence=0.95, rng=None)

# PopenResult fields
result.popen             # float — point estimate
result.ci_lower          # float — lower bootstrap CI bound
result.ci_upper          # float — upper bootstrap CI bound
result.n_bootstrap       # int
result.confidence        # float — e.g. 0.95
result.total_open_time   # float — total open time (s) [NaN for burst_popen]
result.total_time        # float — total time (s) [NaN for burst_popen]
```

---

### `ekdist.stationarity`

```python
result = runs_test(X)
result = cox_lewis_test(X)

# StationarityResult fields
result.statistic    # float — Z-score
result.p_value      # float — two-tailed p-value
result.n            # int — sample size
result.name         # str — test name
result.description  # str — plain-English verdict
```

---

### `ekdist.correlation`

```python
sc = serial_correlation(intervals, max_lag=50, outlier_limit=None)
osc = open_shut_correlation(open_intervals, shut_intervals)

# SerialCorrelationResult fields
sc.lags             # np.ndarray[int]
sc.r                # np.ndarray — pooled correlation at each lag
sc.z                # np.ndarray — Fisher Z normal deviate at each lag
sc.df               # np.ndarray[int] — degrees of freedom at each lag
sc.n_segments       # np.ndarray[int] — number of sub-segments pooled
sc.n_total          # int — non-outlier intervals used
sc.outlier_limit    # float — (inf if not specified)
sc.summary()        # str — formatted table

# CorrelationResult fields
osc.r               # float — Pearson correlation coefficient
osc.z               # float — Fisher Z normal deviate
osc.df              # int — degrees of freedom (n - 3)
osc.n               # int — number of pairs
osc.p_value         # float — two-tailed p-value
osc.name            # str
osc.description     # str — plain-English verdict
```

---

### `ekdist.errors`

```python
from ekdist.errors import ApproximateSD, LikelihoodIntervals

# Approximate SDs from Hessian inversion
asd = ApproximateSD(pdf.theta, pdf.LL, data)
asd.sd             # np.ndarray — approx SD per free parameter
asd.hessian        # np.ndarray — estimated Hessian
asd.covariance     # np.ndarray — inverse Hessian
asd.correlations   # np.ndarray — parameter correlation matrix

# Profile-likelihood intervals
li = LikelihoodIntervals(pdf.theta, pdf, data, SD=asd.sd, m=2.0)
limits = li.calculate()  # np.ndarray shape (n_params, 2): [lower, upper] per parameter
```

---

### `ekdist.plots`

All functions return a `matplotlib.figure.Figure`.

```python
# Stability
plots.stability_intervals(rec, *, show_open, show_shut, show_popen, window, figsize)
plots.stability_amplitudes(rec, *, window, figsize)
plots.stability_rolling(open_intervals, shut_intervals, *, window, figsize)

# Distributions
plots.histogram_amplitudes(rec, fc, *, n_risetimes, nbins, pdf, figsize)
plots.histogram_dwell_times(X, tres, *, pdf, tcrits, xlabel, title, figsize)
plots.histogram_burst_openings(n_openings, *, figsize)

# Correlation
plots.plot_serial_correlation(*results, labels, colors, figsize)
plots.plot_open_shut_scatter(open_intervals, shut_intervals, *, result, figsize)

# Utility
plots.prepare_xlog_hist(X, tres)   # → (xout, yout) for semilogx plotting
```

**`histogram_amplitudes`**: filters openings shorter than `n_risetimes × rise_time`
where `rise_time = 0.3321 / fc`.  If no `pdf` is supplied, a single Gaussian
is auto-fitted and the mean/SD are shown in the title.  When a
`GaussianMixturePDF` is supplied, individual component curves are overlaid as
dashed lines.

**`histogram_dwell_times`**: log-x / sqrt-y axes (Colquhoun lab convention).
Bin width is automatically chosen based on sample size:
5 bins/decade for n ≤ 300, 8 for n ≤ 1000, 10 for n ≤ 3000, 12 otherwise.
The PDF is scaled by `n × log₁₀(bw) × ln(10) / P(t ≥ tres)` so the fitted
curve overlays the histogram without additional normalisation.

---

## Running the tests

```bash
conda activate dcprogs
cd E:\dcprogs\EKDIST
pytest -q
```

~150 tests across 16 test files, running in ≈ 35 s.

| Test file | What is tested |
|-----------|---------------|
| `test_record.py` | `SingleChannelRecord` constructors, `tres` setter, `Periods` |
| `test_resolution.py` | `impose_resolution` edge cases + numerical regression vs AChsim.scn |
| `test_io.py` | SCN file reading (version 9 format) |
| `test_fitting.py` | `ExponentialPDF`, `GaussianMixturePDF` — fit convergence, LL |
| `test_tcrit.py` | `Tcrit` DC/C&N/Jackson values; misclassification arithmetic |
| `test_bursts.py` | `Bursts.from_periods`, per-burst and aggregate metrics, filters |
| `test_popen.py` | `global_popen`, `burst_popen` — point estimates and bootstrap CI |
| `test_stationarity.py` | `runs_test`, `cox_lewis_test` — statistics and p-values |
| `test_correlation.py` | `serial_correlation`, `open_shut_correlation` |
| `test_errors.py` | `ApproximateSD`, `LikelihoodIntervals` |
| `test_plots.py` | All plot functions return `Figure`; amplitude histogram with real data |
| `test_ekdist_functional.py` | End-to-end: load → resolve → fit → burst → Popen |
| `test_ekdist_data_loading.py` | Loading both simulated and experimental SCN files |
| `test_stationarity.py` | Stationarity tests against known distributions |
| `test_utils.py` | `rolling_mean`, `moving_average`, rise-time helpers |
| `test_imports.py` | Smoke test: all public names importable |
| `test_errors.py` | Error classes on synthetic data |

Test data files in `tests/`:

| File | Description |
|------|-------------|
| `AChsim.scn` | Simulated acetylcholine receptor record (SCALCS/SCSIM output) |
| `181003S8.SCN` | Experimental single-channel record (real patch-clamp data) |
| `glyr_experimental.scn` | Glycine receptor experimental record with amplitude data |
| `intervals.txt` | 125 synthetic dwell times for unit testing |

---

## Conversion status, TODO list, and known bugs

### Fortran-to-Python conversion status

The original DCPROGS suite consists of several Fortran programs.  This Python
package covers the `EKDIST.FOR` analysis core plus the `CORCAL.FOR` correlation
module.

| Feature | Fortran source | Python status |
|---------|----------------|---------------|
| SCN file reading | `EKSCN.FOR` | ✅ Done (via `dcio`) |
| Dead-time resolution | `EKDIST.FOR` RESOL subroutine | ✅ Done (`resolution.py`) |
| Amplitude histogram + Gaussian fit | `EKDIST.FOR` AMPL subroutine | ✅ Done (`fitting.GaussianMixturePDF`, `plots.histogram_amplitudes`) |
| Dwell-time histogram (log-x / sqrt-y) | `EKDIST.FOR` DIST subroutine | ✅ Done (`plots.histogram_dwell_times`) |
| Exponential mixture MLE | `EKDIST.FOR` EKFIT subroutine | ✅ Done (`fitting.ExponentialPDF`) |
| Critical-time calculation | `EKDIST.FOR` CRIT subroutine | ✅ Done (`tcrit.Tcrit`) — all 3 criteria |
| Burst grouping | `EKDIST.FOR` BURST subroutine | ✅ Done (`bursts.Bursts`) |
| Stationarity tests | `EKDIST.FOR` STAT subroutine | ✅ Done (`stationarity`) |
| Global Popen | `EKDIST.FOR` POPEN subroutine | ✅ Done (`popen.global_popen`) |
| Bootstrap CI | Not in Fortran | ✅ Added (`popen` module) |
| Serial autocorrelation | `CORCAL.FOR` (IDTYPE 1–10) | ✅ Done (`correlation.serial_correlation`) |
| Open-shut correlation | `CORCAL.FOR` IDTYPE 11 | ✅ Done (`correlation.open_shut_correlation`) |
| Approximate SD (Hessian) | `EKLIB.FOR` APPROXSD | ✅ Done (`errors.ApproximateSD`) |
| Likelihood intervals | `EKLIB.FOR` LINTVALS | ✅ Done (`errors.LikelihoodIntervals`) |
| Streamlit interactive app | — | ✅ Done (`app.py`) |
| Sublevel analysis (`SEQLST.FOR`) | `SEQLST.FOR` | ❌ Not yet |
| Burst-mean fitted distribution (IDTYPE=15) | `EKDIST.FOR` | ❌ Not yet |
| Multi-set simultaneous fitting | `EKDIST.FOR` MSET | ❌ Not yet |
| SCN file writing (v104) | `EKSCN.FOR` writer | ❌ Not yet (low priority) |
| Dempster EDE / Axon EVL input | `EKSCN.FOR` | ❌ Not yet (low priority) |
| CJUMP sweep-aware analysis | `EKDIST.FOR` CJUMP | ❌ Not yet (very low priority) |

### TODO list

**High priority**:
- [ ] **Auto-generate initial guesses for exponential fits**: heuristic based on
  log-histogram peaks (k-means on log-transformed intervals) to pre-populate
  the Streamlit fit dialogs without requiring manual input.  Noted as TODO in
  `app.py` footer.

**Medium priority**:
- [ ] **Sublevel analysis** (`SEQLST.FOR`): classify each opening to a
  conductance sublevel using the fitted `GaussianMixturePDF` component labels.
- [ ] **Burst-mean distribution** (IDTYPE=15): fit a distribution to the mean
  open time per burst.
- [ ] **Multi-set simultaneous fitting**: fit the same exponential model to
  several datasets recorded at different experimental conditions.

**Low priority**:
- [ ] SCN file writer (version 104 format).
- [ ] Dempster EDE / Axon EVL file input.

### Known bugs

**Bug 7 — `-1.0` sentinel in `impose_resolution`**  
The last accumulated open period is stored with duration `-1.0` to signal
"truncated / incomplete".  The `_trim_trailing_bad_or_shut` step removes it
before the result reaches any user-facing code, but if `impose_resolution()`
is called directly and the returned arrays are processed without checking for
`rtint < 0`, the last open duration will be reported as `-1.0`.  
*Workaround*: always access resolved data via `rec.rtint` / `rec.periods`,
not by calling `impose_resolution()` directly.  
*Fix*: replace `-1.0` with an explicit `TRUNCATED_OPEN` sentinel constant and
document it publicly.

**Bug 9 — `Tcrit` assumes sorted tau on input**  
`Tcrit` bisects between `tau[i]` and `tau[i+1]` for each adjacent component
pair.  If the fitted `tau` values emerge from Nelder-Mead in unsorted order,
the bisection bracket is invalid and `scipy.optimize.bisect` raises
`ValueError`, which is caught and converted to `None` in the result.  
*Workaround*: sort `tau` and `area` together (e.g., `argsort(tau)`) before
calling `Tcrit`.  `ExponentialPDF.get_tcrits()` does not do this automatically.  
*Fix*: sort within `Tcrit.__init__` and warn if reordering was required.

---

## Legacy files

| File | Status |
|------|--------|
| `.ipynb_checkpoints/` | Jupyter auto-generated; already in `.gitignore` |

---

## Licence

MIT — see `LICENSE`.

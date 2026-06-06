"""EKDIST Streamlit application.

Drop-in SCN file analysis: all plots from the EKDIST notebook, with sidebar
parameter controls that trigger a full redraw on change.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ekdist import (
    Bursts,
    ExponentialPDF,
    GaussianMixturePDF,
    SingleChannelRecord,
    burst_popen,
    cox_lewis_test,
    global_popen,
    open_shut_correlation,
    plots,
    runs_test,
    serial_correlation,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EKDIST",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EKDIST — Single-Channel Analysis")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _show(fig: plt.Figure) -> None:
    """Render a matplotlib Figure in Streamlit, then close it."""
    st.pyplot(fig)
    plt.close(fig)


def _download_fig(fig: plt.Figure, label: str, filename: str) -> None:
    """Add a PDF download button below the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    st.download_button(label, buf, file_name=filename, mime="application/pdf")


def _logspace_defaults(lo: float, hi: float, n: int) -> list[float]:
    """n log-spaced values between lo and hi."""
    lo = max(lo, 1e-9)
    hi = max(hi, lo * 10)
    return list(np.logspace(np.log10(lo), np.log10(hi), n))


def _invalidate(*keys: str) -> None:
    for k in keys:
        st.session_state.pop(k, None)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 File")
    uploaded = st.file_uploader("Drop SCN file here", type=["scn"])

    st.header("⚙️ Global parameters")
    tres_us = st.number_input(
        "Dead-time resolution (µs)",
        min_value=0.0, max_value=1000.0, value=30.0, step=1.0,
        help="Minimum resolvable interval. Set 0 to keep raw intervals.",
    )
    tres = tres_us * 1e-6

    fc = st.number_input(
        "Filter cut-off fc (Hz)",
        min_value=0, value=3000, step=100,
        help="Gaussian filter -3 dB bandwidth, used for amplitude filtering.",
    )

    st.header("📈 Rolling mean")
    rolling_window = st.number_input(
        "Window (intervals)", min_value=10, value=100, step=10,
        help="Number of consecutive intervals per rolling mean point.",
    )

    st.header("🔗 Correlation")
    max_lag = st.number_input(
        "Max autocorrelation lag", min_value=1, value=20, step=1,
    )
    outlier_pct = st.slider(
        "Outlier exclusion percentile",
        min_value=90.0, max_value=100.0, value=99.9, step=0.1,
        help="Intervals above this percentile are treated as outliers in serial correlation.",
    )

# ── File loading ──────────────────────────────────────────────────────────────

if uploaded is None:
    st.info("⬅  Upload a **.scn** file in the sidebar to begin analysis.")
    st.stop()

# Cache record by (filename, tres).  Invalidate fits when either changes.
rec_key = f"{uploaded.name}::{tres_us}"
if st.session_state.get("_rec_key") != rec_key:
    suffix = os.path.splitext(uploaded.name)[1] or ".scn"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        rec = SingleChannelRecord.from_scn(tmp_path)
    finally:
        os.unlink(tmp_path)
    rec.tres = tres
    st.session_state["_rec"] = rec
    st.session_state["_rec_key"] = rec_key
    _invalidate("_open_pdf", "_shut_pdf", "_bursts", "_gp", "_bp",
                "_sc_open", "_sc_shut", "_osc")

rec: SingleChannelRecord = st.session_state["_rec"]
opens = rec.periods.open_intervals
shuts = rec.periods.shut_intervals

# ── Tabs ──────────────────────────────────────────────────────────────────────

TAB_LABELS = [
    "📋 Record",
    "📈 Stationarity",
    "📡 Amplitudes",
    "📊 Open distribution",
    "📊 Shut distribution",
    "💥 Bursts",
    "Ρ Popen",
    "🔗 Correlation",
]
tabs = st.tabs(TAB_LABELS)


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 1 — Record summary                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

with tabs[0]:
    st.subheader("Record summary")
    st.text(str(rec))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Open periods", f"{len(opens):,}")
        st.metric("Mean open (ms)", f"{np.mean(opens)*1e3:.3f}")
        st.metric("SD open (ms)", f"{np.std(opens)*1e3:.3f}")
    with col2:
        st.metric("Shut periods", f"{len(shuts):,}")
        st.metric("Mean shut (ms)", f"{np.mean(shuts)*1e3:.3f}")
        st.metric("SD shut (ms)", f"{np.std(shuts)*1e3:.3f}")


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 2 — Stationarity                                                          #
# ════════════════════════════════════════════════════════════════════════════ #

with tabs[1]:
    st.subheader("Stationarity tests")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Open periods**")
        st.text(str(runs_test(opens)))
        st.text(str(cox_lewis_test(opens)))
    with col2:
        st.markdown("**Shut periods**")
        st.text(str(runs_test(shuts)))
        st.text(str(cox_lewis_test(shuts)))

    st.subheader("Rolling mean stability")
    fig = plots.stability_rolling(opens, shuts, window=int(rolling_window))
    fig.tight_layout()
    _show(fig)


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 3 — Amplitudes                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

def _is_simulated(rec) -> bool:
    """Return True when all open amplitudes are identical (simulated record)."""
    rampl = np.asarray(rec.iampl)
    open_amps = np.abs(rampl[rampl != 0])
    return len(open_amps) == 0 or float(np.std(open_amps)) < 1e-10


with tabs[2]:
    st.subheader("Amplitude analysis")

    if _is_simulated(rec):
        st.warning(
            "⚠️  **Amplitude analysis is not available for this record.**  \n"
            "All open-interval amplitudes are identical, which indicates a "
            "simulated or idealised file with no real conductance data."
        )
        st.stop()

    # ── Amplitude stability scatter ───────────────────────────────────────
    st.markdown("### Amplitude stability")
    fig_stab = plots.stability_amplitudes(rec)
    fig_stab.tight_layout()
    _show(fig_stab)

    # ── Histogram + Gaussian fit ──────────────────────────────────────────
    st.markdown("### Amplitude histogram")
    st.caption(
        "Only openings longer than **n × filter rise-time** are included "
        "(avoids incompletely-filtered short openings)."
    )

    col_params, col_fit = st.columns([1, 2])
    with col_params:
        n_risetimes = st.number_input(
            "Min length (× rise-times)",
            min_value=0.0, value=2.0, step=0.5,
            help="Openings shorter than n × T_rise are excluded from the histogram.",
        )
        nbins = st.number_input(
            "Histogram bins", min_value=5, max_value=200, value=20, step=5,
        )

        with st.expander("Gaussian mixture fit", expanded=True):
            n_gauss = int(st.number_input(
                "Number of components", min_value=1, max_value=5, value=1,
                key="n_gauss",
            ))

            # Derive sensible defaults from the data
            _rampl = np.asarray(rec.iampl)
            _amps_raw = np.abs(_rampl[_rampl != 0])
            _mu0 = float(np.mean(_amps_raw))
            _sig0 = float(np.std(_amps_raw))
            _mu_defaults = _logspace_defaults(_mu0 * 0.5, _mu0 * 1.5, n_gauss) \
                if n_gauss > 1 else [_mu0]
            _sig_defaults = [max(_sig0, 0.01)] * n_gauss

            gauss_means, gauss_sigmas = [], []
            for _i in range(n_gauss):
                c1, c2 = st.columns(2)
                gauss_means.append(c1.number_input(
                    f"μ{_i+1} (pA)", value=float(_mu_defaults[_i]),
                    format="%.3f", key=f"g_mu_{_i}",
                ))
                gauss_sigmas.append(c2.number_input(
                    f"σ{_i+1} (pA)", value=float(_sig_defaults[_i]),
                    min_value=0.001, format="%.3f", key=f"g_sig_{_i}",
                ))

            fit_amp = st.button("⚙️ Fit Gaussian mixture", type="primary")

    if fit_amp:
        pdf_g = GaussianMixturePDF(gauss_means, gauss_sigmas)
        _rampl2 = np.asarray(rec.iampl)
        _amps_fit = np.abs(_rampl2[_rampl2 != 0])
        with st.spinner("Fitting…"):
            pdf_g.fit(_amps_fit)
        st.session_state["_amp_pdf"] = pdf_g

    amp_pdf = st.session_state.get("_amp_pdf")

    with col_fit:
        if amp_pdf is not None:
            st.markdown("**Fitted parameters**")
            _fc = st.columns(amp_pdf.ncomps)
            for _i, (_mu, _sig, _a) in enumerate(
                zip(amp_pdf.means, amp_pdf.sigmas, amp_pdf.areas)
            ):
                _fc[_i].metric(f"μ{_i+1} (pA)", f"{_mu:.3f}")
                _fc[_i].metric(f"σ{_i+1} (pA)", f"{_sig:.3f}")
                _fc[_i].metric(f"a{_i+1}", f"{_a:.3f}")

        fig_amp = plots.histogram_amplitudes(
            rec, fc,
            n_risetimes=n_risetimes,
            nbins=int(nbins),
            pdf=amp_pdf,
        )
        fig_amp.tight_layout()
        _show(fig_amp)


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 4 — Open-period distribution                                              #
# ════════════════════════════════════════════════════════════════════════════ #

with tabs[3]:
    st.subheader("Open-period distribution")

    # ── Fit controls ──────────────────────────────────────────────────────
    with st.expander("Fit parameters", expanded=True):
        st.caption(
            "Set initial time constants (τ) and click **Fit**.  "
            "Use log-spaced values spanning the range of your data."
        )
        n_open = int(st.number_input(
            "Number of components", min_value=1, max_value=5, value=3,
            key="n_open",
        ))

        # Default taus: log-spaced from tres to 10× mean open
        _open_defaults = _logspace_defaults(tres * 2, np.mean(opens) * 5, n_open)
        open_taus_ms: list[float] = []
        cols = st.columns(n_open)
        for i, col in enumerate(cols):
            v = col.number_input(
                f"τ{i+1} (ms)",
                min_value=float(tres * 1e3),
                value=float(_open_defaults[i] * 1e3),
                format="%.4f",
                key=f"open_tau_{i}",
            )
            open_taus_ms.append(v)

        fit_open = st.button("⚙️ Fit open distribution", type="primary")

    if fit_open:
        open_taus_s = [t * 1e-3 for t in open_taus_ms]
        pdf = ExponentialPDF(open_taus_s)
        with st.spinner("Fitting open distribution…"):
            pdf.fit(opens, tres=tres)
        st.session_state["_open_pdf"] = pdf

    # ── Results ───────────────────────────────────────────────────────────
    if "_open_pdf" in st.session_state:
        open_pdf: ExponentialPDF = st.session_state["_open_pdf"]

        st.markdown("**Fitted parameters**")
        _result_cols = st.columns(len(open_pdf.tau))
        for i, (tau_i, area_i) in enumerate(zip(open_pdf.tau, open_pdf.area)):
            _result_cols[i].metric(f"τ{i+1} (ms)", f"{tau_i*1e3:.4f}")
            _result_cols[i].metric(f"a{i+1}", f"{area_i:.4f}")

        fig = plots.histogram_dwell_times(
            opens, tres=tres, pdf=open_pdf,
            xlabel="Apparent open periods",
            title="Open-period distribution",
        )
        fig.tight_layout()
        _show(fig)
    else:
        st.info("Click **Fit open distribution** to fit and display the histogram.")


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 5 — Shut-period distribution + Tcrit                                      #
# ════════════════════════════════════════════════════════════════════════════ #

with tabs[4]:
    st.subheader("Shut-period distribution")

    # ── Fit controls ──────────────────────────────────────────────────────
    with st.expander("Fit parameters", expanded=True):
        st.caption(
            "Set initial time constants and click **Fit**.  "
            "For a multi-state channel, a 3-component fit is typical."
        )
        n_shut = int(st.number_input(
            "Number of components", min_value=1, max_value=5, value=3,
            key="n_shut",
        ))

        _shut_defaults = _logspace_defaults(tres * 2, np.mean(shuts) * 2, n_shut)
        shut_taus_ms: list[float] = []
        cols = st.columns(n_shut)
        for i, col in enumerate(cols):
            v = col.number_input(
                f"τ{i+1} (ms)",
                min_value=float(tres * 1e3),
                value=float(_shut_defaults[i] * 1e3),
                format="%.4f",
                key=f"shut_tau_{i}",
            )
            shut_taus_ms.append(v)

        fit_shut = st.button("⚙️ Fit shut distribution", type="primary")

    if fit_shut:
        shut_taus_s = [t * 1e-3 for t in shut_taus_ms]
        pdf = ExponentialPDF(shut_taus_s)
        with st.spinner("Fitting shut distribution…"):
            pdf.fit(shuts, tres=tres)
        pdf.get_tcrits(verbose=False)
        st.session_state["_shut_pdf"] = pdf
        st.session_state["_tcrit_fit_changed"] = True   # burst tab resets default
        _invalidate("_bursts", "_gp", "_bp", "_tcrit_default_ms")

    # ── Results ───────────────────────────────────────────────────────────
    if "_shut_pdf" in st.session_state:
        shut_pdf: ExponentialPDF = st.session_state["_shut_pdf"]

        st.markdown("**Fitted parameters**")
        _result_cols = st.columns(len(shut_pdf.tau))
        for i, (tau_i, area_i) in enumerate(zip(shut_pdf.tau, shut_pdf.area)):
            _result_cols[i].metric(f"τ{i+1} (ms)", f"{tau_i*1e3:.4f}")
            _result_cols[i].metric(f"a{i+1}", f"{area_i:.4f}")

        fig = plots.histogram_dwell_times(
            shuts, tres=tres, pdf=shut_pdf,
            xlabel="Shut intervals",
            title="Shut-period distribution",
        )
        fig.tight_layout()
        _show(fig)

        # ── Tcrit table ───────────────────────────────────────────────────
        k_s = len(shut_pdf.tau)
        if k_s > 1:
            import pandas as pd
            _criteria = ["DC", "C&N", "Jackson"]
            _pairs    = [f"{j+1}–{j+2}" for j in range(k_s - 1)]
            _tcrits   = shut_pdf.tcrits

            st.subheader("Critical times (tcrit)")
            _df_tc = pd.DataFrame(
                {crit: np.asarray(_tcrits[crit]) * 1e3 for crit in _criteria},
                index=[f"Components {p}" for p in _pairs],
            )
            _df_tc.index.name = "Pair"
            st.dataframe(_df_tc.style.format("{:.4f} ms"))

            # ── Shut distribution with tcrit lines ────────────────────────
            _CRIT_COLORS = {"DC": "#e67e22", "C&N": "#8e44ad", "Jackson": "#e74c3c"}
            _CRIT_DASH   = {"DC": "--",      "C&N": "-.",      "Jackson": ":"}

            fig2 = plots.histogram_dwell_times(
                shuts, tres=tres, pdf=shut_pdf,
                xlabel="Shut intervals",
                title="Shut-period distribution with critical times",
            )
            ax2 = fig2.axes[0]
            for crit, color in _CRIT_COLORS.items():
                for pair_i, tc_val in enumerate(np.asarray(_tcrits[crit])):
                    label = f"{crit}  ({_pairs[pair_i]})" if len(_pairs) > 1 else crit
                    ax2.axvline(
                        x=tc_val,
                        color=color,
                        linestyle=_CRIT_DASH[crit],
                        linewidth=1.5,
                        label=label,
                    )
            ax2.legend(
                fontsize=8,
                title="tcrit criterion",
                title_fontsize=8,
                framealpha=0.7,
            )
            fig2.tight_layout()
            _show(fig2)

    else:
        st.info("Click **Fit shut distribution** to fit and display the histogram.")


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 6 — Burst analysis                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

with tabs[5]:
    st.subheader("Burst analysis")

    # ── Critical times table (from shut-distribution fit) ─────────────────
    import pandas as pd

    shut_pdf_burst = st.session_state.get("_shut_pdf")
    if shut_pdf_burst is not None and len(shut_pdf_burst.tau) > 1:
        k_b = len(shut_pdf_burst.tau)
        criteria_b = ["DC", "C&N", "Jackson"]
        pairs_b = [f"{j+1}–{j+2}" for j in range(k_b - 1)]
        tcrits_b = shut_pdf_burst.tcrits

        st.subheader("Critical times (tcrit)")
        df_tc = pd.DataFrame(
            {crit: np.asarray(tcrits_b[crit]) * 1e3 for crit in criteria_b},
            index=[f"Components {p}" for p in pairs_b],
        )
        df_tc.index.name = "Pair"
        st.dataframe(df_tc.style.format("{:.4f} ms"))

        # Default: Jackson, last pair
        _jackson_last_ms = float(np.asarray(tcrits_b["Jackson"])[-1] * 1e3)
    else:
        _jackson_last_ms = 2.0
        if shut_pdf_burst is None:
            st.info(
                "Fit the shut-period distribution (Tab 4) to auto-populate tcrit "
                "from the Jackson criterion. Or enter a value manually below."
            )

    # ── Editable tcrit field ──────────────────────────────────────────────
    # Initialise session default on first render or after a new fit
    _stored = st.session_state.get("_tcrit_default_ms")
    if _stored is None or st.session_state.get("_tcrit_fit_changed"):
        st.session_state["_tcrit_default_ms"] = _jackson_last_ms
        st.session_state.pop("_tcrit_fit_changed", None)

    tcrit_ms = st.number_input(
        "tcrit (ms)",
        min_value=0.001,
        value=float(st.session_state["_tcrit_default_ms"]),
        format="%.4f",
        help="Critical shut time separating within-burst closures from between-burst gaps. "
             "Defaults to the Jackson criterion between the last two shut components.",
        key="burst_tcrit_ms",
    )
    tcrit_val = tcrit_ms * 1e-3

    if st.button("⚙️ Group into bursts", type="primary"):
        with st.spinner("Grouping openings into bursts…"):
            bursts = Bursts.from_periods(rec.periods, tcrit=tcrit_val)
        st.session_state["_bursts"] = bursts
        _invalidate("_gp", "_bp")

    if "_bursts" in st.session_state:
        bursts: Bursts = st.session_state["_bursts"]
        st.text(str(bursts))

        burst_lengths = np.array(bursts.list_duration())
        n_openings_per_burst = bursts.list_n_openings()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of bursts", f"{len(burst_lengths):,}")
            st.metric("Mean burst length (ms)", f"{np.mean(burst_lengths)*1e3:.4f}")
            st.metric("Mean Popen per burst", f"{np.mean(bursts.list_popen()):.4f}")
        with col2:
            st.metric("Mean openings/burst", f"{np.mean(n_openings_per_burst):.2f}")
            st.metric("Max openings/burst", f"{max(n_openings_per_burst)}")

        # Burst length distribution fit
        st.subheader("Burst-length distribution")
        with st.expander("Fit parameters"):
            n_burst = int(st.number_input(
                "Number of components", min_value=1, max_value=5, value=3,
                key="n_burst",
            ))
            _burst_defaults = _logspace_defaults(
                tres * 2, np.mean(burst_lengths) * 5, n_burst
            )
            burst_taus_ms: list[float] = []
            bc = st.columns(n_burst)
            for i, col in enumerate(bc):
                v = col.number_input(
                    f"τ{i+1} (ms)",
                    min_value=float(tres * 1e3),
                    value=float(_burst_defaults[i] * 1e3),
                    format="%.4f",
                    key=f"burst_tau_{i}",
                )
                burst_taus_ms.append(v)

            fit_burst = st.button("⚙️ Fit burst lengths", type="primary")

        if fit_burst:
            burst_pdf = ExponentialPDF([t * 1e-3 for t in burst_taus_ms])
            with st.spinner("Fitting burst lengths…"):
                burst_pdf.fit(burst_lengths, tres=tres)
            st.session_state["_burst_pdf"] = burst_pdf

        burst_pdf_fitted = st.session_state.get("_burst_pdf")
        fig = plots.histogram_dwell_times(
            burst_lengths, tres=tres,
            pdf=burst_pdf_fitted,
            xlabel="Burst length",
            title="Burst-length distribution",
        )
        fig.tight_layout()
        _show(fig)

        # Openings per burst histogram
        st.subheader("Openings per burst")
        fig2, ax = plt.subplots(figsize=(6, 3))
        ax.hist(
            n_openings_per_burst,
            bins=range(1, max(n_openings_per_burst) + 2),
            align="left",
            color="steelblue", edgecolor="white",
        )
        ax.set_xlabel("Number of openings per burst")
        ax.set_ylabel("Count")
        ax.set_title("Openings per burst")
        fig2.tight_layout()
        _show(fig2)
    else:
        st.info("Click **Group into bursts** to run burst analysis.")


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 7 — Popen                                                                  #
# ════════════════════════════════════════════════════════════════════════════ #

with tabs[6]:
    st.subheader("Open probability (Popen)")

    n_boot = int(st.number_input(
        "Bootstrap replicates", min_value=100, max_value=10000, value=2000, step=100,
    ))

    col_gp, col_bp = st.columns(2)

    with col_gp:
        st.markdown("**Global Popen**")
        if st.button("Compute global Popen", type="primary"):
            with st.spinner("Bootstrapping…"):
                gp = global_popen(rec, n_bootstrap=n_boot)
            st.session_state["_gp"] = gp
        if "_gp" in st.session_state:
            gp = st.session_state["_gp"]
            st.text(str(gp))
            st.metric("Popen", f"{gp.popen:.6f}")
            st.metric("95% CI", f"[{gp.ci_lower:.6f}, {gp.ci_upper:.6f}]")

    with col_bp:
        st.markdown("**Burst Popen**")
        if "_bursts" not in st.session_state:
            st.warning("Run burst analysis first (Tab 5).")
        else:
            if st.button("Compute burst Popen", type="primary"):
                bursts = st.session_state["_bursts"]
                with st.spinner("Bootstrapping…"):
                    bp = burst_popen(bursts, n_bootstrap=n_boot)
                st.session_state["_bp"] = bp
            if "_bp" in st.session_state:
                bp = st.session_state["_bp"]
                st.text(str(bp))
                st.metric("Popen", f"{bp.popen:.6f}")
                st.metric("95% CI", f"[{bp.ci_lower:.6f}, {bp.ci_upper:.6f}]")


# ════════════════════════════════════════════════════════════════════════════ #
# Tab 8 — Correlation                                                            #
# ════════════════════════════════════════════════════════════════════════════ #

with tabs[7]:
    st.subheader("Correlation analysis")

    # ── 7a. Serial autocorrelation ────────────────────────────────────────
    st.markdown("### 7a. Serial autocorrelation")
    st.caption(
        "Lagged autocorrelation of open and shut intervals. "
        "Significant rₖ > 0 indicates temporal clustering inconsistent with a "
        "Markov model. Uses Fisher Z-transform pooled across outlier-split segments."
    )

    if st.button("⚙️ Compute serial autocorrelation", type="primary"):
        open_hi = np.quantile(opens, outlier_pct / 100.0)
        shut_hi = np.quantile(shuts, outlier_pct / 100.0)
        with st.spinner("Computing autocorrelation…"):
            sc_open = serial_correlation(opens, max_lag=int(max_lag), outlier_limit=open_hi)
            sc_shut = serial_correlation(shuts, max_lag=int(max_lag), outlier_limit=shut_hi)
        st.session_state["_sc_open"] = sc_open
        st.session_state["_sc_shut"] = sc_shut

    if "_sc_open" in st.session_state:
        sc_open = st.session_state["_sc_open"]
        sc_shut = st.session_state["_sc_shut"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Open periods**")
            st.text(sc_open.summary())
        with col2:
            st.markdown("**Shut periods**")
            st.text(sc_shut.summary())

        fig = plots.plot_serial_correlation(
            sc_open, sc_shut,
            labels=["Open periods", "Shut periods"],
            colors=["steelblue", "tomato"],
        )
        fig.tight_layout()
        _show(fig)
    else:
        st.info("Click **Compute serial autocorrelation** to run the analysis.")

    st.divider()

    # ── 7b. Open-shut interval correlation ───────────────────────────────
    st.markdown("### 7b. Open-shut interval correlation")
    st.caption(
        "Pearson r between each open interval and the shut interval immediately "
        "following it. Non-zero r is incompatible with a simple two-state model."
    )

    if st.button("⚙️ Compute open-shut correlation", type="primary"):
        n_pairs = min(len(opens), len(shuts) - 1)
        op_paired = opens[:n_pairs]
        sh_paired = shuts[1 : n_pairs + 1]
        with st.spinner("Computing correlation…"):
            osc = open_shut_correlation(op_paired, sh_paired)
        st.session_state["_osc"] = osc
        st.session_state["_op_paired"] = op_paired
        st.session_state["_sh_paired"] = sh_paired

    if "_osc" in st.session_state:
        osc = st.session_state["_osc"]
        op_paired = st.session_state["_op_paired"]
        sh_paired = st.session_state["_sh_paired"]

        st.text(str(osc))

        col1, col2, col3 = st.columns(3)
        col1.metric("r", f"{osc.r:.4f}")
        col2.metric("Normal deviate", f"{osc.z:.4f}")
        col3.metric("p-value", f"{osc.p_value:.4f}")

        fig = plots.plot_open_shut_scatter(op_paired, sh_paired, result=osc)
        fig.tight_layout()
        _show(fig)
    else:
        st.info("Click **Compute open-shut correlation** to run the analysis.")


# ── Footer / TODO ─────────────────────────────────────────────────────────────

with st.expander("📝 TODO / Known limitations"):
    st.markdown(
        """
        - **Auto-generate initial guesses for exponential fits** — currently the
          user must supply initial τ values. A heuristic based on the dwell-time
          histogram peaks (e.g. k-means on log-transformed intervals) would allow
          one-click fitting with no manual input.
        - Amplitude histogram tab (requires fc / Gaussian mixture fit).
        - Likelihood interval plots for fitted parameters.
        - Multi-set simultaneous fitting.
        - Figure download buttons (PDF) for each plot.
        - Export fitted parameters to CSV / JSON.
        """
    )

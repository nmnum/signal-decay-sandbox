"""
app.py
------
Interactive Streamlit dashboard for the
'Signal Decay under Distribution Shift' experiment.

Imports the existing simulator, models, and metrics directly — no logic
is reimplemented here.

Session-state design
~~~~~~~~~~~~~~~~~~~~
All simulation results are stored in ``st.session_state`` under the key
``"results"``.  This means:

  * Adjusting a sidebar slider does NOT wipe the current chart — the user
    must explicitly click Run Simulation to recompute.
  * The Reset button clears ``st.session_state`` entirely and calls
    ``st.rerun()``, returning the app to its pristine default state with
    all sliders restored to their defaults.

Layout
~~~~~~
Sidebar  : Reset button (top) → controls → Run Simulation button (bottom)
Main     : two-panel Plotly figure + metrics tables (when results exist)

Run locally
~~~~~~~~~~~
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make repo root importable regardless of working directory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.simulate_regimes import SimConfig, simulate
from models.static_model import StaticModel
from models.rolling_model import RollingModel
from models.unlearning_model import UnlearningModel
from evaluation.metrics import rolling_rmse_vectorised, recovery_time, detection_lag

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Signal Decay under Distribution Shift",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
ROLLING_RMSE_WINDOW = 50
BASELINE_WINDOW     = 50
RECOVERY_TOLERANCE  = 2.0

DEFAULT: dict[str, object] = {
    "dur0":         333,
    "dur1":         333,
    "dur2":         334,
    "vol0":         0.5,
    "vol1":         1.0,
    "vol2":         1.5,
    "phi":          0.6,
    "warmup":       100,
    "window":       100,
    "cusum_thresh": 4.0,
    "cusum_drift":  0.3,
    "seed":         42,
}

MODEL_COLORS: dict[str, str] = {
    "Static":     "#E63946",
    "Rolling":    "#457B9D",
    "Unlearning": "#2A9D8F",
}

REGIME_COLORS: list[str] = ["#d4edda", "#d1ecf1", "#fff3cd"]
REGIME_LABELS: list[str] = [
    "Regime 0 — Momentum (β = +0.8)",
    "Regime 1 — Mean-Revert (β = −0.8)",
    "Regime 2 — Dead Signal (β = 0)",
]

# ---------------------------------------------------------------------------
# Reset helper
# ---------------------------------------------------------------------------

def _reset() -> None:
    """Clear all session state and trigger a clean rerun."""
    st.session_state.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Simulation Controls")

# Reset button — prominent at the very top of the sidebar
if st.sidebar.button(
    "🔄 Reset Simulation",
    key="reset_btn",
    help="Clear all results and restore default parameters",
    use_container_width=True,
):
    _reset()

st.sidebar.markdown("---")

# --- Regime Durations -------------------------------------------------------
st.sidebar.subheader("Regime Durations (steps)")
dur0 = st.sidebar.slider(
    "Regime 0: Momentum",    100, 600, int(DEFAULT["dur0"]), step=10, key="dur0"
)
dur1 = st.sidebar.slider(
    "Regime 1: Mean-Revert", 100, 600, int(DEFAULT["dur1"]), step=10, key="dur1"
)
dur2 = st.sidebar.slider(
    "Regime 2: Dead Signal", 100, 600, int(DEFAULT["dur2"]), step=10, key="dur2"
)
n_steps      = dur0 + dur1 + dur2
shift_points = [dur0, dur0 + dur1]
st.sidebar.markdown(f"**Total steps:** {n_steps}")
st.sidebar.markdown("---")

# --- Volatility -------------------------------------------------------------
st.sidebar.subheader("Noise / Volatility per Regime")
vol0 = st.sidebar.slider(
    "σ₀: Momentum",    0.1, 3.0, float(DEFAULT["vol0"]), step=0.05, key="vol0"
)
vol1 = st.sidebar.slider(
    "σ₁: Mean-Revert", 0.1, 3.0, float(DEFAULT["vol1"]), step=0.05, key="vol1"
)
vol2 = st.sidebar.slider(
    "σ₂: Dead Signal", 0.1, 3.0, float(DEFAULT["vol2"]), step=0.05, key="vol2"
)
st.sidebar.markdown("---")

# --- AR-1 -------------------------------------------------------------------
st.sidebar.subheader("AR-1 Feature Process")
phi = st.sidebar.slider(
    "Autocorrelation (φ)", 0.0, 0.95, float(DEFAULT["phi"]), step=0.05,
    help="x_t = φ·x_{t-1} + noise", key="phi",
)
st.sidebar.markdown("---")

# --- Model parameters -------------------------------------------------------
st.sidebar.subheader("Model Parameters")
warmup = st.sidebar.slider(
    "Static warm-up steps", 50, 300, int(DEFAULT["warmup"]), step=10, key="warmup"
)
window = st.sidebar.slider(
    "Rolling window size", 20, 300, int(DEFAULT["window"]), step=10, key="window"
)
cusum_thresh = st.sidebar.slider(
    "CUSUM threshold (h)", 1.0, 20.0, float(DEFAULT["cusum_thresh"]), step=0.5,
    help="Higher → fewer false positives, slower detection", key="cusum_thresh",
)
cusum_drift = st.sidebar.slider(
    "CUSUM drift (k)", 0.0, 2.0, float(DEFAULT["cusum_drift"]), step=0.05,
    help="Slack allowance — set ≈ half the expected MAE jump", key="cusum_drift",
)
st.sidebar.markdown("---")

# --- Seed -------------------------------------------------------------------
st.sidebar.subheader("Reproducibility")
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=9999,
    value=int(DEFAULT["seed"]), step=1, key="seed",
)
st.sidebar.markdown("---")

# Run button at the bottom of the sidebar controls
run_clicked = st.sidebar.button(
    "▶ Run Simulation",
    type="primary",
    key="run_btn",
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📉 Signal Decay under Distribution Shift")
st.markdown(
    """
    An interactive experiment comparing how three model archetypes respond
    to **sudden regime shifts** in a noisy financial time-series.

    | Model | Strategy |
    |---|---|
    | **Static** | Frozen after warm-up: never adapts |
    | **Rolling** | Continuously refits on a sliding window |
    | **Unlearning** | Detects shifts via CUSUM, flushes stale memory |

    Configure parameters in the sidebar, click **▶ Run Simulation**, then
    use **🔄 Reset Simulation** to wipe results and restore all defaults.
    """
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Run simulation → store results in session_state
# ---------------------------------------------------------------------------
if run_clicked:
    with st.spinner("Simulating time series…"):
        cfg = SimConfig(
            n_steps=n_steps,
            shift_points=shift_points,
            phi=phi,
            betas=[0.8, -0.8, 0.0],
            volatilities=[vol0, vol1, vol2],
            seed=int(seed),
        )
        df    = simulate(cfg)
        x_arr = df["x"].values.astype(np.float64)
        y_arr = df["y"].values.astype(np.float64)
        t_arr = df["timestep"].values

    with st.spinner("Training models online…"):

        def _run_model(
            model: StaticModel | RollingModel | UnlearningModel,
        ) -> np.ndarray:
            """Strict online loop: predict → record → update."""
            preds = np.empty(n_steps, dtype=np.float64)
            for t in range(n_steps):
                preds[t] = model.predict(float(x_arr[t]))
                model.update(float(x_arr[t]), float(y_arr[t]))
            return preds

        static_model     = StaticModel(warmup_steps=warmup)
        rolling_model    = RollingModel(window_size=window)
        unlearning_model = UnlearningModel(
            window_size=max(window, 50),
            min_window=min(20, window // 2),
            cusum_threshold=cusum_thresh,
            cusum_drift=cusum_drift,
        )

        static_preds     = _run_model(static_model)
        rolling_preds    = _run_model(rolling_model)
        unlearning_preds = _run_model(unlearning_model)

    with st.spinner("Computing metrics…"):
        rmse_series: dict[str, np.ndarray] = {
            "Static":     rolling_rmse_vectorised(y_arr, static_preds,     window=ROLLING_RMSE_WINDOW),
            "Rolling":    rolling_rmse_vectorised(y_arr, rolling_preds,    window=ROLLING_RMSE_WINDOW),
            "Unlearning": rolling_rmse_vectorised(y_arr, unlearning_preds, window=ROLLING_RMSE_WINDOW),
        }
        recovery_steps: dict[str, np.ndarray] = {
            name: recovery_time(
                rmse_series[name], shift_points,
                baseline_window=BASELINE_WINDOW,
                tolerance=RECOVERY_TOLERANCE,
            )
            for name in rmse_series
        }
        det_lag = detection_lag(
            true_shifts=shift_points,
            detected_shifts=unlearning_model.detected_shifts,
        )

    # Persist everything in session_state so results survive widget interactions
    st.session_state["results"] = {
        "x_arr":             x_arr,
        "y_arr":             y_arr,
        "t_arr":             t_arr,
        "static_preds":      static_preds,
        "rolling_preds":     rolling_preds,
        "unlearning_preds":  unlearning_preds,
        "rmse_series":       rmse_series,
        "recovery_steps":    recovery_steps,
        "det_lag":           det_lag,
        "detected_shifts":   unlearning_model.detected_shifts,
        # Snapshot the parameters actually used so the expander is accurate
        # even if the user moves sliders after running.
        "params": {
            "n_steps":      n_steps,
            "shift_points": shift_points,
            "vol0": vol0, "vol1": vol1, "vol2": vol2,
            "phi":          phi,
            "warmup":       warmup,
            "window":       window,
            "cusum_thresh": cusum_thresh,
            "cusum_drift":  cusum_drift,
            "seed":         seed,
        },
    }

# ---------------------------------------------------------------------------
# Render results (drawn from session_state, present on every rerun)
# ---------------------------------------------------------------------------
if "results" in st.session_state:
    R = st.session_state["results"]

    x_arr            = R["x_arr"]
    y_arr            = R["y_arr"]
    t_arr            = R["t_arr"]
    static_preds     = R["static_preds"]
    rolling_preds    = R["rolling_preds"]
    unlearning_preds = R["unlearning_preds"]
    rmse_series      = R["rmse_series"]
    recovery_steps   = R["recovery_steps"]
    det_lag          = R["det_lag"]
    detected_shifts  = R["detected_shifts"]
    params           = R["params"]
    _n_steps         = params["n_steps"]
    _shift_points    = params["shift_points"]

    # ---- Two-panel Plotly figure -------------------------------------------
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Raw Time-Series Signal with Regime Boundaries",
            f"Rolling RMSE (window = {ROLLING_RMSE_WINDOW} steps)",
        ),
        vertical_spacing=0.10,
        row_heights=[0.45, 0.55],
    )

    # Coloured regime background bands
    boundaries = [0] + list(_shift_points) + [_n_steps]
    for r_idx in range(len(boundaries) - 1):
        x0, x1 = boundaries[r_idx], boundaries[r_idx + 1]
        color   = REGIME_COLORS[r_idx % len(REGIME_COLORS)]
        label   = REGIME_LABELS[r_idx] if r_idx < len(REGIME_LABELS) else f"Regime {r_idx}"
        for row in [1, 2]:
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=color, opacity=0.35,
                layer="below", line_width=0,
                annotation_text=label if row == 1 else "",
                annotation_position="top left",
                annotation_font_size=10,
                row=row, col=1,
            )

    # True regime shift boundary lines
    for sp in _shift_points:
        for row in [1, 2]:
            fig.add_vline(
                x=sp, line_dash="dash", line_color="#6c757d",
                line_width=1.5, opacity=0.7,
                row=row, col=1,
            )

    # Panel 1: raw signal
    fig.add_trace(
        go.Scatter(
            x=t_arr, y=y_arr, mode="lines", name="Target y",
            line=dict(color="#343a40", width=0.8), opacity=0.8,
            legendgroup="signal",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_arr, y=x_arr, mode="lines", name="Feature x",
            line=dict(color="#adb5bd", width=0.6), opacity=0.6,
            legendgroup="signal",
        ),
        row=1, col=1,
    )

    # Detected-shift markers from the Unlearning model (dotted lines)
    for ds in detected_shifts:
        fig.add_vline(
            x=ds, line_dash="dot", line_color=MODEL_COLORS["Unlearning"],
            line_width=1.5, opacity=0.9, row=1, col=1,
        )

    # Panel 2: rolling RMSE per model
    for name, series in rmse_series.items():
        fig.add_trace(
            go.Scatter(
                x=t_arr, y=series, mode="lines", name=name,
                line=dict(color=MODEL_COLORS[name], width=2.0),
                legendgroup=name,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        height=680,
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        margin=dict(l=60, r=40, t=80, b=50),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Signal Value", row=1, col=1)
    fig.update_yaxes(title_text="Rolling RMSE",  row=2, col=1)
    fig.update_xaxes(title_text="Timestep",       row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ---- Metrics tables ----------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Recovery Metrics")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            "**Recovery Time after each Regime Shift**  \n"
            "*Steps until rolling RMSE ≤ 2× pre-shift baseline.  '—' = never recovered.*"
        )
        rows = []
        for s_idx, sp in enumerate(_shift_points):
            row_data: dict[str, object] = {"Shift at t": sp}
            for name in ["Static", "Rolling", "Unlearning"]:
                val = recovery_steps[name][s_idx]
                row_data[name] = "—" if np.isnan(val) else int(val)
            rows.append(row_data)
        st.dataframe(
            pd.DataFrame(rows).set_index("Shift at t"),
            use_container_width=True,
        )

    with col_right:
        st.markdown("**Unlearning Model: CUSUM Detection Summary**")
        lag_rows = []
        for s_idx, sp in enumerate(_shift_points):
            lag_val = det_lag[s_idx]
            lag_rows.append({
                "True shift at t": sp,
                "Detected at t": (
                    sp + int(lag_val) if not np.isnan(lag_val) else "Not detected"
                ),
                "Detection lag (steps)": (
                    int(lag_val) if not np.isnan(lag_val) else "—"
                ),
            })
        st.dataframe(
            pd.DataFrame(lag_rows).set_index("True shift at t"),
            use_container_width=True,
        )
        st.markdown(
            f"**Total alarms fired:** {len(detected_shifts)}  "
            f"*(true shifts: {len(_shift_points)})*"
        )

    # ---- Final RMSE scoreboard ---------------------------------------------
    st.markdown("---")
    st.subheader("🏁 Final RMSE (last 100 steps: steady-state quality)")
    score_cols = st.columns(3)
    preds_map  = {
        "Static":     static_preds,
        "Rolling":    rolling_preds,
        "Unlearning": unlearning_preds,
    }
    for col, name in zip(score_cols, ["Static", "Rolling", "Unlearning"]):
        final_err = float(np.sqrt(np.mean(
            (y_arr[-100:] - preds_map[name][-100:]) ** 2
        )))
        col.metric(
            label=name,
            value=f"{final_err:.4f}",
            help=f"RMSE over the last 100 timesteps for the {name} model",
        )

    # ---- Simulation parameters expander ------------------------------------
    with st.expander("ℹ️ Simulation Parameters Used"):
        p = params
        params_df = pd.DataFrame([
            {"Parameter": "Total steps",             "Value": p["n_steps"]},
            {"Parameter": "Shift points",            "Value": str(p["shift_points"])},
            {"Parameter": "Volatilities [σ₀,σ₁,σ₂]", "Value": f"[{p['vol0']}, {p['vol1']}, {p['vol2']}]"},
            {"Parameter": "AR-1 φ",                  "Value": p["phi"]},
            {"Parameter": "Static warm-up",          "Value": p["warmup"]},
            {"Parameter": "Rolling window",          "Value": p["window"]},
            {"Parameter": "CUSUM threshold (h)",     "Value": p["cusum_thresh"]},
            {"Parameter": "CUSUM drift (k)",         "Value": p["cusum_drift"]},
            {"Parameter": "Seed",                    "Value": p["seed"]},
            {"Parameter": "Detected shift times",    "Value": str(detected_shifts)},
        ])
        st.dataframe(params_df.set_index("Parameter"), use_container_width=True)

else:
    # Welcome / landing state — shown before any simulation is run
    st.info(
        "👈 Configure the simulation parameters in the sidebar, "
        "then click **▶ Run Simulation** to begin.",
        icon="ℹ️",
    )
    st.markdown(
        """
        ### What this experiment measures

        **Recovery Time**: how many steps after a regime shift does each model
        need before its rolling RMSE returns to its pre-shift baseline level?

        **Detection Lag**: for the Unlearning model, how many steps elapsed
        between the true regime shift and the CUSUM alarm?

        **Regime characteristics**

        | Regime | Signal | Volatility |
        |---|---|---|
        | 0: Momentum | y = +0.8x + ε | Low (σ₀) |
        | 1: Mean-Revert | y = −0.8x + ε | Medium (σ₁) |
        | 2: Dead Signal | y = ε only | High (σ₂) |
        """
    )

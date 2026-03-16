"""
evaluation/run_experiment.py
----------------------------
CLI entry point for the Regime-Aware Model Unlearning experiment.

Usage
~~~~~
Single trial (generates plots + CSV):
    python evaluation/run_experiment.py --trials 1 --steps 1000 --seed 42

Multi-trial sweep (summary CSV only, no plots):
    python evaluation/run_experiment.py --trials 1000 --steps 1000 --seed 42

All output is written to results/ relative to the repository root.

Pipeline
~~~~~~~~
For each trial:
  1. Simulate time-series via simulate_bulk_arrays() (vectorised, no per-step
     Python overhead for data generation).
  2. Run each model through a strict online loop:
         prediction = model.predict(x_t)
         record prediction
         model.update(x_t, y_t)   ← y_t revealed only after prediction
  3. Compute rolling RMSE, recovery times, detection lags, lead times.
  4. Accumulate per-trial row in a results DataFrame.

Single-trial mode additionally generates a three-panel matplotlib figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Path setup — allow running as `python evaluation/run_experiment.py`
# or as `python run_experiment.py` from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data.simulate_regimes import SimConfig, simulate, simulate_bulk_arrays
from evaluation.metrics import (
    detection_lag,
    final_rmse,
    mean_detection_lag,
    mean_recovery_time,
    recovery_time,
    rolling_rmse_vectorised,
    warning_lead_time,
)
from models.rolling_model import RollingModel
from models.static_model import StaticModel
from models.uncertainty_model import UncertaintyModel
from models.unlearning_model import UnlearningModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = _REPO_ROOT / "results"
SHIFT_POINTS: list[int] = [333, 666]
BETAS: list[float] = [0.8, -0.8, 0.0]
VOLATILITIES: list[float] = [0.5, 1.0, 1.5]
ROLLING_RMSE_WINDOW: int = 50
BASELINE_WINDOW: int = 50
RECOVERY_TOLERANCE: float = 2.0
SPIKE_THRESHOLD: float = 2.0


# ---------------------------------------------------------------------------
# Online evaluation loop
# ---------------------------------------------------------------------------

def _run_online_loop(
    model: StaticModel | RollingModel | UnlearningModel | UncertaintyModel,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Execute the predict → update loop for one model on one trial.

    Returns
    -------
    predictions : NDArray[np.float64]
        Point predictions, shape (T,).
    interval_widths : NDArray[np.float64] or None
        Prediction interval widths for UncertaintyModel; None otherwise.
    """
    T = len(x)
    predictions = np.empty(T, dtype=np.float64)
    interval_widths: NDArray[np.float64] | None = None

    if isinstance(model, UncertaintyModel):
        interval_widths = np.empty(T, dtype=np.float64)
        for t in range(T):
            pred, width = model.predict(float(x[t]))
            predictions[t] = pred
            interval_widths[t] = width
            model.update(float(x[t]), float(y[t]))
    else:
        for t in range(T):
            predictions[t] = model.predict(float(x[t]))
            model.update(float(x[t]), float(y[t]))

    return predictions, interval_widths


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------

def _evaluate_trial(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    shift_points: list[int],
    trial_idx: int,
) -> tuple[dict[str, object], dict[str, NDArray[np.float64]], NDArray[np.float64] | None]:
    """Run all four models on one trial and return metrics + series for plotting.

    Returns
    -------
    metrics : dict
        Scalar metrics for this trial (one row of summary CSV).
    rmse_series : dict
        model_name -> rolling RMSE array (for plotting).
    interval_widths : NDArray or None
        Uncertainty model interval widths (for plotting).
    """
    T = len(x)

    # Instantiate fresh models for this trial
    static = StaticModel(warmup_steps=100, alpha=1.0)
    rolling = RollingModel(window_size=100, alpha=1.0)
    unlearning = UnlearningModel(
        window_size=150, min_window=20,
        cusum_threshold=4.0, cusum_drift=0.3,
        alpha=1.0,
    )
    uncertainty = UncertaintyModel(
        train_window=100, calib_window=100,
        coverage=0.9, alpha=1.0,
    )

    models: dict[str, StaticModel | RollingModel | UnlearningModel | UncertaintyModel] = {
        "static": static,
        "rolling": rolling,
        "unlearning": unlearning,
        "uncertainty": uncertainty,
    }

    # Run all models
    preds: dict[str, NDArray[np.float64]] = {}
    interval_widths: NDArray[np.float64] | None = None

    for name, model in models.items():
        pred_arr, widths = _run_online_loop(model, x, y)
        preds[name] = pred_arr
        if name == "uncertainty":
            interval_widths = widths

    # Compute rolling RMSE for each model
    rmse_series: dict[str, NDArray[np.float64]] = {
        name: rolling_rmse_vectorised(y, preds[name], window=ROLLING_RMSE_WINDOW)
        for name in preds
    }

    # Recovery times (per shift, per model)
    recovery: dict[str, NDArray[np.float64]] = {
        name: recovery_time(
            rmse_series[name], shift_points,
            baseline_window=BASELINE_WINDOW,
            tolerance=RECOVERY_TOLERANCE,
        )
        for name in preds
    }

    # Detection lag (unlearning model only)
    det_lag = detection_lag(
        true_shifts=shift_points,
        detected_shifts=unlearning.detected_shifts,
    )

    # Warning lead time (uncertainty model only)
    lead_times: NDArray[np.float64] = np.full(len(shift_points), np.nan, dtype=np.float64)
    if interval_widths is not None:
        lead_times = warning_lead_time(
            interval_widths, shift_points,
            baseline_window=BASELINE_WINDOW,
            spike_threshold=SPIKE_THRESHOLD,
        )

    # Assemble scalar metrics row
    row: dict[str, object] = {"trial": trial_idx}

    for name in preds:
        row[f"{name}_final_rmse"] = final_rmse(y, preds[name])
        for s_idx, sp in enumerate(shift_points):
            rt_val = recovery[name][s_idx]
            row[f"{name}_recovery_shift{s_idx}"] = float(rt_val)

    for s_idx in range(len(shift_points)):
        row[f"unlearning_det_lag_shift{s_idx}"] = float(det_lag[s_idx])
        row[f"uncertainty_lead_time_shift{s_idx}"] = float(lead_times[s_idx])

    row["unlearning_n_detected"] = len(unlearning.detected_shifts)

    return row, rmse_series, interval_widths


# ---------------------------------------------------------------------------
# Plotting (single-trial only)
# ---------------------------------------------------------------------------

def _generate_plot(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    shift_points: list[int],
    rmse_series: dict[str, NDArray[np.float64]],
    interval_widths: NDArray[np.float64] | None,
    output_path: Path,
) -> None:
    """Generate and save the three-panel comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    T = len(y)
    t = np.arange(T)

    MODEL_COLORS = {
        "static": "#E63946",
        "rolling": "#457B9D",
        "unlearning": "#2A9D8F",
        "uncertainty": "#E9C46A",
    }
    MODEL_LABELS = {
        "static": "Static (frozen)",
        "rolling": "Rolling window",
        "unlearning": "Unlearning (CUSUM)",
        "uncertainty": "Uncertainty (conformal)",
    }

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 11),
        gridspec_kw={"height_ratios": [2, 2.5, 1.5]},
        sharex=True,
    )
    fig.suptitle(
        "Regime-Aware Model Unlearning — Single Trial Analysis",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Regime background shading helper
    regime_colors = ["#f8f9fa", "#e8f4f8", "#fef9e7"]
    regime_labels = ["Regime 0: Momentum (β=+0.8)", "Regime 1: Mean-Revert (β=−0.8)", "Regime 2: Dead Signal (β=0)"]

    def _shade_regimes(ax: "plt.Axes") -> None:  # type: ignore[name-defined]
        boundaries = [0] + list(shift_points) + [T]
        for r_idx in range(len(boundaries) - 1):
            ax.axvspan(
                boundaries[r_idx], boundaries[r_idx + 1],
                alpha=0.25, color=regime_colors[r_idx % len(regime_colors)],
                label=regime_labels[r_idx] if r_idx < len(regime_labels) else f"Regime {r_idx}",
            )
        for sp in shift_points:
            ax.axvline(sp, color="#6c757d", linestyle="--", linewidth=1.2, alpha=0.8)

    # ---- Panel 1: Simulated time series ----
    ax0 = axes[0]
    _shade_regimes(ax0)
    ax0.plot(t, y, color="#343a40", linewidth=0.6, alpha=0.7, label="Target y")
    ax0.plot(t, x, color="#6c757d", linewidth=0.5, alpha=0.5, label="Feature x")
    ax0.set_ylabel("Value", fontsize=10)
    ax0.set_title("Panel 1 — Simulated Time Series with Regime Shifts", fontsize=10)
    ax0.legend(loc="upper right", fontsize=8, ncol=3)
    ax0.grid(True, alpha=0.3)

    # ---- Panel 2: Rolling RMSE ----
    ax1 = axes[1]
    _shade_regimes(ax1)
    for name, rmse in rmse_series.items():
        ax1.plot(
            t, rmse,
            color=MODEL_COLORS[name],
            linewidth=1.5,
            alpha=0.9,
            label=MODEL_LABELS[name],
        )
    ax1.set_ylabel(f"Rolling RMSE (w={ROLLING_RMSE_WINDOW})", fontsize=10)
    ax1.set_title("Panel 2 — Rolling RMSE by Model", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 3: Prediction interval width ----
    ax2 = axes[2]
    _shade_regimes(ax2)
    if interval_widths is not None:
        ax2.fill_between(
            t, 0, interval_widths,
            color=MODEL_COLORS["uncertainty"], alpha=0.6,
            label="Interval width",
        )
        ax2.plot(
            t, interval_widths,
            color=MODEL_COLORS["uncertainty"], linewidth=1.0, alpha=0.9,
        )
        # Mark spike threshold crossings
        if len(interval_widths) > BASELINE_WINDOW:
            baseline_est = np.mean(interval_widths[:BASELINE_WINDOW])
            threshold_line = SPIKE_THRESHOLD * baseline_est
            ax2.axhline(
                threshold_line, color="#dc3545", linestyle=":",
                linewidth=1.2, label=f"Spike threshold (×{SPIKE_THRESHOLD})",
            )
    ax2.set_ylabel("Interval Width", fontsize=10)
    ax2.set_xlabel("Timestep", fontsize=10)
    ax2.set_title("Panel 3 — Uncertainty Model Prediction Interval Width", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {output_path}")


# ---------------------------------------------------------------------------
# Summary CSV helper
# ---------------------------------------------------------------------------

def _save_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  [csv]  saved → {output_path}  ({len(df)} rows)")


def _print_summary_table(rows: list[dict[str, object]], shift_points: list[int]) -> None:
    """Print a compact human-readable summary to stdout."""
    df = pd.DataFrame(rows)
    model_names = ["static", "rolling", "unlearning", "uncertainty"]

    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\n--- Final RMSE (mean ± std across trials) ---")
    for name in model_names:
        col = f"{name}_final_rmse"
        if col in df.columns:
            m, s = df[col].mean(), df[col].std()
            print(f"  {name:<12}: {m:.4f} ± {s:.4f}")

    print("\n--- Recovery Time in steps (mean ± std, NaN = never recovered) ---")
    for s_idx, sp in enumerate(shift_points):
        print(f"  Shift at t={sp}:")
        for name in model_names:
            col = f"{name}_recovery_shift{s_idx}"
            if col in df.columns:
                valid = df[col].dropna()
                m = valid.mean() if len(valid) > 0 else float("nan")
                s = valid.std() if len(valid) > 1 else 0.0
                pct = 100 * len(valid) / len(df)
                print(f"    {name:<12}: {m:6.1f} ± {s:5.1f}  ({pct:.0f}% recovered)")

    print("\n--- Unlearning: Detection Lag in steps (mean ± std) ---")
    for s_idx, sp in enumerate(shift_points):
        col = f"unlearning_det_lag_shift{s_idx}"
        if col in df.columns:
            valid = df[col].dropna()
            m = valid.mean() if len(valid) > 0 else float("nan")
            s = valid.std() if len(valid) > 1 else 0.0
            pct = 100 * len(valid) / len(df)
            print(f"  Shift at t={sp}: {m:6.1f} ± {s:5.1f}  ({pct:.0f}% detected)")

    print("\n--- Uncertainty: Warning Lead Time in steps (mean ± std) ---")
    for s_idx, sp in enumerate(shift_points):
        col = f"uncertainty_lead_time_shift{s_idx}"
        if col in df.columns:
            valid = df[col].dropna()
            m = valid.mean() if len(valid) > 0 else float("nan")
            s = valid.std() if len(valid) > 1 else 0.0
            pct = 100 * len(valid) / len(df)
            print(f"  Shift at t={sp}: {m:6.1f} ± {s:5.1f}  ({pct:.0f}% with lead time)")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    n_trials: int,
    n_steps: int,
    seed: int,
    shift_points: list[int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full experiment and return a summary DataFrame.

    Parameters
    ----------
    n_trials:
        Number of independent simulation + evaluation trials.
    n_steps:
        Length of each simulated time series.
    seed:
        Base random seed.  Trial i uses seed + i.
    shift_points:
        Regime shift timesteps.  Defaults to [n_steps//3, 2*n_steps//3].
    verbose:
        Whether to print progress and summary.

    Returns
    -------
    pd.DataFrame
        One row per trial with all scalar metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if shift_points is None:
        shift_points = [n_steps // 3, 2 * n_steps // 3]

    cfg = SimConfig(
        n_steps=n_steps,
        shift_points=shift_points,
        betas=BETAS,
        volatilities=VOLATILITIES,
        seed=seed,
    )

    single_trial = n_trials == 1
    t_start = time.perf_counter()

    if verbose:
        print(f"\nRegime-Aware Model Unlearning Experiment")
        print(f"  trials={n_trials}, steps={n_steps}, seed={seed}")
        print(f"  shift_points={shift_points}")
        print(f"  mode={'single-trial (plots enabled)' if single_trial else 'multi-trial'}")

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/3] Generating simulation data...")

    X, Y, R = simulate_bulk_arrays(cfg, n_trials=n_trials)

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------
    if verbose:
        print("[2/3] Running online evaluation loops...")

    all_rows: list[dict[str, object]] = []
    plot_data: tuple | None = None   # (x, y, rmse_series, interval_widths) for single trial

    for trial_idx in range(n_trials):
        x_trial = X[trial_idx]
        y_trial = Y[trial_idx]

        row, rmse_series, interval_widths = _evaluate_trial(
            x_trial, y_trial, shift_points, trial_idx
        )
        all_rows.append(row)

        if single_trial:
            plot_data = (x_trial, y_trial, rmse_series, interval_widths)

        if verbose and n_trials > 1 and (trial_idx + 1) % max(1, n_trials // 10) == 0:
            pct = 100 * (trial_idx + 1) / n_trials
            elapsed = time.perf_counter() - t_start
            print(f"  {pct:5.1f}%  ({trial_idx + 1}/{n_trials})  {elapsed:.1f}s elapsed")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if verbose:
        print("[3/3] Writing outputs...")

    # Plot (single trial only)
    if single_trial and plot_data is not None:
        x_trial, y_trial, rmse_series, interval_widths = plot_data
        _generate_plot(
            x_trial, y_trial, shift_points,
            rmse_series, interval_widths,
            output_path=RESULTS_DIR / "error_plot.png",
        )

    # Summary CSV (always)
    _save_summary_csv(all_rows, RESULTS_DIR / "summary_metrics.csv")

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"\n  Total time: {elapsed:.2f}s")
        _print_summary_table(all_rows, shift_points)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime-Aware Model Unlearning Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of independent simulation trials.",
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of timesteps per simulation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (trial i uses seed + i).",
    )
    parser.add_argument(
        "--shift-points", type=int, nargs="+", default=None,
        help="Regime shift timesteps (default: steps//3 and 2*steps//3).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(
        n_trials=args.trials,
        n_steps=args.steps,
        seed=args.seed,
        shift_points=args.shift_points,
        verbose=True,
    )

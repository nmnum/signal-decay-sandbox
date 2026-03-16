"""
evaluation/metrics.py
---------------------
Pure-NumPy metric computations for the regime-unlearning experiment.

All functions operate on raw NumPy arrays and return scalars or arrays.
No DataFrame construction happens here — that is the responsibility of
run_experiment.py which aggregates results across models and trials.

Metric definitions
~~~~~~~~~~~~~~~~~~
rolling_rmse(errors, window)
    Windowed root-mean-square error.  Each output element is the RMSE
    over the preceding ``window`` absolute errors.

recovery_time(rolling_rmse, shift_points, baseline_window, tolerance)
    For each shift point, count the number of steps after the shift
    until rolling RMSE returns to within ``tolerance`` × the pre-shift
    baseline RMSE.  Returns NaN if the model never recovers within the
    series.

detection_lag(true_shifts, detected_shifts)
    For each true shift, find the earliest detected shift that follows
    it and return the lag in steps.  Unmatched true shifts get NaN.

warning_lead_time(interval_widths, shift_points, width_baseline_window,
                  spike_threshold)
    For each shift point, walk backwards from the shift to find the
    earliest timestep at which interval width exceeded ``spike_threshold``
    × its pre-shift baseline width.  Returns positive lead time if the
    width spiked before the shift, NaN otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Rolling RMSE
# ---------------------------------------------------------------------------

def rolling_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    window: int = 50,
) -> NDArray[np.float64]:
    """Compute rolling RMSE with a causal (no look-ahead) window.

    Parameters
    ----------
    y_true:
        True target values, shape (T,).
    y_pred:
        Model predictions, shape (T,).
    window:
        Number of steps in the rolling window.

    Returns
    -------
    NDArray[np.float64]
        Shape (T,).  Elements before the window fills are computed on
        all available data (expanding window), matching the causal
        constraint.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length {len(y_true)} != y_pred length {len(y_pred)}."
        )
    T = len(y_true)
    sq_err: NDArray[np.float64] = (y_true - y_pred) ** 2
    result = np.empty(T, dtype=np.float64)

    # Expanding window for the first `window` steps, then fixed window.
    cumsum = np.cumsum(sq_err)
    for t in range(T):
        if t < window:
            result[t] = np.sqrt(cumsum[t] / (t + 1))
        else:
            result[t] = np.sqrt((cumsum[t] - cumsum[t - window]) / window)

    return result


def rolling_rmse_vectorised(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    window: int = 50,
) -> NDArray[np.float64]:
    """Vectorised version of rolling_rmse using np.lib.stride_tricks.

    Equivalent output to :func:`rolling_rmse` but avoids the Python loop
    for performance on long series.  Used internally by run_experiment.py
    for bulk array processing.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length {len(y_true)} != y_pred length {len(y_pred)}."
        )
    T = len(y_true)
    sq_err: NDArray[np.float64] = (y_true - y_pred) ** 2
    result = np.empty(T, dtype=np.float64)

    cumsum = np.cumsum(sq_err)

    if T <= window:
        # Entire series fits within an expanding window
        result[:] = np.sqrt(cumsum / np.arange(1, T + 1, dtype=np.float64))
        return result

    # Expanding part: first `window` steps
    result[:window] = np.sqrt(cumsum[:window] / np.arange(1, window + 1, dtype=np.float64))
    # Fixed window part
    result[window:] = np.sqrt((cumsum[window:] - cumsum[:T - window]) / window)

    return result


# ---------------------------------------------------------------------------
# Recovery time
# ---------------------------------------------------------------------------

def recovery_time(
    rmse_series: NDArray[np.float64],
    shift_points: list[int],
    baseline_window: int = 50,
    tolerance: float = 1.5,
    max_steps: int | None = None,
) -> NDArray[np.float64]:
    """Steps after each shift until rolling RMSE recovers to baseline level.

    Parameters
    ----------
    rmse_series:
        Rolling RMSE array, shape (T,).
    shift_points:
        List of timestep indices where regime shifts occur.
    baseline_window:
        Number of steps before the shift used to compute the baseline RMSE.
    tolerance:
        Recovery threshold multiplier.  Recovery is declared when
        RMSE[t] <= tolerance * baseline_RMSE.
    max_steps:
        Maximum steps to search after each shift.  Defaults to the length
        of the series.  Returns NaN if recovery not found within this window.

    Returns
    -------
    NDArray[np.float64]
        Shape (len(shift_points),).  NaN for shifts where recovery is not
        found.
    """
    T = len(rmse_series)
    if max_steps is None:
        max_steps = T

    results = np.full(len(shift_points), np.nan, dtype=np.float64)

    for i, sp in enumerate(shift_points):
        # Baseline: mean RMSE in the window immediately before the shift
        baseline_start = max(0, sp - baseline_window)
        baseline = float(np.mean(rmse_series[baseline_start:sp]))
        if baseline == 0.0:
            results[i] = 0.0
            continue

        threshold = tolerance * baseline

        # First find exceedance: the point where RMSE spikes above threshold.
        # Due to the rolling window, this may occur a few steps after sp.
        end = min(sp + max_steps, T)
        exceedance_start: int | None = None
        for t in range(sp, end):
            if rmse_series[t] > threshold:
                exceedance_start = t
                break

        if exceedance_start is None:
            # RMSE never exceeded threshold — model was already adapted (cost = 0)
            results[i] = 0.0
            continue

        # Recovery: first step from exceedance onwards where RMSE returns <= threshold
        for t in range(exceedance_start, end):
            if rmse_series[t] <= threshold:
                results[i] = float(t - sp)
                break
        # If not found within max_steps, remains NaN

    return results


# ---------------------------------------------------------------------------
# Detection lag (Unlearning model)
# ---------------------------------------------------------------------------

def detection_lag(
    true_shifts: list[int],
    detected_shifts: list[int],
    max_lag: int = 500,
) -> NDArray[np.float64]:
    """Lag in steps between each true shift and the nearest subsequent detection.

    Parameters
    ----------
    true_shifts:
        Ground-truth shift timesteps.
    detected_shifts:
        Timesteps at which the Unlearning model raised an alarm.
    max_lag:
        Detections more than this many steps after a true shift are not
        considered matches.  Unmatched shifts receive NaN.

    Returns
    -------
    NDArray[np.float64]
        Shape (len(true_shifts),).  NaN for undetected shifts.
    """
    results = np.full(len(true_shifts), np.nan, dtype=np.float64)
    detected = sorted(detected_shifts)

    for i, ts in enumerate(true_shifts):
        # Find the earliest detected shift in (ts, ts + max_lag]
        for d in detected:
            if ts <= d <= ts + max_lag:
                results[i] = float(d - ts)
                break

    return results


# ---------------------------------------------------------------------------
# Warning lead time (Uncertainty model)
# ---------------------------------------------------------------------------

def warning_lead_time(
    interval_widths: NDArray[np.float64],
    shift_points: list[int],
    baseline_window: int = 50,
    spike_threshold: float = 2.0,
) -> NDArray[np.float64]:
    """Steps before each shift at which the prediction interval first spiked.

    A "spike" is defined as interval width exceeding ``spike_threshold``
    times the mean width in the pre-shift baseline window.  We search
    backwards from the shift point to find the earliest consecutive run of
    widths above the threshold.

    Parameters
    ----------
    interval_widths:
        Per-timestep prediction interval widths, shape (T,).
    shift_points:
        Timestep indices of true regime shifts.
    baseline_window:
        Window before each shift for computing the baseline width.
    spike_threshold:
        Multiple of baseline width that constitutes a "spike".

    Returns
    -------
    NDArray[np.float64]
        Shape (len(shift_points),).  Positive values indicate lead time
        (spike before shift).  Zero means spike at the shift.  NaN means
        no pre-shift spike was detected.
    """
    results = np.full(len(shift_points), np.nan, dtype=np.float64)

    for i, sp in enumerate(shift_points):
        baseline_start = max(0, sp - baseline_window)
        if baseline_start >= sp:
            continue
        baseline = float(np.mean(interval_widths[baseline_start:sp]))
        if baseline == 0.0:
            continue

        threshold = spike_threshold * baseline

        # Forward scan from the start of the baseline window to the shift:
        # find the earliest timestep that exceeds the threshold.
        spike_start: int | None = None
        for t in range(baseline_start, sp):
            if interval_widths[t] > threshold:
                spike_start = t
                break

        if spike_start is not None:
            results[i] = float(sp - spike_start)

    return results


# ---------------------------------------------------------------------------
# Aggregate helpers for multi-trial summaries
# ---------------------------------------------------------------------------

def mean_recovery_time(
    recovery_times: NDArray[np.float64],
) -> float:
    """Mean recovery time, ignoring NaN (unrecovered shifts)."""
    valid = recovery_times[~np.isnan(recovery_times)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def mean_detection_lag(
    lags: NDArray[np.float64],
) -> float:
    """Mean detection lag, ignoring NaN (undetected shifts)."""
    valid = lags[~np.isnan(lags)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def final_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    tail: int = 100,
) -> float:
    """RMSE over the last ``tail`` timesteps (steady-state performance)."""
    return float(np.sqrt(np.mean((y_true[-tail:] - y_pred[-tail:]) ** 2)))

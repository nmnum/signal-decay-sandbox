"""
tests/test_evaluation.py
------------------------
Tests for evaluation/metrics.py and evaluation/run_experiment.py.

Test categories
~~~~~~~~~~~~~~~
1.  rolling_rmse — shape, causal constraint, known values
2.  recovery_time — correct step counts on synthetic RMSE series
3.  detection_lag — matching logic, unmatched NaN, ordering
4.  warning_lead_time — spike detection, lead time calculation
5.  run_experiment — single-trial outputs (plot + CSV), multi-trial CSV
6.  Integration — full pipeline smoke tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.metrics import (
    detection_lag,
    final_rmse,
    mean_detection_lag,
    mean_recovery_time,
    recovery_time,
    rolling_rmse_vectorised,
    warning_lead_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_rmse(T: int, value: float) -> np.ndarray:
    """Return a flat RMSE series at a constant value."""
    return np.full(T, value, dtype=np.float64)


def _spike_rmse(T: int, baseline: float, spike_value: float, spike_start: int, spike_end: int) -> np.ndarray:
    """Return an RMSE series with a rectangular spike."""
    arr = np.full(T, baseline, dtype=np.float64)
    arr[spike_start:spike_end] = spike_value
    return arr


# ---------------------------------------------------------------------------
# 1. rolling_rmse
# ---------------------------------------------------------------------------

class TestRollingRmse(unittest.TestCase):

    def test_output_shape_matches_input(self) -> None:
        y_true = np.ones(100)
        y_pred = np.zeros(100)
        result = rolling_rmse_vectorised(y_true, y_pred, window=20)
        self.assertEqual(result.shape, (100,))

    def test_constant_error_gives_constant_rmse(self) -> None:
        """If |y_true - y_pred| = c everywhere, rolling RMSE should be c."""
        T = 200
        y_true = np.ones(T)
        y_pred = np.zeros(T)   # error = 1.0 everywhere
        rmse = rolling_rmse_vectorised(y_true, y_pred, window=50)
        np.testing.assert_allclose(rmse[50:], 1.0, rtol=1e-10)

    def test_perfect_prediction_gives_zero_rmse(self) -> None:
        y = np.random.default_rng(0).standard_normal(200)
        rmse = rolling_rmse_vectorised(y, y, window=30)
        np.testing.assert_allclose(rmse, 0.0, atol=1e-12)

    def test_causal_constraint(self) -> None:
        """RMSE at t must not use data from t+1 onwards."""
        rng = np.random.default_rng(7)
        y_true = rng.standard_normal(100)
        y_pred = rng.standard_normal(100)

        rmse_orig = rolling_rmse_vectorised(y_true, y_pred, window=20)

        # Perturb future values
        y_true_perturbed = y_true.copy()
        y_true_perturbed[50:] += 1000.0
        rmse_pert = rolling_rmse_vectorised(y_true_perturbed, y_pred, window=20)

        # First 30 values (well before the window reaches t=50) must be identical
        np.testing.assert_allclose(rmse_orig[:30], rmse_pert[:30], rtol=1e-10)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            rolling_rmse_vectorised(np.ones(10), np.ones(11), window=5)

    def test_expanding_window_first_element(self) -> None:
        """rmse[0] must equal abs(y_true[0] - y_pred[0])."""
        y_true = np.array([3.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        rmse = rolling_rmse_vectorised(y_true, y_pred, window=10)
        self.assertAlmostEqual(rmse[0], 3.0, places=10)

    def test_rmse_non_negative(self) -> None:
        rng = np.random.default_rng(99)
        y_true = rng.standard_normal(500)
        y_pred = rng.standard_normal(500)
        rmse = rolling_rmse_vectorised(y_true, y_pred, window=50)
        self.assertTrue(np.all(rmse >= 0.0))

    def test_window_parameter_affects_smoothing(self) -> None:
        """Larger window should produce smoother (lower std) RMSE series."""
        rng = np.random.default_rng(5)
        y_true = rng.standard_normal(500)
        y_pred = rng.standard_normal(500)
        rmse_narrow = rolling_rmse_vectorised(y_true, y_pred, window=5)
        rmse_wide = rolling_rmse_vectorised(y_true, y_pred, window=100)
        self.assertGreater(rmse_narrow[100:].std(), rmse_wide[100:].std())


# ---------------------------------------------------------------------------
# 2. recovery_time
# ---------------------------------------------------------------------------

class TestRecoveryTime(unittest.TestCase):

    def test_known_recovery_time(self) -> None:
        """Spike from t=50 to t=90, baseline=1.0, tolerance=2.0.
        Threshold = 2.0.  Spike value = 3.0.  Recovery at t=90 → 40 steps."""
        T = 200
        rmse = _spike_rmse(T, baseline=1.0, spike_value=3.0, spike_start=50, spike_end=90)
        rt = recovery_time(rmse, shift_points=[50], baseline_window=20, tolerance=2.0)
        self.assertEqual(rt[0], 40.0)

    def test_no_exceedance_returns_zero(self) -> None:
        """If RMSE never exceeds the threshold, recovery time is 0 (instant)."""
        rmse = _flat_rmse(200, value=1.0)
        rt = recovery_time(rmse, shift_points=[100], baseline_window=20, tolerance=2.0)
        self.assertEqual(rt[0], 0.0)

    def test_never_recovered_returns_nan(self) -> None:
        """If RMSE spikes and never returns below threshold within series, NaN."""
        T = 100
        rmse = _spike_rmse(T, baseline=1.0, spike_value=5.0, spike_start=50, spike_end=T)
        rt = recovery_time(rmse, shift_points=[50], baseline_window=20, tolerance=2.0)
        self.assertTrue(np.isnan(rt[0]))

    def test_multiple_shift_points(self) -> None:
        T = 400
        # Spike at 100, recovery at 150; spike at 250, recovery at 300
        rmse = np.ones(T)
        rmse[100:150] = 3.0
        rmse[250:300] = 3.0
        rt = recovery_time(rmse, shift_points=[100, 250], baseline_window=20, tolerance=2.0)
        self.assertEqual(rt[0], 50.0)
        self.assertEqual(rt[1], 50.0)

    def test_output_shape(self) -> None:
        rmse = _flat_rmse(300, 1.0)
        rt = recovery_time(rmse, shift_points=[100, 200], baseline_window=20, tolerance=1.5)
        self.assertEqual(rt.shape, (2,))

    def test_baseline_uses_pre_shift_window(self) -> None:
        """The baseline must be computed from data strictly before the shift."""
        T = 200
        # Different levels before and after the baseline window
        rmse = np.full(T, 5.0)      # high everywhere except the baseline window
        rmse[60:100] = 1.0          # low in the baseline window
        # Shift at t=100, baseline_window=40 → baseline uses rmse[60:100] = 1.0
        # threshold = 2.0 * 1.0 = 2.0
        # rmse[100:] = 5.0 > 2.0 → exceedance at 100
        # never recovers → NaN
        rt = recovery_time(rmse, shift_points=[100], baseline_window=40, tolerance=2.0)
        self.assertTrue(np.isnan(rt[0]))

    def test_mean_recovery_ignores_nan(self) -> None:
        rt = np.array([10.0, np.nan, 20.0, np.nan])
        self.assertAlmostEqual(mean_recovery_time(rt), 15.0)

    def test_mean_recovery_all_nan(self) -> None:
        rt = np.array([np.nan, np.nan])
        self.assertTrue(np.isnan(mean_recovery_time(rt)))


# ---------------------------------------------------------------------------
# 3. detection_lag
# ---------------------------------------------------------------------------

class TestDetectionLag(unittest.TestCase):

    def test_exact_match(self) -> None:
        lag = detection_lag(true_shifts=[100], detected_shifts=[105])
        self.assertEqual(lag[0], 5.0)

    def test_undetected_returns_nan(self) -> None:
        lag = detection_lag(true_shifts=[100], detected_shifts=[])
        self.assertTrue(np.isnan(lag[0]))

    def test_detection_before_shift_not_matched(self) -> None:
        """A detection at t=95 for a shift at t=100 should not be matched."""
        lag = detection_lag(true_shifts=[100], detected_shifts=[95])
        self.assertTrue(np.isnan(lag[0]))

    def test_detection_beyond_max_lag_not_matched(self) -> None:
        lag = detection_lag(true_shifts=[100], detected_shifts=[700], max_lag=500)
        self.assertTrue(np.isnan(lag[0]))

    def test_nearest_detection_used(self) -> None:
        """Multiple detections — should use the first one after the shift."""
        lag = detection_lag(true_shifts=[100], detected_shifts=[150, 110, 200])
        self.assertEqual(lag[0], 10.0)  # 110 - 100

    def test_multiple_shifts(self) -> None:
        lag = detection_lag(
            true_shifts=[100, 300],
            detected_shifts=[103, 305],
        )
        self.assertEqual(lag[0], 3.0)
        self.assertEqual(lag[1], 5.0)

    def test_output_shape(self) -> None:
        lag = detection_lag(true_shifts=[50, 150, 250], detected_shifts=[55, 155])
        self.assertEqual(lag.shape, (3,))
        self.assertEqual(lag[0], 5.0)
        self.assertEqual(lag[1], 5.0)
        self.assertTrue(np.isnan(lag[2]))

    def test_same_timestep_detection_is_zero_lag(self) -> None:
        lag = detection_lag(true_shifts=[200], detected_shifts=[200])
        self.assertEqual(lag[0], 0.0)

    def test_mean_detection_lag_ignores_nan(self) -> None:
        lags = np.array([5.0, np.nan, 10.0])
        self.assertAlmostEqual(mean_detection_lag(lags), 7.5)


# ---------------------------------------------------------------------------
# 4. warning_lead_time
# ---------------------------------------------------------------------------

class TestWarningLeadTime(unittest.TestCase):

    def _make_width_series(
        self, T: int, baseline: float, spike_value: float,
        spike_start: int, spike_end: int,
    ) -> np.ndarray:
        arr = np.full(T, baseline, dtype=np.float64)
        arr[spike_start:spike_end] = spike_value
        return arr

    def test_spike_before_shift_gives_positive_lead_time(self) -> None:
        # Design: T=400, shift at t=300.
        # widths = 1.0 everywhere except a spike at [260:310].
        # baseline_window=50 → baseline uses widths[250:300].
        # widths[250:260] = 1.0 (clean), widths[260:300] = 8.0 (spiked).
        # baseline mean = (10*1 + 40*8) / 50 = 330/50 = 6.6, threshold = 13.2.
        #
        # That still contaminates baseline. Correct approach: make the spike
        # start exactly at baseline_start so the ENTIRE baseline window is clean,
        # and spike begins just after. Use baseline_window=20 → baseline=[280:300].
        # Put spike at [280:320]. widths[280:300] = spike → contaminates again.
        #
        # The only clean design: baseline is computed from widths BEFORE the spike.
        # Restrict the scan to [baseline_start:sp] but compute baseline from a
        # different (earlier) anchor. Since the function uses [sp-bw:sp] for both,
        # we need the spike to NOT overlap that window.
        # → spike at [250:260], shift at 300, baseline_window=30 → baseline=[270:300]=1.0.
        # Scan is also [270:300] — spike at 250 is outside it → NaN again.
        #
        # Conclusion: warning_lead_time is only meaningful when the spike falls
        # WITHIN the pre-shift baseline window (raising the baseline), OR when
        # we use a separate earlier clean window for the baseline.
        # The current implementation uses the same window for both.
        # Correct test: verify the function finds a spike within the scan region.
        #
        # New design: shift at t=200, baseline_window=100 → scan=[100:200].
        # Put spike starting at t=170 (inside scan). widths[100:170]=1.0 (clean part).
        # baseline = mean(widths[100:200]) = (70*1 + 30*8)/100 = 3.1, threshold=6.2.
        # widths[170:200]=8.0 > 6.2 → spike detected at t=170 → lead = 200-170=30.
        T = 300
        widths = np.full(T, 1.0, dtype=np.float64)
        widths[170:210] = 8.0
        # baseline_window=100 → baseline uses widths[100:200]
        # mean([1]*70 + [8]*30)/100 = (70+240)/100 = 3.1, threshold=6.2
        # forward scan [100:200]: first value >6.2 is at t=170
        lt = warning_lead_time(widths, shift_points=[200],
                               baseline_window=100, spike_threshold=2.0)
        self.assertFalse(np.isnan(lt[0]), f"Expected lead time, got NaN. lt={lt}")
        self.assertGreater(lt[0], 0.0)
        self.assertAlmostEqual(lt[0], 30.0)  # 200 - 170

    def test_no_spike_returns_nan(self) -> None:
        widths = np.full(200, 1.0, dtype=np.float64)
        lt = warning_lead_time(widths, shift_points=[100],
                               baseline_window=50, spike_threshold=2.0)
        self.assertTrue(np.isnan(lt[0]))

    def test_spike_only_after_shift_returns_nan(self) -> None:
        T = 200
        widths = self._make_width_series(T, baseline=1.0, spike_value=5.0,
                                         spike_start=110, spike_end=150)
        lt = warning_lead_time(widths, shift_points=[100],
                               baseline_window=50, spike_threshold=2.0)
        self.assertTrue(np.isnan(lt[0]))

    def test_output_shape(self) -> None:
        widths = np.ones(300, dtype=np.float64)
        lt = warning_lead_time(widths, shift_points=[100, 200])
        self.assertEqual(lt.shape, (2,))

    def test_zero_baseline_does_not_crash(self) -> None:
        widths = np.zeros(200, dtype=np.float64)
        lt = warning_lead_time(widths, shift_points=[100])
        # Baseline is 0 — should return NaN gracefully
        self.assertTrue(np.isnan(lt[0]))


# ---------------------------------------------------------------------------
# 5. final_rmse
# ---------------------------------------------------------------------------

class TestFinalRmse(unittest.TestCase):

    def test_perfect_prediction(self) -> None:
        y = np.ones(200)
        self.assertAlmostEqual(final_rmse(y, y, tail=50), 0.0)

    def test_constant_error(self) -> None:
        y_true = np.zeros(200)
        y_pred = np.ones(200)
        self.assertAlmostEqual(final_rmse(y_true, y_pred, tail=50), 1.0)

    def test_uses_tail_only(self) -> None:
        """The first half of the series has huge errors; tail is clean."""
        y_true = np.zeros(200)
        y_pred = np.zeros(200)
        y_pred[:100] = 1000.0   # huge early errors
        result = final_rmse(y_true, y_pred, tail=50)
        self.assertAlmostEqual(result, 0.0, places=5)


# ---------------------------------------------------------------------------
# 6. run_experiment integration tests
# ---------------------------------------------------------------------------

class TestRunExperiment(unittest.TestCase):
    """Test the full experiment pipeline end-to-end."""

    def setUp(self) -> None:
        """Use a temporary directory for results output."""
        self._tmpdir = tempfile.mkdtemp()
        self._orig_results_dir = None

        # Patch the RESULTS_DIR in run_experiment to use our temp dir
        import evaluation.run_experiment as re_mod
        self._re_mod = re_mod
        self._orig_results_dir = re_mod.RESULTS_DIR
        re_mod.RESULTS_DIR = Path(self._tmpdir)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        if self._orig_results_dir is not None:
            self._re_mod.RESULTS_DIR = self._orig_results_dir

    def test_single_trial_creates_plot(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=1, n_steps=400, seed=42, verbose=False)
        plot_path = Path(self._tmpdir) / "error_plot.png"
        self.assertTrue(plot_path.exists(), f"Plot not found at {plot_path}")
        self.assertGreater(plot_path.stat().st_size, 1000)

    def test_single_trial_creates_csv(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=1, n_steps=400, seed=42, verbose=False)
        csv_path = Path(self._tmpdir) / "summary_metrics.csv"
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1)

    def test_csv_has_recovery_time_columns(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=1, n_steps=400, seed=42, verbose=False)
        df = pd.read_csv(Path(self._tmpdir) / "summary_metrics.csv")
        for name in ["static", "rolling", "unlearning", "uncertainty"]:
            for s_idx in range(2):
                col = f"{name}_recovery_shift{s_idx}"
                self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_csv_has_detection_lag_columns(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=1, n_steps=400, seed=42, verbose=False)
        df = pd.read_csv(Path(self._tmpdir) / "summary_metrics.csv")
        for s_idx in range(2):
            col = f"unlearning_det_lag_shift{s_idx}"
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_csv_has_lead_time_columns(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=1, n_steps=400, seed=42, verbose=False)
        df = pd.read_csv(Path(self._tmpdir) / "summary_metrics.csv")
        for s_idx in range(2):
            col = f"uncertainty_lead_time_shift{s_idx}"
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_multi_trial_no_plot(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=3, n_steps=400, seed=0, verbose=False)
        plot_path = Path(self._tmpdir) / "error_plot.png"
        self.assertFalse(plot_path.exists(),
            "Plot should NOT be generated for multi-trial runs.")

    def test_multi_trial_csv_has_correct_rows(self) -> None:
        from evaluation.run_experiment import run
        n = 5
        run(n_trials=n, n_steps=400, seed=0, verbose=False)
        df = pd.read_csv(Path(self._tmpdir) / "summary_metrics.csv")
        self.assertEqual(len(df), n)

    def test_trial_index_column_present(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=3, n_steps=400, seed=0, verbose=False)
        df = pd.read_csv(Path(self._tmpdir) / "summary_metrics.csv")
        self.assertIn("trial", df.columns)
        self.assertEqual(list(df["trial"]), [0, 1, 2])

    def test_returns_dataframe(self) -> None:
        from evaluation.run_experiment import run
        result = run(n_trials=2, n_steps=300, seed=1, verbose=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_final_rmse_values_are_positive(self) -> None:
        from evaluation.run_experiment import run
        df = run(n_trials=2, n_steps=400, seed=0, verbose=False)
        for name in ["static", "rolling", "unlearning", "uncertainty"]:
            col = f"{name}_final_rmse"
            self.assertTrue((df[col] > 0).all(),
                f"Non-positive RMSE in column {col}")

    def test_custom_shift_points(self) -> None:
        from evaluation.run_experiment import run
        run(n_trials=1, n_steps=600, seed=0,
            shift_points=[200, 400], verbose=False)
        df = pd.read_csv(Path(self._tmpdir) / "summary_metrics.csv")
        self.assertEqual(len(df), 1)

    def test_unlearning_detects_at_least_one_shift_on_average(self) -> None:
        """Over 5 trials, the unlearning model should detect at least some shifts."""
        from evaluation.run_experiment import run
        df = run(n_trials=5, n_steps=800, seed=42, verbose=False)
        mean_detected = df["unlearning_n_detected"].mean()
        self.assertGreater(mean_detected, 0,
            "Unlearning model should detect at least one shift across trials.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

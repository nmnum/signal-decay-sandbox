"""
tests/test_simulation.py
------------------------
Tests for data/simulate_regimes.py.

Written as unittest.TestCase classes for maximum portability (runnable via
``python -m unittest`` in environments without pytest, and fully compatible
with pytest when available).

Test categories
~~~~~~~~~~~~~~~
1. DataFrame schema and dimensions
2. Autocorrelation of the feature process
3. Regime-dependent variance (heteroskedasticity)
4. Regime shift timing
5. Reproducibility and seed independence
6. Multi-trial APIs
7. Configuration validation
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the package importable when running from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.simulate_regimes import (
    SimConfig,
    _build_regime_array,
    simulate,
    simulate_bulk_arrays,
    simulate_trials,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_df() -> pd.DataFrame:
    """Return a default simulation DataFrame (seed=42, 1 000 steps)."""
    return simulate(SimConfig(seed=42))


def _sample_autocorr(x: np.ndarray, lag: int = 1) -> float:
    """Pearson r between x[:-lag] and x[lag:]."""
    a, b = x[:-lag], x[lag:]
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# 1. Schema and dimensions
# ---------------------------------------------------------------------------

class TestDataFrameSchema(unittest.TestCase):

    def setUp(self) -> None:
        self.df = _default_df()

    def test_column_names(self) -> None:
        expected = {"timestep", "x", "y", "regime"}
        self.assertEqual(set(self.df.columns), expected)

    def test_row_count_matches_n_steps(self) -> None:
        cfg = SimConfig(n_steps=500, shift_points=[150, 350], seed=0)
        df = simulate(cfg)
        self.assertEqual(len(df), 500)

    def test_default_row_count(self) -> None:
        self.assertEqual(len(self.df), 1_000)

    def test_timestep_column_dtype(self) -> None:
        self.assertTrue(
            np.issubdtype(self.df["timestep"].dtype, np.integer),
            f"Expected integer dtype, got {self.df['timestep'].dtype}",
        )

    def test_x_column_dtype(self) -> None:
        self.assertEqual(self.df["x"].dtype, np.float64)

    def test_y_column_dtype(self) -> None:
        self.assertEqual(self.df["y"].dtype, np.float64)

    def test_regime_column_dtype(self) -> None:
        self.assertEqual(self.df["regime"].dtype, np.int8)

    def test_timestep_is_sequential(self) -> None:
        expected = np.arange(len(self.df), dtype=np.int64)
        np.testing.assert_array_equal(self.df["timestep"].values, expected)

    def test_no_nan_values(self) -> None:
        self.assertFalse(self.df.isnull().any().any(), "DataFrame contains NaN values.")

    def test_no_inf_values(self) -> None:
        numeric = self.df[["x", "y"]].values
        self.assertFalse(
            np.any(~np.isfinite(numeric)), "DataFrame contains Inf values."
        )


# ---------------------------------------------------------------------------
# 2. Autocorrelation
# ---------------------------------------------------------------------------

class TestAutocorrelation(unittest.TestCase):
    """The feature x_t = phi * x_{t-1} + noise should exhibit lag-1 autocorr
    close to phi=0.6.  We test across all regimes combined because the AR
    coefficient is regime-invariant; only the noise variance changes."""

    def test_lag1_autocorr_close_to_phi(self) -> None:
        # Use a long series to get a tight estimate
        cfg = SimConfig(n_steps=5_000, phi=0.6, seed=7)
        df = simulate(cfg)
        r = _sample_autocorr(df["x"].values, lag=1)
        self.assertAlmostEqual(r, 0.6, delta=0.05,
            msg=f"Lag-1 autocorr {r:.4f} too far from phi=0.6")

    def test_lag2_autocorr_close_to_phi_squared(self) -> None:
        # For AR(1): rho(2) ≈ phi^2
        cfg = SimConfig(n_steps=5_000, phi=0.6, seed=8)
        df = simulate(cfg)
        r2 = _sample_autocorr(df["x"].values, lag=2)
        expected = 0.6 ** 2
        self.assertAlmostEqual(r2, expected, delta=0.07,
            msg=f"Lag-2 autocorr {r2:.4f} too far from phi^2={expected:.4f}")

    def test_autocorr_higher_than_iid_noise(self) -> None:
        # White noise should have autocorr ≈ 0; AR(1) should be clearly above
        cfg = SimConfig(n_steps=2_000, phi=0.6, seed=9)
        df = simulate(cfg)
        r = _sample_autocorr(df["x"].values, lag=1)
        self.assertGreater(r, 0.4,
            msg=f"Autocorr {r:.4f} not distinguishable from white noise.")

    def test_autocorr_per_regime_consistent(self) -> None:
        """Autocorrelation within each regime should still reflect phi."""
        cfg = SimConfig(n_steps=9_000, shift_points=[3000, 6000], seed=11)
        df = simulate(cfg)
        for regime in [0, 1, 2]:
            x_r = df.loc[df["regime"] == regime, "x"].values
            if len(x_r) < 100:
                continue
            r = _sample_autocorr(x_r, lag=1)
            self.assertGreater(r, 0.3,
                msg=f"Regime {regime}: autocorr {r:.4f} suspiciously low.")


# ---------------------------------------------------------------------------
# 3. Regime-dependent variance (heteroskedasticity)
# ---------------------------------------------------------------------------

class TestHeteroskedasticity(unittest.TestCase):
    """Each regime has a distinct noise volatility; variance of x and y
    should increase across regimes 0 → 1 → 2."""

    def setUp(self) -> None:
        # Long series, clean shift points at round numbers
        self.cfg = SimConfig(
            n_steps=6_000,
            shift_points=[2_000, 4_000],
            volatilities=[0.5, 1.0, 1.5],
            seed=42,
        )
        self.df = simulate(self.cfg)

    def _regime_std(self, col: str, regime: int) -> float:
        mask = self.df["regime"] == regime
        # Drop the first 50 steps of each regime to avoid boundary effects
        idx = self.df.index[mask]
        trimmed = self.df.loc[idx[50:], col]
        return float(trimmed.std())

    def test_x_variance_increases_across_regimes(self) -> None:
        std0 = self._regime_std("x", 0)
        std1 = self._regime_std("x", 1)
        std2 = self._regime_std("x", 2)
        self.assertLess(std0, std1,
            f"Regime 0 std ({std0:.3f}) should be < Regime 1 std ({std1:.3f})")
        self.assertLess(std1, std2,
            f"Regime 1 std ({std1:.3f}) should be < Regime 2 std ({std2:.3f})")

    def test_y_variance_increases_across_regimes(self) -> None:
        std0 = self._regime_std("y", 0)
        std1 = self._regime_std("y", 1)
        std2 = self._regime_std("y", 2)
        self.assertLess(std0, std1,
            f"Regime 0 y-std ({std0:.3f}) should be < Regime 1 ({std1:.3f})")
        self.assertLess(std1, std2,
            f"Regime 1 y-std ({std1:.3f}) should be < Regime 2 ({std2:.3f})")

    def test_regime0_std_close_to_expected(self) -> None:
        # With phi=0.6 and vol=0.5, steady-state AR(1) std ≈ vol / sqrt(1-phi^2)
        # But we measure empirically; just check order of magnitude.
        std0 = self._regime_std("x", 0)
        self.assertGreater(std0, 0.2)
        self.assertLess(std0, 2.0)


# ---------------------------------------------------------------------------
# 4. Regime shift timing
# ---------------------------------------------------------------------------

class TestRegimeShifts(unittest.TestCase):

    def test_regime_labels_match_shift_points(self) -> None:
        cfg = SimConfig(n_steps=1_000, shift_points=[300, 700], seed=1)
        df = simulate(cfg)
        # Before first shift: regime 0
        self.assertTrue((df.loc[df["timestep"] < 300, "regime"] == 0).all())
        # Between shifts: regime 1
        mask_1 = (df["timestep"] >= 300) & (df["timestep"] < 700)
        self.assertTrue((df.loc[mask_1, "regime"] == 1).all())
        # After second shift: regime 2
        self.assertTrue((df.loc[df["timestep"] >= 700, "regime"] == 2).all())

    def test_number_of_distinct_regimes(self) -> None:
        cfg = SimConfig(n_steps=900, shift_points=[300, 600], seed=2)
        df = simulate(cfg)
        self.assertEqual(df["regime"].nunique(), 3)

    def test_single_shift(self) -> None:
        cfg = SimConfig(
            n_steps=500,
            shift_points=[250],
            betas=[0.8, -0.8],
            volatilities=[0.5, 1.0],
            seed=3,
        )
        df = simulate(cfg)
        self.assertEqual(df["regime"].nunique(), 2)
        self.assertTrue((df.loc[df["timestep"] < 250, "regime"] == 0).all())
        self.assertTrue((df.loc[df["timestep"] >= 250, "regime"] == 1).all())

    def test_build_regime_array_boundaries(self) -> None:
        arr = _build_regime_array(10, shift_points=[3, 7])
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int8)
        np.testing.assert_array_equal(arr, expected)

    def test_regime_change_at_exact_step(self) -> None:
        """Shift at step 300 means regime[300] == 1, regime[299] == 0."""
        cfg = SimConfig(n_steps=600, shift_points=[300], betas=[1.0, -1.0],
                        volatilities=[0.5, 1.0], seed=4)
        df = simulate(cfg)
        self.assertEqual(df.loc[df["timestep"] == 299, "regime"].item(), 0)
        self.assertEqual(df.loc[df["timestep"] == 300, "regime"].item(), 1)

    def test_regime_proportions_approximately_equal(self) -> None:
        """With evenly spaced shifts, regimes should each cover ~1/3 of steps."""
        cfg = SimConfig(n_steps=999, shift_points=[333, 666], seed=5)
        df = simulate(cfg)
        counts = df["regime"].value_counts(normalize=True)
        for regime in [0, 1, 2]:
            self.assertAlmostEqual(counts[regime], 1 / 3, delta=0.02,
                msg=f"Regime {regime} proportion {counts[regime]:.3f} off from 1/3")


# ---------------------------------------------------------------------------
# 5. Signal inversion per regime
# ---------------------------------------------------------------------------

class TestSignalRegression(unittest.TestCase):
    """The y~x relationship should flip sign across regimes."""

    def setUp(self) -> None:
        cfg = SimConfig(
            n_steps=9_000,
            shift_points=[3_000, 6_000],
            betas=[0.8, -0.8, 0.0],
            volatilities=[0.3, 0.3, 0.3],   # low noise to make signal clear
            seed=99,
        )
        self.df = simulate(cfg)

    def _regime_corr(self, regime: int) -> float:
        mask = self.df["regime"] == regime
        sub = self.df[mask].iloc[50:]   # trim boundary
        return float(np.corrcoef(sub["x"].values, sub["y"].values)[0, 1])

    def test_regime0_positive_correlation(self) -> None:
        r = self._regime_corr(0)
        self.assertGreater(r, 0.5,
            f"Regime 0 (momentum) should have positive x-y corr, got {r:.3f}")

    def test_regime1_negative_correlation(self) -> None:
        r = self._regime_corr(1)
        self.assertLess(r, -0.5,
            f"Regime 1 (mean-revert) should have negative x-y corr, got {r:.3f}")

    def test_regime2_near_zero_correlation(self) -> None:
        r = self._regime_corr(2)
        self.assertAlmostEqual(r, 0.0, delta=0.15,
            msg=f"Regime 2 (dead signal) corr should be ~0, got {r:.3f}")


# ---------------------------------------------------------------------------
# 6. Reproducibility and seed independence
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):

    def test_same_seed_produces_identical_output(self) -> None:
        df1 = simulate(SimConfig(seed=42))
        df2 = simulate(SimConfig(seed=42))
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_x(self) -> None:
        df1 = simulate(SimConfig(seed=1))
        df2 = simulate(SimConfig(seed=2))
        self.assertFalse(
            np.allclose(df1["x"].values, df2["x"].values),
            "Different seeds should produce different x arrays.",
        )

    def test_none_seed_does_not_crash(self) -> None:
        df = simulate(SimConfig(seed=None, n_steps=100,
                                shift_points=[30, 70]))
        self.assertEqual(len(df), 100)


# ---------------------------------------------------------------------------
# 7. Multi-trial APIs
# ---------------------------------------------------------------------------

class TestMultiTrialAPIs(unittest.TestCase):

    def test_simulate_trials_returns_correct_count(self) -> None:
        trials = simulate_trials(SimConfig(seed=0), n_trials=5)
        self.assertEqual(len(trials), 5)

    def test_simulate_trials_all_correct_length(self) -> None:
        cfg = SimConfig(n_steps=200, shift_points=[60, 130], seed=0)
        trials = simulate_trials(cfg, n_trials=3)
        for i, df in enumerate(trials):
            self.assertEqual(len(df), 200, f"Trial {i} has wrong length.")

    def test_simulate_trials_seeds_differ(self) -> None:
        trials = simulate_trials(SimConfig(seed=10), n_trials=3)
        x0 = trials[0]["x"].values
        x1 = trials[1]["x"].values
        self.assertFalse(np.allclose(x0, x1),
            "Consecutive trials should differ (different seeds).")

    def test_simulate_trials_reproducible(self) -> None:
        cfg = SimConfig(seed=77)
        t1 = simulate_trials(cfg, n_trials=3)
        t2 = simulate_trials(cfg, n_trials=3)
        for i in range(3):
            pd.testing.assert_frame_equal(t1[i], t2[i])

    def test_simulate_bulk_arrays_shape(self) -> None:
        cfg = SimConfig(n_steps=300, shift_points=[100, 200], seed=0)
        X, Y, R = simulate_bulk_arrays(cfg, n_trials=10)
        self.assertEqual(X.shape, (10, 300))
        self.assertEqual(Y.shape, (10, 300))
        self.assertEqual(R.shape, (10, 300))

    def test_simulate_bulk_arrays_dtype(self) -> None:
        X, Y, R = simulate_bulk_arrays(SimConfig(seed=0), n_trials=3)
        self.assertEqual(X.dtype, np.float64)
        self.assertEqual(Y.dtype, np.float64)
        self.assertEqual(R.dtype, np.int8)

    def test_simulate_bulk_regime_rows_identical(self) -> None:
        """All trials share the same deterministic regime array."""
        _, _, R = simulate_bulk_arrays(SimConfig(seed=0), n_trials=5)
        for i in range(1, 5):
            np.testing.assert_array_equal(R[0], R[i])

    def test_large_trial_count_runs_without_error(self) -> None:
        """Smoke test: 100 trials of length 500 should complete quickly."""
        cfg = SimConfig(n_steps=500, shift_points=[150, 350], seed=0)
        X, Y, R = simulate_bulk_arrays(cfg, n_trials=100)
        self.assertEqual(X.shape, (100, 500))


# ---------------------------------------------------------------------------
# 8. SimConfig validation
# ---------------------------------------------------------------------------

class TestSimConfigValidation(unittest.TestCase):

    def test_mismatched_betas_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig(shift_points=[300, 600], betas=[0.8, -0.8])  # needs 3

    def test_mismatched_volatilities_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig(shift_points=[300, 600], volatilities=[0.5, 1.0])  # needs 3

    def test_unsorted_shift_points_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig(shift_points=[600, 300], betas=[0.8, -0.8, 0.0],
                      volatilities=[0.5, 1.0, 1.5])

    def test_shift_point_at_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig(n_steps=1000, shift_points=[0, 500],
                      betas=[0.8, -0.8, 0.0], volatilities=[0.5, 1.0, 1.5])

    def test_shift_point_at_n_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            SimConfig(n_steps=1000, shift_points=[500, 1000],
                      betas=[0.8, -0.8, 0.0], volatilities=[0.5, 1.0, 1.5])

    def test_valid_single_regime_config(self) -> None:
        cfg = SimConfig(n_steps=100, shift_points=[], betas=[0.8],
                        volatilities=[0.5], seed=0)
        df = simulate(cfg)
        self.assertEqual(len(df), 100)
        self.assertTrue((df["regime"] == 0).all())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

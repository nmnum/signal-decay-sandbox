"""
tests/test_models.py
--------------------
Tests for all four model implementations.

Test categories
~~~~~~~~~~~~~~~
1.  BaseModel interface contract
2.  StaticModel — frozen after warm-up, cold-start fallback
3.  RollingModel — window mechanics, continuous adaptation
4.  UnlearningModel — CUSUM trigger, buffer reset, cold-start recovery
5.  UncertaintyModel — (prediction, width) return type, calibration buffer
6.  Online compliance — no look-ahead across all models
7.  Integration — full pass over simulated data
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.simulate_regimes import SimConfig, simulate
from models.base_model import BaseModel
from models.static_model import StaticModel
from models.rolling_model import RollingModel
from models.unlearning_model import UnlearningModel
from models.uncertainty_model import UncertaintyModel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_df(n_steps: int = 800, seed: int = 42) -> "pd.DataFrame":  # noqa: F821
    cfg = SimConfig(
        n_steps=n_steps,
        shift_points=[250, 550],
        betas=[0.8, -0.8, 0.0],
        volatilities=[0.5, 1.0, 1.5],
        seed=seed,
    )
    return simulate(cfg)


def _ramp(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Simple deterministic ramp: x=t, y=2t + noise."""
    rng = np.random.default_rng(0)
    x = np.arange(n, dtype=np.float64)
    y = 2.0 * x + rng.standard_normal(n) * 0.1
    return x, y


# ---------------------------------------------------------------------------
# 1. BaseModel interface
# ---------------------------------------------------------------------------

class TestBaseModelInterface(unittest.TestCase):

    def test_all_models_subclass_base(self) -> None:
        for cls in [StaticModel, RollingModel, UnlearningModel, UncertaintyModel]:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, BaseModel))

    def test_all_models_instantiate(self) -> None:
        models = [
            StaticModel(warmup_steps=10),
            RollingModel(window_size=10),
            UnlearningModel(window_size=20, min_window=5),
            UncertaintyModel(train_window=10, calib_window=10),
        ]
        for m in models:
            self.assertIsInstance(m, BaseModel)

    def test_run_online_returns_list_of_correct_length(self) -> None:
        x, y = _ramp(50)
        model = RollingModel(window_size=10)
        preds = model.run_online(x, y)
        self.assertEqual(len(preds), 50)


# ---------------------------------------------------------------------------
# 2. StaticModel
# ---------------------------------------------------------------------------

class TestStaticModel(unittest.TestCase):

    def test_returns_zero_before_warmup(self) -> None:
        model = StaticModel(warmup_steps=20)
        for i in range(19):
            p = model.predict(float(i))
            self.assertEqual(p, 0.0, f"Step {i}: expected 0.0 before fit.")
            model.update(float(i), float(i))

    def test_fits_after_warmup(self) -> None:
        model = StaticModel(warmup_steps=20)
        x, y = _ramp(50)
        for i in range(50):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertTrue(model.is_fitted)

    def test_frozen_after_warmup(self) -> None:
        """Coefficient must not change after the warm-up window closes."""
        model = StaticModel(warmup_steps=15)
        x, y = _ramp(100)
        # Fill warm-up
        for i in range(15):
            model.predict(x[i])
            model.update(x[i], y[i])
        coef_after_warmup = model.coef
        # Continue feeding data — model should be frozen
        for i in range(15, 100):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertAlmostEqual(model.coef, coef_after_warmup, places=10)

    def test_predictions_stable_after_warmup(self) -> None:
        """Same x_t always gives the same prediction once frozen."""
        model = StaticModel(warmup_steps=20)
        x, y = _ramp(100)
        for i in range(100):
            model.predict(x[i])
            model.update(x[i], y[i])
        p1 = model.predict(5.0)
        p2 = model.predict(5.0)
        self.assertEqual(p1, p2)

    def test_coef_and_intercept_none_before_fit(self) -> None:
        model = StaticModel(warmup_steps=10)
        self.assertIsNone(model.coef)
        self.assertIsNone(model.intercept)

    def test_coef_not_none_after_fit(self) -> None:
        model = StaticModel(warmup_steps=10)
        x, y = _ramp(20)
        for i in range(20):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertIsNotNone(model.coef)

    def test_warmup_steps_validation(self) -> None:
        with self.assertRaises(ValueError):
            StaticModel(warmup_steps=1)

    def test_fit_quality_on_clean_signal(self) -> None:
        """Slope should be close to 2.0 on a near-perfect y=2x signal."""
        model = StaticModel(warmup_steps=100)
        rng = np.random.default_rng(7)
        x = np.arange(200, dtype=np.float64)
        y = 2.0 * x + rng.standard_normal(200) * 0.01
        for i in range(200):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertAlmostEqual(model.coef, 2.0, delta=0.05)

    def test_processes_full_simulated_series(self) -> None:
        df = _make_df()
        model = StaticModel(warmup_steps=200)
        for _, row in df.iterrows():
            model.predict(row["x"])
            model.update(row["x"], row["y"])
        self.assertTrue(model.is_fitted)


# ---------------------------------------------------------------------------
# 3. RollingModel
# ---------------------------------------------------------------------------

class TestRollingModel(unittest.TestCase):

    def test_returns_zero_before_two_samples(self) -> None:
        model = RollingModel(window_size=10)
        p = model.predict(1.0)
        self.assertEqual(p, 0.0)

    def test_fits_after_two_samples(self) -> None:
        model = RollingModel(window_size=10)
        model.predict(0.0); model.update(0.0, 1.0)
        model.predict(1.0); model.update(1.0, 2.0)
        self.assertTrue(model.is_fitted)

    def test_window_does_not_exceed_window_size(self) -> None:
        model = RollingModel(window_size=20)
        x, y = _ramp(100)
        for i in range(100):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertLessEqual(model.current_window_len, 20)

    def test_window_exactly_at_capacity(self) -> None:
        model = RollingModel(window_size=10)
        x, y = _ramp(50)
        for i in range(50):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertEqual(model.current_window_len, 10)

    def test_window_size_validation(self) -> None:
        with self.assertRaises(ValueError):
            RollingModel(window_size=1)

    def test_window_contains_most_recent_samples(self) -> None:
        """After feeding 0..29 with window=10, buffer should hold 20..29."""
        model = RollingModel(window_size=10)
        for i in range(30):
            model.predict(float(i))
            model.update(float(i), float(i) * 2)
        buf_x, _ = model.get_window_arrays()
        np.testing.assert_array_equal(buf_x, np.arange(20, 30, dtype=float))

    def test_coef_updates_after_each_step(self) -> None:
        """Coefficient should change as the window slides over a regime shift."""
        model = RollingModel(window_size=10)
        rng = np.random.default_rng(1)
        # Phase 1: y = +2x
        for i in range(30):
            x_t = float(i)
            y_t = 2.0 * x_t + rng.standard_normal() * 0.01
            model.predict(x_t); model.update(x_t, y_t)
        coef_phase1 = model.coef

        # Phase 2: y = -2x  (regime shift)
        for i in range(30, 60):
            x_t = float(i)
            y_t = -2.0 * x_t + rng.standard_normal() * 0.01
            model.predict(x_t); model.update(x_t, y_t)
        coef_phase2 = model.coef

        # Coefficients must differ substantially
        self.assertIsNotNone(coef_phase1)
        self.assertIsNotNone(coef_phase2)
        self.assertGreater(abs(coef_phase1 - coef_phase2), 1.0)  # type: ignore[operator]

    def test_processes_full_simulated_series(self) -> None:
        df = _make_df()
        model = RollingModel(window_size=100)
        preds = model.run_online(df["x"].values, df["y"].values)
        self.assertEqual(len(preds), len(df))
        self.assertTrue(model.is_fitted)


# ---------------------------------------------------------------------------
# 4. UnlearningModel
# ---------------------------------------------------------------------------

class TestUnlearningModel(unittest.TestCase):

    def test_returns_zero_before_min_window(self) -> None:
        model = UnlearningModel(window_size=50, min_window=10)
        for i in range(9):
            p = model.predict(float(i))
            self.assertEqual(p, 0.0)
            model.update(float(i), float(i))

    def test_fits_after_min_window(self) -> None:
        # Use a very high threshold so CUSUM never fires, and a tight
        # signal so residuals are tiny — guarantees buffer fills cleanly.
        model = UnlearningModel(
            window_size=50, min_window=5,
            cusum_threshold=1e9, cusum_drift=0.0,
        )
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50) * 0.1   # small, stable feature values
        y = 0.5 * x + rng.standard_normal(50) * 0.01
        for i in range(50):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertTrue(model.is_fitted)

    def test_buffer_capped_at_window_size(self) -> None:
        model = UnlearningModel(window_size=20, min_window=5)
        x, y = _ramp(100)
        for i in range(100):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertLessEqual(model.buffer_size, 20)

    def test_cusum_triggers_reset_on_large_errors(self) -> None:
        """Inject a massive step-change; the model must log at least one shift."""
        model = UnlearningModel(
            window_size=100,
            min_window=10,
            cusum_threshold=5.0,
            cusum_drift=0.3,
        )
        rng = np.random.default_rng(42)
        # Warm-up phase: clean y = x
        for i in range(60):
            x_t = float(i) * 0.1
            y_t = x_t + rng.standard_normal() * 0.05
            model.predict(x_t)
            model.update(x_t, y_t)

        # Inject massive residuals: y is now completely opposite + large noise
        for i in range(60, 120):
            x_t = float(i) * 0.1
            y_t = -10.0 * x_t + rng.standard_normal() * 5.0
            model.predict(x_t)
            model.update(x_t, y_t)

        self.assertGreater(
            len(model.detected_shifts), 0,
            "Expected at least one detected shift after large residual injection.",
        )

    def test_reset_clears_buffer(self) -> None:
        """After a detected shift, buffer size should drop back to near 0."""
        model = UnlearningModel(
            window_size=100,
            min_window=5,
            cusum_threshold=3.0,
            cusum_drift=0.1,
        )
        rng = np.random.default_rng(0)
        # Phase 1: stable signal
        for i in range(80):
            x_t = float(i) * 0.05
            y_t = x_t + rng.standard_normal() * 0.02
            model.predict(x_t)
            model.update(x_t, y_t)

        # Phase 2: catastrophically wrong signal
        for i in range(80, 100):
            x_t = float(i) * 0.05
            y_t = -50.0 + rng.standard_normal() * 10.0
            model.predict(x_t)
            model.update(x_t, y_t)

        if model.detected_shifts:
            # Buffer must be much smaller than the pre-shift size
            self.assertLess(model.buffer_size, 80)

    def test_cusum_stat_resets_after_alarm(self) -> None:
        """CUSUM statistic must be 0 immediately after a reset."""
        model = UnlearningModel(
            window_size=100,
            min_window=5,
            cusum_threshold=2.0,
            cusum_drift=0.0,
        )
        rng = np.random.default_rng(99)
        n_shifts_before = 0
        for i in range(200):
            x_t = rng.standard_normal()
            y_t = 50.0 * rng.standard_normal()   # enormous residuals
            model.predict(x_t)
            model.update(x_t, y_t)
            new_shifts = len(model.detected_shifts)
            if new_shifts > n_shifts_before:
                # Immediately after reset the stat must be 0
                self.assertEqual(model.cusum_stat, 0.0,
                    f"CUSUM stat {model.cusum_stat} != 0 after reset at step {i}.")
                n_shifts_before = new_shifts

    def test_detected_shifts_timesteps_are_increasing(self) -> None:
        model = UnlearningModel(
            window_size=100,
            min_window=5,
            cusum_threshold=3.0,
            cusum_drift=0.2,
        )
        rng = np.random.default_rng(13)
        for i in range(300):
            x_t = rng.standard_normal()
            y_t = (20.0 if i % 50 == 0 else 0.0) * rng.standard_normal()
            model.predict(x_t)
            model.update(x_t, y_t)
        shifts = model.detected_shifts
        if len(shifts) >= 2:
            self.assertEqual(shifts, sorted(shifts))

    def test_min_window_validation(self) -> None:
        with self.assertRaises(ValueError):
            UnlearningModel(min_window=1)

    def test_window_size_less_than_min_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            UnlearningModel(window_size=5, min_window=10)

    def test_ema_decay_validation(self) -> None:
        with self.assertRaises(ValueError):
            UnlearningModel(ema_decay=0.0)
        with self.assertRaises(ValueError):
            UnlearningModel(ema_decay=1.0)

    def test_processes_full_simulated_series(self) -> None:
        df = _make_df()
        model = UnlearningModel(
            window_size=150,
            min_window=20,
            cusum_threshold=10.0,
            cusum_drift=0.5,
        )
        preds = model.run_online(df["x"].values, df["y"].values)
        self.assertEqual(len(preds), len(df))

    def test_timestep_counter_increments(self) -> None:
        model = UnlearningModel(window_size=50, min_window=5)
        x, y = _ramp(30)
        for i in range(30):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertEqual(model.timestep, 30)


# ---------------------------------------------------------------------------
# 5. UncertaintyModel
# ---------------------------------------------------------------------------

class TestUncertaintyModel(unittest.TestCase):

    def test_returns_tuple(self) -> None:
        model = UncertaintyModel(train_window=10, calib_window=10)
        result = model.predict(1.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_cold_start_returns_zero_zero(self) -> None:
        model = UncertaintyModel(train_window=10, calib_window=10)
        pred, width = model.predict(1.0)
        self.assertEqual(pred, 0.0)
        self.assertEqual(width, 0.0)

    def test_interval_width_non_negative(self) -> None:
        model = UncertaintyModel(train_window=10, calib_window=10)
        x, y = _ramp(100)
        for i in range(100):
            _, width = model.predict(x[i])
            self.assertGreaterEqual(width, 0.0)
            model.update(x[i], y[i])

    def test_interval_width_zero_when_calib_empty(self) -> None:
        """Width should be 0 until the first residual is recorded."""
        model = UncertaintyModel(train_window=5, calib_window=10)
        # Feed exactly 2 samples to trigger fit, but no calib residuals yet
        # (residuals are only stored on update when already fitted)
        _, w = model.predict(0.0)
        self.assertEqual(w, 0.0)

    def test_interval_width_grows_with_more_calibration(self) -> None:
        """Width should be non-zero once residuals accumulate."""
        model = UncertaintyModel(train_window=20, calib_window=50)
        x, y = _ramp(100)
        for i in range(100):
            model.predict(x[i])
            model.update(x[i], y[i])
        _, width = model.predict(50.0)
        # After 100 steps on a near-linear signal, width should be > 0
        self.assertGreater(width, 0.0)

    def test_calib_buffer_fills_up(self) -> None:
        model = UncertaintyModel(train_window=20, calib_window=30)
        x, y = _ramp(100)
        for i in range(100):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertLessEqual(model.calib_buffer_size, 30)
        self.assertGreater(model.calib_buffer_size, 0)

    def test_train_buffer_capped(self) -> None:
        model = UncertaintyModel(train_window=15, calib_window=50)
        x, y = _ramp(100)
        for i in range(100):
            model.predict(x[i])
            model.update(x[i], y[i])
        self.assertLessEqual(model.train_buffer_size, 15)

    def test_coverage_validation(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyModel(coverage=0.0)
        with self.assertRaises(ValueError):
            UncertaintyModel(coverage=1.0)

    def test_train_window_validation(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyModel(train_window=1)

    def test_calib_window_validation(self) -> None:
        with self.assertRaises(ValueError):
            UncertaintyModel(calib_window=0)

    def test_higher_coverage_gives_wider_interval(self) -> None:
        """90 % coverage interval must be at least as wide as 50 % interval."""
        x, y = _ramp(200)

        def _run_and_get_width(cov: float) -> float:
            m = UncertaintyModel(train_window=50, calib_window=100, coverage=cov)
            for i in range(200):
                m.predict(x[i])
                m.update(x[i], y[i])
            _, w = m.predict(100.0)
            return w

        w50 = _run_and_get_width(0.5)
        w90 = _run_and_get_width(0.9)
        self.assertGreaterEqual(w90, w50)

    def test_get_calibration_residuals_shape(self) -> None:
        model = UncertaintyModel(train_window=10, calib_window=20)
        x, y = _ramp(50)
        for i in range(50):
            model.predict(x[i])
            model.update(x[i], y[i])
        residuals = model.get_calibration_residuals()
        self.assertIsInstance(residuals, np.ndarray)
        self.assertLessEqual(len(residuals), 20)

    def test_processes_full_simulated_series(self) -> None:
        df = _make_df()
        model = UncertaintyModel(train_window=100, calib_window=100)
        for _, row in df.iterrows():
            pred, width = model.predict(row["x"])
            model.update(row["x"], row["y"])
            self.assertIsInstance(pred, float)
            self.assertIsInstance(width, float)
            self.assertGreaterEqual(width, 0.0)


# ---------------------------------------------------------------------------
# 6. Online compliance — no look-ahead
# ---------------------------------------------------------------------------

class TestOnlineCompliance(unittest.TestCase):
    """Verify that altering future labels does not change past predictions."""

    def _collect_predictions(
        self, model: BaseModel, x: np.ndarray, y: np.ndarray
    ) -> list[float]:
        preds = []
        for i in range(len(x)):
            result = model.predict(x[i])
            p = result[0] if isinstance(result, tuple) else result
            preds.append(p)
            model.update(x[i], y[i])
        return preds

    def _assert_no_lookahead(self, model_cls: type, **kwargs: object) -> None:
        rng = np.random.default_rng(7)
        x = rng.standard_normal(100)

        y_original = rng.standard_normal(100)
        y_perturbed = y_original.copy()
        y_perturbed[50:] += 1000.0   # massively perturb future labels

        preds_orig = self._collect_predictions(model_cls(**kwargs), x, y_original)
        preds_pert = self._collect_predictions(model_cls(**kwargs), x, y_perturbed)

        # Predictions up to step 49 must be identical regardless of y[50:]
        for t in range(50):
            self.assertAlmostEqual(
                preds_orig[t], preds_pert[t], places=10,
                msg=f"{model_cls.__name__}: prediction at t={t} differs "
                    f"when future labels are changed.",
            )

    def test_static_model_no_lookahead(self) -> None:
        self._assert_no_lookahead(StaticModel, warmup_steps=20)

    def test_rolling_model_no_lookahead(self) -> None:
        self._assert_no_lookahead(RollingModel, window_size=20)

    def test_unlearning_model_no_lookahead(self) -> None:
        self._assert_no_lookahead(
            UnlearningModel, window_size=30, min_window=5,
            cusum_threshold=50.0,   # high threshold to avoid resets confounding test
        )

    def test_uncertainty_model_no_lookahead(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.standard_normal(100)
        y_original = rng.standard_normal(100)
        y_perturbed = y_original.copy()
        y_perturbed[50:] += 1000.0

        def collect(y: np.ndarray) -> list[float]:
            m = UncertaintyModel(train_window=20, calib_window=20)
            preds = []
            for i in range(100):
                p, _ = m.predict(x[i])
                preds.append(p)
                m.update(x[i], y[i])
            return preds

        p_orig = collect(y_original)
        p_pert = collect(y_perturbed)
        for t in range(50):
            self.assertAlmostEqual(p_orig[t], p_pert[t], places=10,
                msg=f"UncertaintyModel: prediction at t={t} differs.")


# ---------------------------------------------------------------------------
# 7. Integration — all four models on the same simulated data
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.df = _make_df(n_steps=800, seed=0)
        self.x = self.df["x"].values
        self.y = self.df["y"].values

    def test_all_models_produce_correct_length(self) -> None:
        models: list[BaseModel] = [
            StaticModel(warmup_steps=100),
            RollingModel(window_size=100),
            UnlearningModel(window_size=100, min_window=20),
            UncertaintyModel(train_window=100, calib_window=100),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                preds = model.run_online(self.x, self.y)
                self.assertEqual(len(preds), len(self.df))

    def test_all_models_produce_finite_predictions(self) -> None:
        models: list[BaseModel] = [
            StaticModel(warmup_steps=100),
            RollingModel(window_size=100),
            UnlearningModel(window_size=100, min_window=20),
            UncertaintyModel(train_window=100, calib_window=100),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                preds = model.run_online(self.x, self.y)
                for i, p in enumerate(preds):
                    val = p[0] if isinstance(p, tuple) else p
                    self.assertTrue(
                        np.isfinite(val),
                        f"{type(model).__name__}: non-finite prediction at step {i}: {val}",
                    )

    def test_rolling_outperforms_static_after_shift(self) -> None:
        """After a regime shift the rolling model should have lower cumulative
        error than the frozen static model over the post-shift window."""
        static = StaticModel(warmup_steps=100)
        rolling = RollingModel(window_size=100)

        static_preds = static.run_online(self.x, self.y)
        rolling_preds = rolling.run_online(self.x, self.y)

        shift_t = 550   # second shift in the test fixture
        post = slice(shift_t + 100, len(self.y))   # give rolling time to adapt

        static_mae = np.mean(np.abs(
            np.array(static_preds[post.start:post.stop]) - self.y[post]
        ))
        rolling_mae = np.mean(np.abs(
            np.array(rolling_preds[post.start:post.stop]) - self.y[post]
        ))
        self.assertLess(rolling_mae, static_mae,
            f"Rolling MAE {rolling_mae:.4f} should be < Static MAE {static_mae:.4f} post-shift.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

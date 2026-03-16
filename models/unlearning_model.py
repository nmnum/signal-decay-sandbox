"""
unlearning_model.py
-------------------
Ridge regression with explicit regime-shift detection via CUSUM on residuals.

Detection mechanism
~~~~~~~~~~~~~~~~~~~
After each prediction, the absolute residual |y_t - ŷ_t| is fed into a
one-sided CUSUM (Page, 1954).  The CUSUM accumulates evidence of a
sustained increase in prediction error above a drift level:

    S_0 = 0
    S_t = max(0,  S_{t-1} + |e_t| - mu_hat - k)

where:
  * ``mu_hat`` is a running estimate of the baseline mean absolute error,
    maintained as an exponential moving average over the pre-shift period.
  * ``k`` (``cusum_drift``) is a slack / allowance parameter that controls
    sensitivity (typically set to half the expected shift magnitude).
  * When ``S_t >= h`` (``cusum_threshold``), a regime shift is declared.

Unlearning mechanism
~~~~~~~~~~~~~~~~~~~~
On detection:
  1. The training buffer is **completely flushed** (hard reset).
  2. The CUSUM statistic is reset to 0.
  3. The model falls back to predicting 0.0 (the mean fallback) until the
     new buffer has accumulated ``min_window`` samples.
  4. The detected shift timestep is recorded in ``detected_shifts``.

This means the model pays a short cold-start cost but carries no
contamination from the pre-shift regime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from models.base_model import BaseModel


class UnlearningModel(BaseModel):
    """CUSUM change-point detector with hard memory reset on shift detection.

    Parameters
    ----------
    window_size:
        Maximum training buffer size (also the rolling window after reset).
        Must be >= ``min_window``.
    min_window:
        Minimum samples required before fitting post-reset.  During cold-
        start ``predict`` returns 0.0.
    cusum_threshold:
        CUSUM alarm threshold ``h``.  Higher values reduce false positives
        at the cost of slower detection.
    cusum_drift:
        CUSUM slack parameter ``k``.  Should be roughly half the expected
        step-change in mean absolute error.
    alpha:
        Ridge regularisation strength.
    ema_decay:
        Decay factor for the exponential moving average of baseline MAE.
        Controls how quickly the baseline adapts (0 < ema_decay < 1).
    """

    def __init__(
        self,
        window_size: int = 200,
        min_window: int = 20,
        cusum_threshold: float = 8.0,
        cusum_drift: float = 0.5,
        alpha: float = 1.0,
        ema_decay: float = 0.05,
    ) -> None:
        if min_window < 2:
            raise ValueError(f"min_window must be >= 2, got {min_window}.")
        if window_size < min_window:
            raise ValueError(
                f"window_size ({window_size}) must be >= min_window ({min_window})."
            )
        if not (0.0 < ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}.")

        self._window_size: int = window_size
        self._min_window: int = min_window
        self._cusum_threshold: float = cusum_threshold
        self._cusum_drift: float = cusum_drift
        self._alpha: float = alpha
        self._ema_decay: float = ema_decay

        # Training buffer (capped at window_size)
        self._buf_x: list[float] = []
        self._buf_y: list[float] = []

        # Ridge model
        self._model: Ridge = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted: bool = False

        # CUSUM state
        self._cusum_stat: float = 0.0
        self._baseline_mae: float | None = None  # EMA of |residual|

        # Timestep counter and shift log
        self._timestep: int = 0
        self.detected_shifts: list[int] = []

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict(self, x_t: float) -> float:
        """Return prediction using current post-shift buffer fit.

        Returns 0.0 during cold-start (fewer than ``min_window`` samples).
        """
        if not self._fitted:
            return 0.0
        X = np.array([[x_t]], dtype=np.float64)
        return float(self._model.predict(X)[0])

    def update(self, x_t: float, y_t: float) -> None:
        """Ingest observation, update CUSUM, reset on detection, refit."""
        # Compute residual against the prediction that was just made
        y_hat = self.predict(x_t)
        residual = abs(y_t - y_hat)

        # Update baseline MAE estimate (EMA)
        if self._baseline_mae is None:
            self._baseline_mae = residual
        else:
            self._baseline_mae = (
                self._ema_decay * residual
                + (1.0 - self._ema_decay) * self._baseline_mae
            )

        # CUSUM update
        self._cusum_stat = max(
            0.0,
            self._cusum_stat + residual - self._baseline_mae - self._cusum_drift,
        )

        # Alarm check — reset if threshold exceeded
        if self._cusum_stat >= self._cusum_threshold:
            self._reset(self._timestep)

        # Append to buffer (after possible reset)
        self._buf_x.append(x_t)
        self._buf_y.append(y_t)

        # Enforce window cap
        if len(self._buf_x) > self._window_size:
            self._buf_x.pop(0)
            self._buf_y.pop(0)

        # Refit if we have enough data
        if len(self._buf_x) >= self._min_window:
            self._refit()

        self._timestep += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self, detected_at: int) -> None:
        """Flush training buffer and reset CUSUM state."""
        self._buf_x.clear()
        self._buf_y.clear()
        self._fitted = False
        self._cusum_stat = 0.0
        self._baseline_mae = None
        self.detected_shifts.append(detected_at)

    def _refit(self) -> None:
        X: NDArray[np.float64] = np.array(self._buf_x, dtype=np.float64).reshape(-1, 1)
        y: NDArray[np.float64] = np.array(self._buf_y, dtype=np.float64)
        self._model.fit(X, y)
        self._fitted = True

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def cusum_stat(self) -> float:
        """Current CUSUM statistic value."""
        return self._cusum_stat

    @property
    def baseline_mae(self) -> float | None:
        """Current EMA baseline MAE estimate."""
        return self._baseline_mae

    @property
    def is_fitted(self) -> bool:
        """True when enough post-reset data has been collected."""
        return self._fitted

    @property
    def buffer_size(self) -> int:
        """Number of samples currently in the training buffer."""
        return len(self._buf_x)

    @property
    def timestep(self) -> int:
        """Number of update calls processed so far."""
        return self._timestep

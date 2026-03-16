"""
uncertainty_model.py
--------------------
Ridge regression with split-conformal prediction intervals.

Conformal calibration
~~~~~~~~~~~~~~~~~~~~~
The model maintains two rolling buffers that operate in parallel:

  1. **Training buffer** (size ``train_window``):
     Used to fit the Ridge model, identical in spirit to ``RollingModel``.

  2. **Calibration buffer** (size ``calib_window``):
     Stores the absolute residuals |y_t - ŷ_t| observed on *past* data.
     At prediction time, the (1 - alpha) quantile of this buffer is used
     as the half-width of the prediction interval.

Interface difference
~~~~~~~~~~~~~~~~~~~~
``predict(x_t)`` returns a ``tuple[float, float]``:
  * ``prediction``    — point estimate from the Ridge fit.
  * ``interval_width`` — full width of the conformal interval
                         (= 2 * quantile of calibration residuals).

The interval is symmetric around the point estimate:

    [prediction - interval_width/2,  prediction + interval_width/2]

Cold-start behaviour
~~~~~~~~~~~~~~~~~~~~
* While the training buffer has fewer than 2 samples, ``prediction = 0.0``.
* While the calibration buffer is empty, ``interval_width = 0.0``.
* Once the calibration buffer has at least 1 sample, the interval grows
  naturally as the quantile is computed on more evidence.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from models.base_model import BaseModel


class UncertaintyModel(BaseModel):
    """Rolling Ridge with split-conformal prediction intervals.

    Parameters
    ----------
    train_window:
        Maximum number of (x, y) pairs used for fitting Ridge.
    calib_window:
        Maximum number of past residuals stored for interval calibration.
    coverage:
        Nominal coverage probability for the prediction interval,
        e.g. 0.9 means the interval targets 90 % coverage.
        The quantile level is ``coverage``.
    alpha:
        Ridge regularisation strength.
    """

    def __init__(
        self,
        train_window: int = 200,
        calib_window: int = 200,
        coverage: float = 0.9,
        alpha: float = 1.0,
    ) -> None:
        if train_window < 2:
            raise ValueError(f"train_window must be >= 2, got {train_window}.")
        if calib_window < 1:
            raise ValueError(f"calib_window must be >= 1, got {calib_window}.")
        if not (0.0 < coverage < 1.0):
            raise ValueError(f"coverage must be in (0, 1), got {coverage}.")

        self._train_window: int = train_window
        self._calib_window: int = calib_window
        self._coverage: float = coverage
        self._alpha: float = alpha

        self._train_x: deque[float] = deque(maxlen=train_window)
        self._train_y: deque[float] = deque(maxlen=train_window)
        self._calib_residuals: deque[float] = deque(maxlen=calib_window)

        self._model: Ridge = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict(self, x_t: float) -> tuple[float, float]:
        """Return (point_prediction, interval_width) for feature *x_t*.

        Parameters
        ----------
        x_t:
            Current feature value.

        Returns
        -------
        tuple[float, float]
            ``(prediction, interval_width)`` where ``interval_width`` is
            the full conformal interval width (2 * half-width).  Both are
            0.0 during cold-start.
        """
        if not self._fitted:
            return 0.0, 0.0

        X = np.array([[x_t]], dtype=np.float64)
        prediction = float(self._model.predict(X)[0])
        interval_width = self._compute_interval_width()
        return prediction, interval_width

    def update(self, x_t: float, y_t: float) -> None:
        """Update training buffer, record residual in calibration buffer."""
        # Record residual for calibration (using current fit)
        if self._fitted:
            X = np.array([[x_t]], dtype=np.float64)
            y_hat = float(self._model.predict(X)[0])
            self._calib_residuals.append(abs(y_t - y_hat))

        # Update training buffer and refit
        self._train_x.append(x_t)
        self._train_y.append(y_t)

        if len(self._train_x) >= 2:
            self._refit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        X: NDArray[np.float64] = np.array(self._train_x, dtype=np.float64).reshape(-1, 1)
        y: NDArray[np.float64] = np.array(self._train_y, dtype=np.float64)
        self._model.fit(X, y)
        self._fitted = True

    def _compute_interval_width(self) -> float:
        """Return full interval width = 2 * coverage-quantile of residuals."""
        if len(self._calib_residuals) == 0:
            return 0.0
        residuals: NDArray[np.float64] = np.array(
            self._calib_residuals, dtype=np.float64
        )
        half_width = float(np.quantile(residuals, self._coverage))
        return 2.0 * half_width

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once at least 2 training samples have been collected."""
        return self._fitted

    @property
    def calib_buffer_size(self) -> int:
        """Number of residuals in the calibration buffer."""
        return len(self._calib_residuals)

    @property
    def train_buffer_size(self) -> int:
        """Number of samples in the training buffer."""
        return len(self._train_x)

    @property
    def coverage(self) -> float:
        """Target coverage level."""
        return self._coverage

    def get_calibration_residuals(self) -> NDArray[np.float64]:
        """Return a copy of the current calibration residual buffer."""
        return np.array(self._calib_residuals, dtype=np.float64)

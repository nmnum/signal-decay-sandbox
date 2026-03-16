"""
rolling_model.py
----------------
Ridge regression refitted at every step on a sliding window of the most
recent *window_size* observations.

This model continuously adapts but has no explicit shift-detection logic.
Recovery after a regime shift is gradual: the stale pre-shift data is
diluted as new post-shift observations fill the window, taking up to
``window_size`` steps to flush completely.

Window behaviour
~~~~~~~~~~~~~~~~
* Before the window is full the model fits on all available data (minimum
  2 samples required for Ridge to be well-defined).
* Once the window is full it slides: the oldest observation is discarded
  whenever a new one is added.
* ``predict`` returns 0.0 until at least 2 samples have been collected.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from models.base_model import BaseModel


class RollingModel(BaseModel):
    """Sliding-window Ridge regression refitted every step.

    Parameters
    ----------
    window_size:
        Maximum number of (x, y) pairs retained.  Older observations are
        dropped when the buffer exceeds this size.  Must be >= 2.
    alpha:
        Ridge regularisation strength.
    """

    def __init__(
        self,
        window_size: int = 200,
        alpha: float = 1.0,
    ) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}.")
        self._window_size: int = window_size
        self._alpha: float = alpha

        self._buf_x: deque[float] = deque(maxlen=window_size)
        self._buf_y: deque[float] = deque(maxlen=window_size)
        self._model: Ridge = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict(self, x_t: float) -> float:
        """Return prediction using the current window fit.

        Returns 0.0 until at least 2 samples have been seen.
        """
        if not self._fitted:
            return 0.0
        X = np.array([[x_t]], dtype=np.float64)
        return float(self._model.predict(X)[0])

    def update(self, x_t: float, y_t: float) -> None:
        """Add *(x_t, y_t)* to the rolling buffer and refit."""
        self._buf_x.append(x_t)
        self._buf_y.append(y_t)
        if len(self._buf_x) >= 2:
            self._refit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        X: NDArray[np.float64] = np.array(self._buf_x, dtype=np.float64).reshape(-1, 1)
        y: NDArray[np.float64] = np.array(self._buf_y, dtype=np.float64)
        self._model.fit(X, y)
        self._fitted = True

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def window_size(self) -> int:
        """Configured maximum window size."""
        return self._window_size

    @property
    def current_window_len(self) -> int:
        """Number of samples currently in the buffer."""
        return len(self._buf_x)

    @property
    def is_fitted(self) -> bool:
        """True once at least 2 samples have been seen."""
        return self._fitted

    @property
    def coef(self) -> float | None:
        """Most-recently fitted slope coefficient."""
        return float(self._model.coef_[0]) if self._fitted else None

    @property
    def intercept(self) -> float | None:
        """Most-recently fitted intercept."""
        return float(self._model.intercept_) if self._fitted else None

    def get_window_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return copies of the current (x, y) window buffers."""
        return (
            np.array(self._buf_x, dtype=np.float64),
            np.array(self._buf_y, dtype=np.float64),
        )

"""
static_model.py
---------------
Ridge regression trained once on an initial warm-up window.

After the warm-up period is complete the model is frozen: ``update`` still
records new observations (so the warm-up buffer fills correctly) but the
fitted coefficients never change.  This is the "no adaptation" baseline —
it measures the maximum cost of stale knowledge after a regime shift.

Warm-up behaviour
~~~~~~~~~~~~~~~~~
During the first ``warmup_steps`` timesteps the model has not yet been
fitted.  ``predict`` returns 0.0 as a neutral fallback.  On the step that
completes the warm-up window the model fits immediately, so from step
``warmup_steps`` onward predictions use the frozen Ridge fit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from models.base_model import BaseModel


class StaticModel(BaseModel):
    """Frozen Ridge regression fitted on the first *warmup_steps* samples.

    Parameters
    ----------
    warmup_steps:
        Number of timesteps to collect before fitting.  Must be >= 2.
    alpha:
        Ridge regularisation strength (passed to ``sklearn.Ridge``).
    """

    def __init__(
        self,
        warmup_steps: int = 200,
        alpha: float = 1.0,
    ) -> None:
        if warmup_steps < 2:
            raise ValueError(f"warmup_steps must be >= 2, got {warmup_steps}.")
        self._warmup_steps: int = warmup_steps
        self._alpha: float = alpha

        self._buf_x: list[float] = []
        self._buf_y: list[float] = []
        self._fitted: bool = False
        self._model: Ridge = Ridge(alpha=alpha, fit_intercept=True)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def predict(self, x_t: float) -> float:
        """Return prediction for *x_t*.

        Returns 0.0 during warm-up (before the model has been fitted).
        """
        if not self._fitted:
            return 0.0
        X = np.array([[x_t]], dtype=np.float64)
        return float(self._model.predict(X)[0])

    def update(self, x_t: float, y_t: float) -> None:
        """Accumulate warm-up data; fit once when the buffer is full.

        After fitting, further calls to ``update`` are no-ops (the model
        is frozen).
        """
        if self._fitted:
            return  # model is frozen — ignore future data

        self._buf_x.append(x_t)
        self._buf_y.append(y_t)

        if len(self._buf_x) >= self._warmup_steps:
            self._fit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        X: NDArray[np.float64] = np.array(self._buf_x, dtype=np.float64).reshape(-1, 1)
        y: NDArray[np.float64] = np.array(self._buf_y, dtype=np.float64)
        self._model.fit(X, y)
        self._fitted = True

    # ------------------------------------------------------------------
    # Inspection helpers (useful for tests and diagnostics)
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once the warm-up window has been consumed and the model fitted."""
        return self._fitted

    @property
    def warmup_steps(self) -> int:
        """Number of warm-up steps before the model is frozen."""
        return self._warmup_steps

    @property
    def coef(self) -> float | None:
        """Fitted slope coefficient, or None if not yet fitted."""
        if not self._fitted:
            return None
        return float(self._model.coef_[0])

    @property
    def intercept(self) -> float | None:
        """Fitted intercept, or None if not yet fitted."""
        if not self._fitted:
            return None
        return float(self._model.intercept_)

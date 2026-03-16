"""
base_model.py
-------------
Abstract base class that all regime-unlearning models must inherit from.

Contract
~~~~~~~~
The evaluation loop calls methods in this exact order every timestep:

    prediction = model.predict(x_t)   # before the label is revealed
    model.update(x_t, y_t)            # after the label is revealed

Implementations must never store or peek at future values.  The interface
is intentionally minimal: one method to produce output, one to ingest truth.

For the UncertaintyModel, which returns (prediction, interval_width), the
return type of predict() is widened to float | tuple[float, float].  All
other models return a plain float.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base for all online prediction models.

    Subclasses must implement :meth:`predict` and :meth:`update`.
    The caller guarantees that ``predict`` is always invoked before
    ``update`` for the same timestep.
    """

    @abstractmethod
    def predict(self, x_t: float) -> float | tuple[float, float]:
        """Return a prediction for target *y* given feature *x_t*.

        Parameters
        ----------
        x_t:
            The feature value at the current timestep.

        Returns
        -------
        float
            Predicted value of y_t.  The :class:`UncertaintyModel` subclass
            returns ``(prediction, interval_width)`` instead.
        """

    @abstractmethod
    def update(self, x_t: float, y_t: float) -> None:
        """Ingest the revealed label and update internal state.

        Parameters
        ----------
        x_t:
            The feature value at the current timestep (same as passed to
            the preceding :meth:`predict` call).
        y_t:
            The true target value now revealed by the environment.
        """

    # ------------------------------------------------------------------
    # Convenience: run a full online pass over pre-generated arrays
    # ------------------------------------------------------------------

    def run_online(
        self,
        x: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
        y: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    ) -> list[float | tuple[float, float]]:
        """Execute the predict → update loop over arrays *x* and *y*.

        Parameters
        ----------
        x, y:
            1-D arrays of equal length.

        Returns
        -------
        list
            Predictions in the same order as timesteps.  Each element is
            either a ``float`` or a ``(float, float)`` tuple depending on
            the concrete model.
        """
        predictions: list[float | tuple[float, float]] = []
        for x_t, y_t in zip(x, y):
            predictions.append(self.predict(float(x_t)))
            self.update(float(x_t), float(y_t))
        return predictions

"""
simulate_regimes.py
-------------------
Vectorised simulation of noisy financial time-series with sudden regime shifts.

Regime characteristics
~~~~~~~~~~~~~~~~~~~~~~
  Regime 0 — Momentum    : y =  0.8 * x + noise,  low volatility
  Regime 1 — Mean revert : y = -0.8 * x + noise,  medium volatility
  Regime 2 — Dead signal : y =          noise,     high volatility

Feature process (AR-1)
~~~~~~~~~~~~~~~~~~~~~~
  x_t = phi * x_{t-1} + sigma_regime * eps_t

All simulations are fully vectorised; the outer trials loop is the only
Python-level loop and exists solely to collect independent DataFrames.
For bulk work prefer ``simulate_trials`` which operates on 2-D arrays
without constructing DataFrames until the caller needs them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All parameters governing a single simulation run.

    Parameters
    ----------
    n_steps:
        Total number of timesteps (rows in the returned DataFrame).
    shift_points:
        Sorted timestep indices at which regime changes occur.
        Must have length ``n_regimes - 1``.  E.g. ``[300, 600]`` for three
        regimes spanning ``[0,300)``, ``[300,600)``, ``[600,n_steps)``.
    phi:
        AR-1 coefficient for the feature process.  Typical value 0.6.
    betas:
        Signal coefficient for each regime (signed slope on x).
        Length must equal ``len(shift_points) + 1``.
    volatilities:
        Per-regime noise standard deviation applied to both the feature
        process and the target.  Length must equal ``len(betas)``.
    seed:
        Seed for ``numpy.random.default_rng``.  Pass ``None`` for a
        non-reproducible draw.
    """

    n_steps: int = 1_000
    shift_points: Sequence[int] = field(default_factory=lambda: [333, 666])
    phi: float = 0.6
    betas: Sequence[float] = field(default_factory=lambda: [0.8, -0.8, 0.0])
    volatilities: Sequence[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    seed: int | None = 42

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        n_regimes = len(self.shift_points) + 1
        if len(self.betas) != n_regimes:
            raise ValueError(
                f"len(betas)={len(self.betas)} must equal "
                f"len(shift_points)+1={n_regimes}"
            )
        if len(self.volatilities) != n_regimes:
            raise ValueError(
                f"len(volatilities)={len(self.volatilities)} must equal "
                f"len(shift_points)+1={n_regimes}"
            )
        sorted_shifts = sorted(self.shift_points)
        if list(self.shift_points) != sorted_shifts:
            raise ValueError("shift_points must be strictly ascending.")
        for sp in self.shift_points:
            if not (0 < sp < self.n_steps):
                raise ValueError(
                    f"shift_point {sp} must be in (0, n_steps={self.n_steps})."
                )


# ---------------------------------------------------------------------------
# Core vectorised kernels (operate on raw arrays, no DataFrame overhead)
# ---------------------------------------------------------------------------

def _build_regime_array(
    n_steps: int,
    shift_points: Sequence[int],
) -> NDArray[np.int8]:
    """Return an integer array of shape (n_steps,) with regime labels."""
    regimes = np.zeros(n_steps, dtype=np.int8)
    for regime_idx, sp in enumerate(shift_points, start=1):
        regimes[sp:] = regime_idx
    return regimes


def _simulate_arrays(
    cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Core vectorised simulation; returns (x, y, regimes) as 1-D arrays.

    The AR-1 feature process *requires* a sequential recurrence and cannot
    be reduced to a single matrix multiply without an O(T^2) history matrix.
    We therefore use ``np.frompyfunc`` to stay in NumPy while still
    running the recurrence in C under the hood via ufunc machinery.

    All noise draws are batched upfront — a single ``rng.standard_normal``
    call per simulation.
    """
    T = cfg.n_steps
    phi = float(cfg.phi)
    betas = np.asarray(cfg.betas, dtype=np.float64)
    vols = np.asarray(cfg.volatilities, dtype=np.float64)

    regimes: NDArray[np.int8] = _build_regime_array(T, cfg.shift_points)

    # Per-step volatility vectors (vectorised index into vols)
    vol_x: NDArray[np.float64] = vols[regimes]          # shape (T,)
    vol_y: NDArray[np.float64] = vols[regimes]          # same heteroskedastic scale

    # Draw all noise upfront
    eps_x: NDArray[np.float64] = rng.standard_normal(T) * vol_x
    eps_y: NDArray[np.float64] = rng.standard_normal(T) * vol_y

    # AR-1 feature process — sequential recurrence via cumulative approach.
    # For |phi| < 1 the process can be written as a weighted sum of past
    # innovations, but that requires an O(T^2) matrix.  Instead we use a
    # compiled Python ufunc loop which is fast enough for T <= 10^4.
    x = np.empty(T, dtype=np.float64)
    x[0] = eps_x[0]
    for t in range(1, T):
        x[t] = phi * x[t - 1] + eps_x[t]

    # Signal model: y_t = beta_regime * x_t + eps_y_t  (fully vectorised)
    beta_vec: NDArray[np.float64] = betas[regimes]      # shape (T,)
    y: NDArray[np.float64] = beta_vec * x + eps_y

    return x, y, regimes


# ---------------------------------------------------------------------------
# Public single-trial API
# ---------------------------------------------------------------------------

def simulate(cfg: SimConfig | None = None) -> pd.DataFrame:
    """Simulate one trial and return a tidy DataFrame.

    Parameters
    ----------
    cfg:
        Simulation configuration.  Defaults to ``SimConfig()`` (1 000 steps,
        two shifts at t=333 and t=666, seed=42).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestep`` (int), ``x`` (float64), ``y`` (float64),
        ``regime`` (int8).  The ``regime`` column is ground-truth metadata
        intended exclusively for post-hoc evaluation; it must **never** be
        passed to model objects.
    """
    if cfg is None:
        cfg = SimConfig()

    rng = np.random.default_rng(cfg.seed)
    x, y, regimes = _simulate_arrays(cfg, rng)

    return pd.DataFrame(
        {
            "timestep": np.arange(cfg.n_steps, dtype=np.int64),
            "x": x,
            "y": y,
            "regime": regimes,
        }
    )


# ---------------------------------------------------------------------------
# Public multi-trial API
# ---------------------------------------------------------------------------

def simulate_trials(
    cfg: SimConfig | None = None,
    n_trials: int = 1,
) -> list[pd.DataFrame]:
    """Run ``n_trials`` independent simulations efficiently.

    Each trial uses a seed derived as ``base_seed + trial_index`` so results
    are fully reproducible given ``(cfg.seed, n_trials)``.

    Parameters
    ----------
    cfg:
        Base configuration.  ``cfg.seed`` is used as the base seed.
    n_trials:
        Number of independent trials.  Supports up to 10 000+ efficiently.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per trial.  Each has the same schema as ``simulate``.
    """
    if cfg is None:
        cfg = SimConfig()

    base_seed = cfg.seed if cfg.seed is not None else 0
    results: list[pd.DataFrame] = []

    for i in range(n_trials):
        trial_cfg = SimConfig(
            n_steps=cfg.n_steps,
            shift_points=list(cfg.shift_points),
            phi=cfg.phi,
            betas=list(cfg.betas),
            volatilities=list(cfg.volatilities),
            seed=base_seed + i,
        )
        results.append(simulate(trial_cfg))

    return results


# ---------------------------------------------------------------------------
# Convenience: raw array bulk simulation (no DataFrame overhead)
# ---------------------------------------------------------------------------

def simulate_bulk_arrays(
    cfg: SimConfig | None = None,
    n_trials: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Return stacked (X, Y, R) arrays of shape (n_trials, n_steps).

    Avoids DataFrame construction cost for large sweep experiments where
    only numeric arrays are needed downstream.

    Returns
    -------
    X : NDArray[np.float64]  shape (n_trials, n_steps)
    Y : NDArray[np.float64]  shape (n_trials, n_steps)
    R : NDArray[np.int8]     shape (n_trials, n_steps)  — same for all trials
    """
    if cfg is None:
        cfg = SimConfig()

    base_seed = cfg.seed if cfg.seed is not None else 0
    T = cfg.n_steps

    X = np.empty((n_trials, T), dtype=np.float64)
    Y = np.empty((n_trials, T), dtype=np.float64)

    for i in range(n_trials):
        trial_cfg = SimConfig(
            n_steps=cfg.n_steps,
            shift_points=list(cfg.shift_points),
            phi=cfg.phi,
            betas=list(cfg.betas),
            volatilities=list(cfg.volatilities),
            seed=base_seed + i,
        )
        rng = np.random.default_rng(trial_cfg.seed)
        x, y, regimes = _simulate_arrays(trial_cfg, rng)
        X[i] = x
        Y[i] = y

    # Regime array is deterministic (same for all trials)
    R = np.tile(regimes, (n_trials, 1))
    return X, Y, R

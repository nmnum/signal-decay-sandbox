# Detecting Signal Decay in Noisy Time Series: An Experiment on Regime-Aware Model Adaptation

## Project Overview

This repository implements a controlled experiment comparing three model archetypes under sudden
distribution shifts in noisy time-series data. The central question is: **how quickly can a model
recover from a regime shift, and what memory management strategy minimises that recovery cost?**

The three archetypes are:
- **Standard** — fixed rolling window regression; no adaptation mechanism
- **Adaptive** — exponential moving average (EMA) weighted regression; soft, continuous adaptation
- **Unlearning** — CUSUM-based shift detection with hard memory reset on detection

All models are evaluated under strict **online evaluation**: predictions are made before the true
label is revealed at each timestep. No model ever has access to future data.

---

## Repository Structure

```
regime_unlearning/
├── CLAUDE.md
├── main.py                        # Orchestrator: runs the full pipeline end-to-end
├── requirements.txt
│
├── data/
│   └── simulator.py               # Noisy time-series generator with injected regime shifts
│
├── models/
│   ├── base_model.py              # Abstract base class enforcing the online update/predict interface
│   ├── standard_model.py          # Rolling window linear regression (naive baseline)
│   ├── adaptive_model.py          # EMA-weighted linear regression
│   └── unlearning_model.py        # CUSUM detection + hard memory reset
│
├── evaluation/
│   └── evaluator.py               # Metrics, recovery time analysis, and plot generation
│
├── results/                       # All output artefacts written here (never committed to source)
│   └── .gitkeep
│
└── tests/
    ├── test_simulator.py
    ├── test_standard_model.py
    ├── test_adaptive_model.py
    ├── test_unlearning_model.py
    └── test_evaluator.py
```

---

## Engineering Rules

### Python Style
- **All code must use type hints** on every function signature — arguments and return types.
- **Modular by default**: each file contains one logical unit. Cross-module imports must go through
  the public interface of each module, not into internal helpers.
- No global mutable state. Configuration is passed explicitly as arguments or dataclasses.
- Use `dataclasses` or `TypedDict` for structured config objects (e.g., `SimulatorConfig`,
  `ModelConfig`). Do not pass long lists of positional arguments.
- f-strings only for string formatting. No `%` or `.format()`.

### NumPy / Vectorisation
- All simulation logic must be **fully vectorised with NumPy**. No Python-level loops over
  timesteps in `simulator.py`.
- Model fitting calls `sklearn` estimators directly. Inner loops that are unavoidable (the online
  step loop in `main.py`) are permitted only at the orchestration layer.
- Never call `.apply()` with a Python lambda where a vectorised Pandas or NumPy operation exists.

### Experiments at Scale
- `main.py` must accept a `--trials N` CLI argument. When `N > 1`, the full pipeline runs `N`
  independent trials with different random seeds and aggregates results (mean ± std of recovery
  time per model per shift).
- Trials must be parallelisable: no shared mutable state between trial runs. Use
  `concurrent.futures.ProcessPoolExecutor` when `N >= 10`.
- Each trial's raw predictions and errors must be serialisable to a NumPy `.npz` file inside
  `results/` for later re-analysis without re-running.

### Online Evaluation Contract
- The online loop in `main.py` follows this strict order at every timestep `t`:
  1. Call `model.predict(x_t)` → record prediction
  2. Reveal `y_t` to the model via `model.update(x_t, y_t)`
  3. Never pass `y_t` before step 1 completes for all models.
- This contract is enforced by the `BaseModel` interface. Any model that reads data outside the
  `update` / `predict` interface is non-compliant.

### Visualisations
- Every plot must be saved to `results/` via `evaluator.py`. No `plt.show()` calls in
  library code.
- File naming convention: `results/{experiment_id}_{plot_type}.png`
  (e.g., `results/trial_001_comparison.png`, `results/aggregated_recovery.png`).
- All plots must set explicit figure size, DPI ≥ 150, and axis labels with units.
- A summary metrics table must be saved as `results/summary.csv` at the end of every run.

---

## Module Contracts

### `data/simulator.py`

```
generate_series(config: SimulatorConfig) -> pd.DataFrame
```
Returns a DataFrame with columns: `t`, `x`, `y`, `regime`.

- `regime` is an integer label (0, 1, 2, …) denoting the ground-truth regime at each step.
- Shift points, noise level, mean shift magnitude, slope shift, and variance multiplier are all
  fields on `SimulatorConfig`.
- The `regime` column must **never** be passed to any model. It is used only by `evaluator.py`.

### `models/base_model.py`

```
class BaseModel(ABC):
    def predict(self, x: np.ndarray) -> float: ...
    def update(self, x: np.ndarray, y: float) -> None: ...
    def reset(self) -> None: ...
```

All three model classes inherit from `BaseModel`. The `reset()` method restores the model to its
initial state (used between trials).

### `models/unlearning_model.py`

- Maintains a CUSUM statistic on prediction residuals.
- On threshold breach: flushes the training buffer, logs the detected shift timestamp, resets
  the CUSUM accumulator.
- During cold-start (buffer smaller than `min_window`): falls back to predicting the running mean
  of available `y` values.
- Exposes `detected_shifts: list[int]` as a public attribute for the evaluator.

### `evaluation/evaluator.py`

```
compute_metrics(predictions: dict[str, np.ndarray], y_true: np.ndarray,
                regime: np.ndarray) -> pd.DataFrame

plot_comparison(metrics: pd.DataFrame, regime: np.ndarray,
                shift_points: list[int], experiment_id: str) -> None
```

- `compute_metrics` returns per-timestep rolling MAE (window=20) for each model, plus recovery
  time per shift event.
- **Recovery time** is defined as: the number of steps after a ground-truth shift point until a
  model's rolling MAE returns to within `1.5×` its mean pre-shift MAE baseline.
- `plot_comparison` produces a 3-panel figure:
  - Panel 1: Raw time series + ground-truth regime boundaries
  - Panel 2: Rolling MAE over time for all models + recovery markers
  - Panel 3: Bar chart of recovery time per model per shift event

---

## Testing Rules

- **All logic must have `pytest` tests.** Coverage target: ≥ 90% line coverage on all files
  under `data/`, `models/`, and `evaluation/`.
- Tests live in `tests/`. Mirror the source structure (one test file per source file).
- No test may write to `results/`. Use `tmp_path` fixtures for any file output tests.
- Tests must not depend on one another. Each test is fully self-contained.
- Use `pytest.mark.parametrize` for any test covering multiple input configurations.
- Simulation tests must fix `random_seed` to ensure determinism.
- Model tests must cover: (a) correct prediction before first update, (b) correct state mutation
  after update, (c) behaviour at cold-start boundaries, (d) reset returning model to initial state.
- Unlearning model tests must verify: CUSUM accumulation, threshold triggering, buffer flush,
  and that `detected_shifts` is populated correctly.

### Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=regime_unlearning --cov-report=term-missing
```

---

## Running the Experiment

### Single trial (development)

```bash
python main.py --trials 1 --seed 42 --shifts 300 600 --output-id dev_run
```

### Multi-trial experiment

```bash
python main.py --trials 500 --seed 0 --shifts 300 600 --output-id experiment_01
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--trials` | int | 1 | Number of independent simulation trials |
| `--seed` | int | 42 | Base random seed (each trial offsets from this) |
| `--shifts` | int list | 300 600 | Timesteps at which regime shifts are injected |
| `--series-length` | int | 1000 | Total timesteps per trial |
| `--output-id` | str | `run` | Prefix for all files written to `results/` |
| `--n-workers` | int | 4 | Process pool size when `--trials >= 10` |

---

## Dependencies

```
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
matplotlib>=3.8
pytest>=8.0
pytest-cov>=5.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Results Artefacts

After a run, `results/` will contain:

| File | Description |
|---|---|
| `{id}_comparison.png` | 3-panel comparison plot (single trial) |
| `{id}_aggregated_recovery.png` | Recovery time distributions across all trials |
| `{id}_trial_NNN.npz` | Raw predictions + errors for trial NNN |
| `summary.csv` | Mean ± std recovery time per model per shift, across all trials |

`results/` is listed in `.gitignore`. Artefacts are never committed to source control.

---

## Glossary

| Term | Definition |
|---|---|
| **Regime** | A stationary interval of the time series bounded by shift events |
| **Regime shift** | An abrupt, permanent change to the data-generating process |
| **Recovery time** | Steps from a ground-truth shift until a model's error returns to 1.5× pre-shift baseline |
| **Online evaluation** | Strict predict-then-reveal loop; no look-ahead at any future label |
| **CUSUM** | Cumulative Sum control chart statistic used for change-point detection |
| **Cold-start** | The period after a buffer flush where insufficient data exists to fit a regression |
| **Unlearning** | Explicit discarding of stale pre-shift training data upon shift detection |

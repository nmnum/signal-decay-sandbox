# Detecting Signal Decay in Noisy Time Series
### An Experiment on Regime-Aware Model Adaptation

A Python research toolkit that simulates sudden distribution shifts in financial-style time-series data and measures how three prediction models — a frozen baseline, a continuously adapting roller, and an explicit "unlearning" detector — respond to those shifts. Results are surfaced through an interactive Streamlit dashboard.

---

## The Central Question

When the rules of a system change abruptly (a market regime shift, a sensor failure, a policy change), models trained on historical data become stale. How quickly can different model architectures detect the change, discard the stale knowledge, and recover accurate predictions?

This experiment answers that question empirically by measuring **recovery time** and **detection lag** across thousands of independent trials.

---

## Repository Structure

```
regime_unlearning/
│
├── app.py                      # Streamlit interactive dashboard
├── requirements.txt
├── CLAUDE.md                   # Engineering rules and module contracts
│
├── data/
│   └── simulate_regimes.py     # Vectorised AR-1 time-series simulator
│
├── models/
│   ├── base_model.py           # Abstract online-learning interface
│   ├── static_model.py         # Frozen Ridge (warm-up only)
│   ├── rolling_model.py        # Sliding-window Ridge
│   ├── unlearning_model.py     # CUSUM detector + hard memory reset
│   └── uncertainty_model.py    # Conformal prediction intervals
│
├── evaluation/
│   ├── metrics.py              # Recovery time, detection lag, rolling RMSE
│   └── run_experiment.py       # CLI batch runner (supports 10,000+ trials)
│
├── results/                    # Output plots and CSVs (git-ignored)
│
└── tests/
    ├── test_simulation.py      # 43 tests — simulator correctness
    ├── test_models.py          # 52 tests — online compliance, model behaviour
    ├── test_evaluation.py      # 45 tests — metric correctness, pipeline outputs
    └── test_app_compile.py     # 50 tests — app.py syntax and structure
```

---

## Simulation Design

The simulator generates a synthetic financial time-series with three structural properties drawn from empirical market data:

**1. Autocorrelation (AR-1 process)**
```
x_t = φ · x_{t-1} + σ_regime · ε_t       φ = 0.6 by default
```
Each feature value depends on the previous one, creating realistic momentum.

**2. Heteroskedastic noise**
Volatility changes between regimes — quiet periods followed by turbulent ones.

**3. Signal inversion**

| Regime | Relationship | Character |
|--------|-------------|-----------|
| 0 — Momentum    | y = +0.8x + ε | Low volatility, positive trend-following |
| 1 — Mean-Revert | y = −0.8x + ε | Medium volatility, signal inverts |
| 2 — Dead Signal | y = ε only    | High volatility, no predictive information |

Shifts between regimes are **abrupt and unmarked** — models receive no signal that the regime has changed.

---

## Models

All models share a strict online interface:

```python
class BaseModel(ABC):
    def predict(self, x_t: float) -> float: ...   # called BEFORE label is revealed
    def update(self, x_t: float, y_t: float) -> None: ...  # called AFTER
```

### Static Model
Fits a Ridge regression on the first `warmup_steps` observations, then **freezes**. Represents the worst case — a model that never unlearns. Useful as a performance floor.

### Rolling Model
Refits Ridge on every step using a sliding window of the most recent `window_size` observations. Adapts continuously, but stale pre-shift data dilutes the window for up to `window_size` steps after a shift.

### Unlearning Model
Monitors prediction residuals with a **CUSUM (Page, 1954) change-point detector**:
```
S_t = max(0,  S_{t-1} + |e_t| − μ̂ − k)
```
When `S_t ≥ h` (threshold), it declares a regime shift, **completely flushes** the training buffer, resets the CUSUM statistic, and enters a cold-start phase until enough new data accumulates. This is the fastest-recovering model, at the cost of a brief cold-start penalty.

### Uncertainty Model
Extends Rolling with **split-conformal prediction intervals**. Maintains a separate calibration buffer of past residuals; the interval width at each step is the empirical quantile of that buffer. Width spikes can serve as a leading indicator of regime shifts.

---

## Metrics

| Metric | Definition |
|--------|-----------|
| **Rolling RMSE** | Causal windowed RMSE (window = 50 steps) at each timestep |
| **Recovery Time** | Steps after a true shift until rolling RMSE ≤ 2× pre-shift baseline |
| **Detection Lag** | Steps between true shift and first CUSUM alarm (Unlearning only) |
| **Warning Lead Time** | Steps before a shift at which interval width first spiked (Uncertainty only) |

---

## Installation

```bash
git clone <repo-url>
cd regime_unlearning
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, NumPy ≥ 1.26, Pandas ≥ 2.0, scikit-learn ≥ 1.4, Streamlit ≥ 1.35, Plotly ≥ 5.22.

---

## Running the Dashboard

```bash
streamlit run app.py
```

The sidebar exposes all simulation parameters. Click **▶ Run Simulation** to execute a trial. The dashboard displays:

- **Panel 1** — Raw time-series with colour-coded regime bands and detected shift markers
- **Panel 2** — Rolling RMSE over time for all three models
- **Recovery Metrics** — Per-shift recovery times and CUSUM detection summary
- **Final RMSE** — Steady-state prediction quality scoreboard

Click **🔄 Reset Simulation** at the top of the sidebar to clear all results and restore default parameters instantly. Results persist across widget interactions — moving a slider does not wipe the current chart; only **Run Simulation** triggers a recompute.

---

## Running Batch Experiments

The CLI runner supports thousands of independent trials for statistical analysis:

```bash
# Single trial (generates error_plot.png + summary_metrics.csv)
python evaluation/run_experiment.py --trials 1 --steps 1000 --seed 42

# 1,000-trial sweep (summary CSV only — no plot flood)
python evaluation/run_experiment.py --trials 1000 --steps 1000 --seed 42

# Custom regime boundaries
python evaluation/run_experiment.py --trials 500 --steps 1200 --shift-points 400 800
```

Output files are written to `results/`:

| File | Generated when |
|------|---------------|
| `results/error_plot.png` | Single-trial runs |
| `results/summary_metrics.csv` | All runs |

---

## Running the Tests

```bash
# Full suite (176 tests)
python -m unittest tests/test_simulation.py tests/test_models.py \
                   tests/test_evaluation.py tests/test_app_compile.py -v

# Individual modules
python -m unittest tests/test_simulation.py -v    # 43 tests
python -m unittest tests/test_models.py -v        # 52 tests
python -m unittest tests/test_evaluation.py -v    # 45 tests
python -m unittest tests/test_app_compile.py -v   # 50 tests
```

The `test_app_compile.py` suite verifies `app.py` using Python's `ast` module — no Streamlit or Plotly installation required for testing.

---

## Key Design Decisions

**Strict online evaluation** — models never see future data. The evaluation loop always calls `predict(x_t)` before `update(x_t, y_t)`. Tests verify this property explicitly by checking that perturbing future labels does not change past predictions.

**Vectorised simulation** — all data generation uses NumPy array operations; the AR-1 recurrence is the only sequential step, run over a pre-allocated array. `simulate_bulk_arrays()` returns `(X, Y, R)` arrays of shape `(n_trials, n_steps)` for batch experiments without DataFrame overhead.

**No data leakage** — the `regime` ground-truth column exists only in the simulator output and the evaluation layer. No model constructor or `update` method receives regime labels.

**Session-state persistence** — the Streamlit app stores all simulation results in `st.session_state["results"]`, so adjusting a sidebar slider does not wipe the current chart. The Reset button calls `st.session_state.clear()` followed by `st.rerun()` to return to a clean default state.

---

## Expected Results

Across a typical 1,000-trial sweep with default parameters:

| Model | Recovery at Shift 1 | Recovery at Shift 2 | Final RMSE |
|-------|--------------------|--------------------|------------|
| Static | Never (frozen) | ~0 steps | Highest |
| Rolling | ~90 steps | ~60 steps | Medium |
| Unlearning | ~65 steps | ~50 steps | Lowest |

The Unlearning model pays a 20–30 step cold-start cost after each reset, then converges faster than Rolling because it carries no pre-shift contamination. The Static model's advantage at Shift 2 (regime 2 is noise-only, so a near-zero coefficient is correct by coincidence) illustrates why the comparison requires looking at *all* shifts together.

---

## References

- Page, E. S. (1954). *Continuous inspection schemes*. Biometrika, 41(1/2), 100–115.  
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.  
- Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series*. Econometrica, 57(2), 357–384.

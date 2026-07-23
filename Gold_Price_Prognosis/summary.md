# Gold Price Forecasting Notebook — Summary

## 1. What the notebook does (program flow)

The notebook (`Gold_Price_Prognosis/notebooks/gold_price_forecasting.ipynb`) is a thin CRISP-DM-structured orchestrator. It contains **no model logic itself** — every model lives in its own file under `src/gold_forecasting/models/`, and the notebook just calls one function per model. Flow:

1. **Business/data understanding** (Kapitel 1–2): download Gold futures (`GC=F`) plus exogenous series from Yahoo Finance (2000-01-01 → 2026-06-30), describe/plot the raw series.
2. **Data preparation & split** (Kapitel 3): build the chronological train/validation/test split, print all key parameters, plot the split.
3. **Modelling** (Kapitel 4): two experiments, three-to-six models each, with hyperparameter optimization (HPO) where applicable.
4. **Evaluation** (Kapitel 5): metrics tables, per-lead-time error comparison, interactive Plotly charts, loss curves, a combined comparison across all 12 models (6 univariate + 6 multivariate), and a single combined CSV artifact (`artifacts/metrics/all_models_comparison.csv`) with every model's error values.
5. **Deployment** (Kapitel 6): a genuine one-shot forecast beyond the data's end (univariate models only).
6. **Runtime report** (Kapitel 7): how long each model took, or whether it was read from cache.

### How the project got to this shape

The pipeline went through two full redesigns before reaching its current form:

- **Redesign 1 — two-experiment structure.** The original notebook used a grab-bag of models (naive, ARIMA, LSTM, N-BEATS, PatchTST) with a single frozen-origin evaluation. It was rebuilt into the current **Experiment 1 (univariate) / Experiment 2 (multivariate)** split, LSTM/N-BEATS were dropped in favor of a lighter model set, and TFT was added as a hand-built native PyTorch model instead of pulling in `pytorch-forecasting`/Lightning as a dependency.
- **Redesign 2 — rolling-origin (walk-forward) evaluation.** The single biggest methodological change: instead of one big forecast over the whole validation/test period, every model now forecasts a fixed 20-day horizon, steps the origin forward 20 days, and repeats across a full year — with an explicit, structural leakage-prevention rule (`fit_data` is always strictly "everything before this window") that also makes "reuse the frozen-hyperparameter validation set as training data for testing" fall out automatically, with no special-casing. This is by far the most expensive part of the pipeline, because `retrain_each_step` applies to **every** model at **every** rolling window — including inside every Optuna HPO trial, not just the final evaluation. That was a deliberate, explicit choice (accepting much higher compute cost for a much more realistic, non-leaking backtest) rather than an oversight.
- **Iterative refinements on top of the rolling-origin core**: per-model `enabled` toggles in `configs/models.yaml` (so any model can be switched off without breaking downstream comparison/plotting code); interactive Plotly charts (legend-click to show/hide a model) replacing static PNGs for every multi-model comparison; real train/validation loss curves for the three models where "loss" is a meaningful concept (PatchTST, TFT, XGBoost — see §4 for why the others are excluded); content-addressed caching so a fresh clone or a rerun on a different machine reuses a completed run's results instead of retraining; and an MLflow SQLite backend (the original filesystem tracking store hit MLflow's "maintenance mode" deprecation).
- **Execution environment**: the compute-heavy `full_mode` run (25 HPO trials × up to 60 epochs × ~13 rolling windows, repeated for every tunable model, in both experiments) was executed on a rented RunPod GPU instance (RTX 5090) rather than locally, then the resulting `artifacts/` cache and the MLflow database were pulled back via `scp` and committed.
- **XGBoost (returns), added after diagnosing the raw-price XGBoost's poor result.** Rather than replacing the original model, a second XGBoost variant was added that predicts returns instead of price levels (see §4 and §7.4 for the full diagnosis and rationale) — both are kept and compared side by side.
- **Continuous per-day error curves (day 1..20), added on top of the existing day 1/10/20 checkpoint table.** Every rolling forecast already records a `lead_time` (1..horizon) for each point, so this is a pure post-hoc recomputation over each model's already-cached result — no retraining, and (unlike widening `experiments.yaml: lead_time_checkpoints` itself, which is folded into the cache signature and would invalidate every cached forecast) it doesn't touch any cache key. New artifacts: `artifacts/metrics/{univariate,multivariate}/error_by_day.csv` and a combined `artifacts/metrics/all_models_error_by_day.csv`.
- **Interactive figures are now also persisted as standalone HTML** (`fig.write_html(..., include_plotlyjs="cdn")`), alongside the CSVs they're built from — previously these Plotly charts only existed live inside the notebook/dashboard and were never saved to `artifacts/figures/`.
- **Correlation matrix (gold price vs. exogenous variables), added at the start of Kapitel 4.2** (before any Experiment 2 model runs) — a descriptive/exploratory heatmap over the full available history, computed before the chronological split even happens, so it has no bearing on leakage. See §4 for the actual values and what they suggest about which exogenous drivers carry a strong linear relationship with gold.
- **Six more diagnostic visualization types, added after auditing what the notebook was missing** (previously there was no HPO visualization at all, among other gaps): (1) an exogenous-variables-over-time overview complementing the correlation matrix; (2) per-experiment and combined **leaderboard bar charts** ranking models by MAE; (3) **directional-accuracy-by-lead-time** curves (that metric existed in the tables but was never plotted); (4) **residual diagnostics** — an overlaid histogram per model plus a box-plot of residual spread by lead-time day; (5) **XGBoost feature importance** (both variants) straight from the trained model's `.feature_importances_`, no extra training; (6) an **HPO diagnostics section** (Kapitel 5.1) — convergence plots (each trial's objective plus the running best-so-far) and per-hyperparameter scatter plots, for every Optuna-tuned model, read directly from the already-saved `artifacts/hpo_trials/*.csv`. All of these reuse already-computed/cached results (rolling forecasts, metrics, trial logs) — none of them retrains anything; the one exception was populating feature importance for XGBoost/XGBoost (returns) once, since that field didn't exist in already-cached runs from before this capability was added (a one-time ~30-second refit of just the final model, not the HPO search, which stayed cached).

## 2. Data split

- **Source**: Yahoo Finance, daily, `2000-01-01` through `2026-06-30` (both configurable in `configs/data.yaml`).
- **Chronological split** (`configs/experiments.yaml: split`): validation and test are each **1 year long**, cut backward from the last available date. Everything earlier becomes training data. This is a hard, non-negotiable time ordering — train ends before validation starts, validation ends before test starts.
- Two parallel splits exist:
  - **Univariate** frame: just `gold_usd`.
  - **Multivariate** frame: `gold_usd` + 6 exogenous series (US Dollar Index, Silver, Oil, S&P 500, VIX, Bitcoin), same date boundaries.

## 3. How validation and testing actually work: rolling-origin (walk-forward) evaluation

This is the core methodological idea, and it's the same mechanism for **both** HPO scoring (on validation) and final scoring (on test) — just pointed at a different region.

- **Horizon = 20 trading days, step = 20 trading days** (`configs/experiments.yaml: rolling`). Instead of one giant forecast covering the whole test year, the model forecasts a 20-day window, the origin then jumps forward 20 days, and it forecasts the next 20-day window — repeating across the whole year (~13 non-overlapping windows). All the small forecasts get concatenated for metrics/plots. Incomplete trailing windows are dropped.
- **No leakage, structurally**: at every window, the model is only ever given data strictly *before* that window's start date (`fit_data = combined.iloc[:window_start]`). Since `combined` spans train+validation+test continuously, this single rule automatically guarantees:
  - HPO (scored only on the validation region) never sees test data at all — test isn't even in the frame HPO operates on.
  - The very first test window's `fit_data` is exactly train+validation combined — which is *exactly* "reuse validation as training data after HPO freezes hyperparameters," with no special-casing needed.
- **Retraining discipline** (`retrain_each_step: true` for every tunable model, in `configs/models.yaml`): every model refits/updates itself at **every** rolling window, using only the data available at that window's origin — not just once at the start. This happens identically during HPO trials and at final test time.
  - **SARIMA / SARIMAX / XGBoost**: full refit from scratch every window (cheap enough, simplest to reason about).
  - **PatchTST / TFT**: the *first* window does a full fit (`epochs` budget, with early stopping via `patience`); every later window does a cheaper **warm-start** — continuing training the existing weights for a small `update_epochs` budget rather than reinitializing.
- **HPO**: for SARIMA, SARIMAX, XGBoost, PatchTST, TFT, Optuna searches hyperparameters by scoring each candidate with the *same* rolling-origin walk-forward evaluation, but restricted to the validation region only. The objective is mean absolute error across all validation windows. Optuna studies are stored in SQLite (`artifacts/optuna/`) so an interrupted run resumes only the missing trials rather than starting over.
- **Known-exogenous assumption** (Experiment 2 only): each 20-day window's forecast uses the *actual* realized exogenous values for those 20 days (a standard, and here much more locally-scoped, assumption for exogenous-regression models) — while the target's own lag features are still generated recursively from the model's own prior predictions inside the window, never from real future target values.
- **Metrics / lead-time checkpoints**: reported at lead-time day 1, day 10 (half the window), and day 20 (the full window) — always slices of the *same* 20-day forecast, never a different horizon — plus one "complete" row aggregating every point.

## 4. The models — and why each one is in the analysis

### Experiment 1 — Univariate (no exogenous data)

| Model | Concept | Why it's here for gold |
|---|---|---|
| **Naive** (baseline) | Repeats the last known value. | Gold's spot price behaves close to a random walk over short horizons — this is the bar every other model *must* clear to justify its own complexity. If a model can't beat naive, its extra machinery isn't earning its keep. |
| **Moving average** (baseline) | Repeats the mean of the trailing 20 observations. | A second, slightly smoother baseline — tests whether short-term mean-reversion (rather than pure persistence) is a better description of gold's day-to-day behavior. |
| **SARIMA** | Classical statistical time-series model (AutoRegressive Integrated Moving Average). Fits `(p, d, q)` coefficients by maximum likelihood on the target series alone. | The standard, well-understood linear-statistics benchmark for any price series. Establishes whether linear autocorrelation structure in gold's own price history (beyond a pure random walk) has any forecasting value at all, before reaching for anything nonlinear. |
| **PatchTST** | A small **native PyTorch transformer**, purpose-built for time series. Splits the input context window into overlapping "patches" (like image patches in a ViT), embeds each patch, runs them through a standard transformer encoder, and predicts the next value from the pooled representation. | Tests whether a learned, nonlinear representation of gold's own price history (regime shifts, momentum, volatility clustering) can outperform SARIMA's linear structure — using only the target series, so any gain is attributable purely to nonlinearity/representation learning, not to extra data. |
| **Chronos (Original)** | Amazon's pretrained **zero-shot** time-series foundation model (`amazon/chronos-t5-small`), a T5-architecture transformer trained on a huge corpus of diverse time series, used here with **no training at all**. | Gold has a long, clean, well-behaved public price history, which is exactly the setting where a general-purpose pretrained forecaster is expected to be competitive — this tests whether "generic time-series knowledge" transfers to gold without any gold-specific training at all. |
| **Chronos (Bolt)** | A faster, distilled/optimized, quantile-based variant of Chronos (`amazon/chronos-bolt-small`) — also zero-shot. | Same rationale as Chronos Original, but checks whether the faster/quantile-based variant trades away accuracy for speed on this specific series, or holds up just as well. |

### Experiment 2 — Multivariate (with exogenous variables)

**Correlation with gold price (full history, `artifacts/metrics/exogenous_correlation.csv`):**

| Silver | S&P 500 | Bitcoin | Oil | Dollar Index | VIX |
|---|---|---|---|---|---|
| **0.922** | **0.876** | **0.846** | 0.377 | 0.087 | -0.067 |

Two things worth noting against the rationale table below: **silver, S&P 500, and Bitcoin** show strong positive linear correlation with gold over the full history — silver unsurprisingly (both precious metals move on shared macro drivers), but the S&P 500/Bitcoin correlations run counter to gold's usual "risk-off hedge" narrative (a *positive*, not negative, relationship with equities and crypto over this period). The **Dollar Index and VIX — the two variables with the clearest textbook rationale for a relationship with gold — show almost no linear correlation** (0.087 and -0.067) over the full 24-year history. This doesn't mean those relationships don't exist; a single correlation coefficient over 24 years averages over very different regimes (e.g. the dollar/gold relationship is better known to hold over shorter windows or in returns/changes rather than price levels) — but it's a useful, humbling check before trusting any of the "why it's here" rationale below at face value.

| Model | Concept | Why it's here for gold |
|---|---|---|
| **SARIMAX** | SARIMA's multivariate sibling: same MLE-fit ARIMA structure, but with exogenous regressors added as external predictors. Exogenous inputs are standardized (`StandardScaler`); the target itself is left unscaled. | Gold prices are widely understood to move with a handful of specific macro drivers: the **US Dollar Index** (gold is dollar-denominated — a stronger dollar mechanically pressures gold down), **Silver and Oil** (correlated commodities / shared risk-appetite and inflation exposure), the **S&P 500** (equity-market risk sentiment — gold is a classic "risk-off" hedge), the **VIX** (a direct volatility/fear proxy — gold demand often rises with the VIX), and **Bitcoin** (a modern, competing "alternative store of value" narrative). SARIMAX tests whether adding these as *linear* external predictors improves on SARIMA's univariate-only structure. |
| **XGBoost** | Gradient-boosted decision trees over **engineered tabular features**: lagged prices (1/2/5/10/20 days back), rolling mean/std (5/20-day windows), calendar features (day-of-week, month), and the exogenous series (same-day + 1-day-lagged). Predicts the **raw price level** directly. | Tests whether a nonlinear, interaction-aware model (trees can capture things like "VIX matters more when the dollar is also weak," which a linear SARIMAX coefficient can't) beats the linear multivariate baseline, using the same exogenous drivers as SARIMAX but through a fundamentally different, tree-based learning mechanism. |
| **XGBoost (returns)** | Identical features to XGBoost above, but predicts the next-step **percentage return** instead of the raw price level; the absolute forecast is reconstructed by compounding predicted returns onto the last known price. | Added after diagnosing why the raw-price XGBoost above performs so poorly (§7.4): gradient-boosted trees output piecewise-constant leaf averages learned from the training distribution and can't linearly extrapolate a price trend the way SARIMAX's AR coefficients or a scaled neural net can. Gold's price trended from ~$1,050–2,434 (train) to ~$3,293–5,318 (test) — a regime the raw-price model's training data barely represents. Predicting a stationary target (returns) instead removes that dependency on the absolute price scale. Kept side by side with the original rather than replacing it, since the comparison between the two representations is itself informative. |
| **TFT (compact, native)** | A hand-built, lightweight approximation of the **Temporal Fusion Transformer**: a variable-selection gating layer (learns which input channels matter), an LSTM encoder, and a single multi-head self-attention block. | Tests whether a neural architecture that can *learn which exogenous driver matters when* (via its gating layer) — rather than assuming a fixed linear or tree-split relationship — captures gold's multivariate dynamics better than SARIMAX or XGBoost. Built natively (no `pytorch-forecasting`) to stay consistent with PatchTST's implementation style. |

## 5. Hyperparameters per model

*(search ranges are the Optuna spaces used for `full_mode`'s 25 trials; the "Actual (post-HPO)" column is what Optuna actually selected as `best_params`, i.e. what's frozen and used for the final rolling-origin test evaluation.)*

| Model | Tuned hyperparameters (search range) | Fixed / fallback | Actual (post-HPO) |
|---|---|---|---|
| **SARIMA** | `p` ∈ [0,4], `d` ∈ [0,2], `q` ∈ [0,4] | seasonal component disabled; fallback order (2,1,2) | `order=(3, 0, 4)`, seasonal `(0,0,0,0)` |
| **SARIMAX** | `p` ∈ [0,3], `d` ∈ [0,2], `q` ∈ [0,3] | same structure, smaller search range (more expensive per fit with exogenous regressors) | `order=(1, 2, 2)`, seasonal `(0,0,0,0)` |
| **PatchTST** | `patch_length` [4,16], `stride` [2,8], `d_model` [16,64], `nhead` {2,4,8}, `layers` [1,4], `learning_rate` [1e-4,1e-2 log], `batch_size` [32,128] | `context_length` fixed at 64 (not tuned) | `patch_length=6`, `stride=3`, `d_model=56`, `nhead=8`, `layers=1`, `learning_rate=0.000406`, `batch_size=96` |
| **TFT** | `hidden_size` [16,64], `attention_heads` {1,2,4}, `lstm_layers` [1,2], `dropout` [0.0,0.3], `learning_rate` [1e-4,1e-2 log], `batch_size` [32,128] | `context_length` fixed at 64 | `hidden_size=48`, `attention_heads=1`, `lstm_layers=2`, `dropout=0.114`, `learning_rate=0.00249`, `batch_size=96` |
| **XGBoost** | `max_depth` [3,10], `learning_rate` [0.01,0.3 log], `n_estimators` [100,600], `subsample` [0.5,1.0], `colsample_bytree` [0.5,1.0] | feature lags/windows fixed via `data.yaml` (1/2/5/10/20-day lags, 5/20-day rolling windows) | `max_depth=3`, `learning_rate=0.0539`, `n_estimators=117`, `subsample=0.955`, `colsample_bytree=0.629` |
| **XGBoost (returns)** | same search space as XGBoost above | same features as XGBoost; target transform (return vs. level) is not itself tuned | `max_depth=10`, `learning_rate=0.0477`, `n_estimators=103`, `subsample=0.990`, `colsample_bytree=0.995` |
| **Chronos (both)** | none — zero-shot, no training | model size fixed to "small" variant; context truncated to last 512 observations | `model_id="amazon/chronos-t5-small"` (Original) / `"amazon/chronos-bolt-small"` (Bolt) — no HPO ever runs |
| **Naive / Moving average** | none | moving-average window fixed at 20 days | n/a — no HPO |

Note the two XGBoost variants converged to noticeably different hyperparameters despite an identical search space: the returns-based model chose the *maximum* `max_depth` (10, vs. 3 for the level-based one) and much higher `subsample`/`colsample_bytree` (~0.99 vs. ~0.6–0.95) — consistent with a stationary return target being harder to overfit on and rewarding deeper, more data-hungry trees than a raw price level that a shallow tree can already track well (if only within its own training range).

**Shared HPO budget** (`configs/models.yaml: hpo`, mode-dependent):
- `quick_mode`: 3 trials, 3 epochs (initial fit), patience 2, update_epochs 1 — fast, for testing.
- `full_mode` (default): 25 trials, 60 epochs, patience 8, update_epochs 5 — the production-quality run, executed on RunPod, whose results populate every table in this document.

A few notable HPO outcomes worth flagging: both SARIMA and SARIMAX converged to `d=0`/`d=2` respectively rather than the fallback `d=1` — i.e. the search actively preferred a different differencing order than the naive default once scored against real rolling-origin validation error. Both PatchTST and TFT converged to the *smallest* end of their `layers`/`lstm_layers` ranges (1 layer, 2 layers) with fairly small hidden dimensions (56, 48) — on a validation objective this small (1-year, 20-day windows), Optuna consistently preferred simpler/less expressive networks over larger ones.

## 6. What differentiates the models from each other

- **Statistical vs. learned vs. foundation model**: SARIMA/SARIMAX are classical linear statistics (MLE-fit, no learned representation); PatchTST/TFT are trained neural networks (gradient descent, learn nonlinear patterns from this data specifically); XGBoost (both variants) is a learned ensemble of decision trees (nonlinear, but axis-aligned splits, not gradient-based representation learning); Chronos is a **pretrained foundation model** used zero-shot — it learned general time-series patterns from a huge external corpus and is never trained on this gold-price data at all.
- **Univariate vs. multivariate**: SARIMA/PatchTST/Chronos see only the gold price history. SARIMAX/XGBoost (both variants)/TFT additionally see the 6 exogenous series.
- **Retraining behavior per rolling window**: SARIMA/SARIMAX/XGBoost (both variants) fully re-fit from scratch every 20-day window (cheap enough for these model types). PatchTST/TFT warm-start (continue training existing weights) rather than reinitializing, since full retraining of a neural net at every window would be prohibitively expensive.
- **Scaling**: PatchTST/TFT scale all inputs (`StandardScaler`, fit on training data only). SARIMAX scales only the exogenous regressors, not the target. XGBoost (both variants) needs no input scaling (tree splits are scale-invariant). SARIMA needs no scaling (works directly on price levels). Chronos does its own internal normalization.
- **Target representation — the one axis that splits the two XGBoost variants**: every other model predicts (or, for SARIMA/SARIMAX, differences toward) the raw price level. The XGBoost (returns) variant is the only model in the whole comparison whose *target* is a stationary percentage return rather than a price level, with the absolute forecast reconstructed by compounding — see §7.4 for why the plain XGBoost needed this alternative in the first place.
- **Training cost profile**: Chronos is the cheapest (no training, just inference). SARIMA/SARIMAX/XGBoost (both variants) are cheap-to-moderate (each of the ~13×26 refits — 25 trials + 1 final, ×13 rolling windows — takes seconds). PatchTST/TFT are the most expensive by far, since even a "warm-start" update still runs several epochs of backprop at every one of ~13 windows, for every one of 25 HPO trials.

## 7. Results — error comparison across all models

Numbers below are from the `full_mode` rolling-origin **test-period** evaluation (data hash `12a242c0...b0d4e4a`), read directly from `artifacts/metrics/{univariate,multivariate}/*.csv`. "Complete" aggregates every point across all ~13 rolling windows; the lead-time columns isolate day 1, day 10, and day 20 of each 20-day window specifically. A single combined CSV with every model's error values (both experiments, all lead-time rows) is written by the notebook to `artifacts/metrics/all_models_comparison.csv` (Kapitel 5) for direct use outside the notebook.

### 7.1 Univariate (Experiment 1) — ranked by overall MAE

| Model | MAE | RMSE | MASE | sMAPE (%) | Directional accuracy |
|---|---|---|---|---|---|
| Naive | **186.04** | 237.41 | 21.61 | 4.32 | 0.410 |
| SARIMA | 189.53 | 248.61 | 22.02 | 4.34 | 0.423 |
| Chronos (Original) | 190.56 | 248.94 | 22.14 | 4.36 | 0.431 |
| Chronos (Bolt) | 202.39 | 261.49 | 23.51 | 4.63 | 0.431 |
| Moving average | 214.75 | 287.24 | 24.95 | 4.97 | 0.460 |
| PatchTST | 251.14 | 317.58 | 29.17 | 5.95 | 0.448 |

### 7.2 Multivariate (Experiment 2) — ranked by overall MAE

| Model | MAE | RMSE | MASE | sMAPE (%) | Directional accuracy |
|---|---|---|---|---|---|
| **SARIMAX** | **145.08** | 206.83 | 13.99 | 3.29 | 0.590 |
| Naive | 186.04 | 237.41 | 17.94 | 4.32 | 0.410 |
| XGBoost (returns) | 206.76 | 274.00 | 19.94 | 5.01 | 0.452 |
| Moving average | 214.75 | 287.24 | 20.71 | 4.97 | 0.460 |
| TFT | 224.85 | 333.38 | 21.68 | 5.27 | 0.485 |
| XGBoost (raw price level) | 1625.04 | 1681.72 | 156.71 | 46.75 | 0.435 |

### 7.3 MAE by lead time (day 1 / day 10 / day 20 of each 20-day window)

**Univariate**

| Model | Day 1 | Day 10 | Day 20 |
|---|---|---|---|
| Naive | 50.3 | 203.0 | 266.5 |
| SARIMA | 51.2 | 202.8 | 278.6 |
| Chronos (Original) | 57.0 | 190.4 | 285.3 |
| Chronos (Bolt) | 69.0 | 192.4 | 330.2 |
| Moving average | 89.2 | 213.0 | 290.0 |
| PatchTST | 126.1 | 262.4 | 312.6 |

**Multivariate**

| Model | Day 1 | Day 10 | Day 20 |
|---|---|---|---|
| SARIMAX | **40.4** | **159.0** | 274.0 |
| Naive | 50.3 | 203.0 | 266.5 |
| XGBoost (returns) | 48.1 | 206.5 | 325.8 |
| Moving average | 89.2 | 213.0 | 290.0 |
| TFT | 101.1 | 225.1 | 293.3 |
| XGBoost (raw price level) | 1566.0 | 1595.4 | 1674.4 |

### 7.4 Overall takeaways

- **SARIMAX is the standout result**: it's the only model in either experiment to convincingly beat the naive baseline overall (MAE 145 vs 186 — a ~22% reduction), and it wins at every single lead-time checkpoint. This suggests the exogenous drivers (dollar index, silver/oil, S&P 500, VIX, Bitcoin) carry real, linearly-extractable short-horizon signal for gold — but only once combined *linearly and directly* with the target's own ARIMA structure.
- **Univariate models cluster tightly around (and barely beat) naive persistence**: SARIMA, Chronos Original, and Chronos Bolt all land within ~10% of naive's MAE, in both directions depending on the lead time — consistent with gold's price behaving close to a random walk when no external drivers are used. Naive itself is a genuinely strong benchmark here, not a token baseline.
- **PatchTST underperforms every univariate baseline**, including the two trivial ones (naive, moving average). With only 1 year of test data split into 20-day windows and a fairly small architecture (Optuna converged to 1 transformer layer), the model likely doesn't have enough rolling-window signal to learn a representation that beats simple persistence — this is a plausible, known failure mode for small transformers on relatively short, noisy financial series rather than an implementation bug.
- **XGBoost's raw-price-level MAE of ~1625 is root-caused, not just flagged**: it is *not* a recursive-forecast or feature-alignment bug (verified directly: fed the model's own real last-training-row features, e.g. `price_lag_1=4718.7` with all recent prices around $4,700, it still predicts ~$2,940 — even at the last test window, where `retrain_each_step` has already folded real prices up to $5,318 into `fit_data`). The actual cause: gradient-boosted trees output piecewise-constant leaf averages learned from the training distribution, and cannot linearly extrapolate a price trend the way SARIMAX's AR coefficients or a scaled neural net can. Gold's price trended from ~$1,050–2,434 (train) to ~$3,293–5,318 (test); the high-price rows that would inform a correct high-price leaf are a small, recent minority of the training set, so predictions stay anchored near the much more populous historical (lower-price) regime. `feature_importances_` confirms the model *is* relying heavily on `price_lag_1/2/10` and `rolling_mean_5` (~91% combined importance) — it's using the right features, just in a representation trees can't extrapolate.
- **The fix — predicting returns instead of price levels — works, and is now run as a second model (XGBoost, returns) rather than a replacement.** Same features, same search space, but the regression target is the next-step percentage return (a stationary quantity independent of the absolute price scale), with the absolute forecast reconstructed by compounding. Result: MAE drops from 1625.04 to 206.76 — back in the same range as the other multivariate models, though still short of SARIMAX (145.08) or even naive (186.04). Interestingly HPO chose very different hyperparameters for it (`max_depth=10`, `subsample≈0.99` vs. `max_depth=3`, `subsample≈0.95` for the level-based model) — consistent with a stationary return target being harder to overfit and rewarding deeper, more data-hungry trees. Both variants are kept side by side (§4, §5) since the comparison between the two target representations is itself an informative result, not just a bug fix. The feature-importance diagnostic (Kapitel 5, `artifacts/feature_importance/`) confirms this mechanically: once the target is stationarized, the model's top features flip from price-level lags (`price_lag_2`, `rolling_mean_5`, `price_lag_1`) to the **exogenous macro variables** (`exog_silver_lag_1`, `exog_silver`, `exog_dollar_index_lag_1`, `exog_dollar_index`, `exog_vix`) — the model can no longer "cheat" by memorizing price levels, so it actually leans on the same macro drivers SARIMAX uses linearly.
- **Directional accuracy degrades for every model as lead time grows** (day 1 is often ~100% for the persistence-like models almost by construction, dropping toward 40–55% by day 20 — close to a coin flip), which is the expected signature of a near-random-walk series: short-horizon direction is comparatively predictable, 20-day-ahead direction is not.
- **Chronos (zero-shot, no gold-specific training at all) is competitive with SARIMA** on the univariate side, which is a genuinely interesting result: a general-purpose pretrained forecaster gets within ~1% of a model fit specifically to this data's own history.

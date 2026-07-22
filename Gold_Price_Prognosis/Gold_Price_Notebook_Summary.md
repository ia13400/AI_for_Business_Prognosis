# Gold Price Forecasting Notebook — Private Summary

*Not part of the repo — for personal reference only.*

## 1. What the notebook does (program flow)

The notebook (`Gold_Price_Prognosis/notebooks/gold_price_forecasting.ipynb`) is a thin CRISP-DM-structured orchestrator. It contains **no model logic itself** — every model lives in its own file under `src/gold_forecasting/models/`, and the notebook just calls one function per model. Flow:

1. **Business/data understanding** (Kapitel 1–2): download Gold futures (`GC=F`) plus exogenous series from Yahoo Finance (2000-01-01 → 2026-06-30), describe/plot the raw series.
2. **Data preparation & split** (Kapitel 3): build the chronological train/validation/test split, print all key parameters, plot the split.
3. **Modelling** (Kapitel 4): two experiments, three models each, with hyperparameter optimization (HPO) where applicable.
4. **Evaluation** (Kapitel 5): metrics tables + a combined comparison across all 6 models.
5. **Deployment** (Kapitel 6): a genuine one-shot forecast beyond the data's end (univariate models only).
6. **Runtime report** (Kapitel 7): how long each model took, or whether it was read from cache.

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

## 4. The models

### Experiment 1 — Univariate (no exogenous data)

| Model | Concept |
|---|---|
| **Naive** (baseline) | Repeats the last known value. The bar every model must clear to be worth its complexity. |
| **Moving average** (baseline) | Repeats the mean of the trailing 20 observations. |
| **SARIMA** | Classical statistical time-series model (AutoRegressive Integrated Moving Average). Fits `(p, d, q)` coefficients by maximum likelihood on the target series alone. No neural network, no learned representations — just linear autoregression on differenced data. |
| **PatchTST** | A small **native PyTorch transformer**, purpose-built for time series. Splits the input context window into overlapping "patches" (like image patches in a ViT), embeds each patch, runs them through a standard transformer encoder, and predicts the next value from the pooled representation. Learns nonlinear temporal patterns the linear SARIMA can't. |
| **Chronos (Original)** | Amazon's pretrained **zero-shot** time-series foundation model (`amazon/chronos-t5-small`), a T5-architecture transformer trained on a huge corpus of diverse time series and used here with **no training at all** — just fed the recent price history as context and asked to forecast. |
| **Chronos (Bolt)** | A faster, distilled/optimized variant of Chronos (`amazon/chronos-bolt-small`) — also zero-shot, predicts quantiles directly (not autoregressive sampling) so it's much quicker at inference. |

### Experiment 2 — Multivariate (with exogenous variables)

| Model | Concept |
|---|---|
| **SARIMAX** | SARIMA's multivariate sibling: same MLE-fit ARIMA structure, but with exogenous regressors (Dollar Index, Silver, Oil, S&P 500, VIX, Bitcoin) added as external predictors. Exogenous inputs are standardized (`StandardScaler`); the target itself is left unscaled. |
| **XGBoost** | Gradient-boosted decision trees over **engineered tabular features**: lagged prices (1/2/5/10/20 days back), rolling mean/std (5/20-day windows), calendar features (day-of-week, month), and the exogenous series (same-day + 1-day-lagged). Tree-based, so no scaling needed (scale-invariant splits). Multi-step forecasts are generated recursively within each 20-day window. |
| **TFT (compact, native)** | A hand-built, lightweight approximation of the **Temporal Fusion Transformer**: a variable-selection gating layer (learns which input channels matter), an LSTM encoder, and a single multi-head self-attention block. Built natively in PyTorch (no `pytorch-forecasting`/Lightning dependency) to stay consistent with PatchTST's style, deliberately simplified relative to the full TFT paper architecture. |

## 5. Hyperparameters per model

*(ranges below are the Optuna search spaces used for full_mode's 25 trials; "fallback" values are only used if HPO is skipped, e.g. 0 trials configured)*

| Model | Tuned hyperparameters (search range) | Fixed / fallback |
|---|---|---|
| **SARIMA** | `p` ∈ [0,4], `d` ∈ [0,2], `q` ∈ [0,4] | seasonal component disabled; fallback order (2,1,2) |
| **SARIMAX** | `p` ∈ [0,3], `d` ∈ [0,2], `q` ∈ [0,3] | same structure, smaller search range (more expensive per fit with exogenous regressors) |
| **PatchTST** | `patch_length` [4,16], `stride` [2,8], `d_model` [16,64], `nhead` {2,4,8}, `layers` [1,4], `learning_rate` [1e-4,1e-2 log], `batch_size` [32,128] | `context_length` fixed at 64 (not tuned) |
| **TFT** | `hidden_size` [16,64], `attention_heads` {1,2,4}, `lstm_layers` [1,2], `dropout` [0.0,0.3], `learning_rate` [1e-4,1e-2 log], `batch_size` [32,128] | `context_length` fixed at 64 |
| **XGBoost** | `max_depth` [3,10], `learning_rate` [0.01,0.3 log], `n_estimators` [100,600], `subsample` [0.5,1.0], `colsample_bytree` [0.5,1.0] | feature lags/windows fixed via `data.yaml` (1/2/5/10/20-day lags, 5/20-day rolling windows) |
| **Chronos (both)** | none — zero-shot, no training | model size fixed to "small" variant; context truncated to last 512 observations |
| **Naive / Moving average** | none | moving-average window fixed at 20 days |

**Shared HPO budget** (`configs/models.yaml: hpo`, mode-dependent):
- `quick_mode`: 3 trials, 3 epochs (initial fit), patience 2, update_epochs 1 — fast, for testing.
- `full_mode` (default): 25 trials, 60 epochs, patience 8, update_epochs 5 — the production-quality run currently executing on RunPod.

## 6. What differentiates the models from each other

- **Statistical vs. learned vs. foundation model**: SARIMA/SARIMAX are classical linear statistics (MLE-fit, no learned representation); PatchTST/TFT are trained neural networks (gradient descent, learn nonlinear patterns from this data specifically); XGBoost is a learned ensemble of decision trees (nonlinear, but axis-aligned splits, not gradient-based representation learning); Chronos is a **pretrained foundation model** used zero-shot — it learned general time-series patterns from a huge external corpus and is never trained on this gold-price data at all.
- **Univariate vs. multivariate**: SARIMA/PatchTST/Chronos see only the gold price history. SARIMAX/XGBoost/TFT additionally see the 6 exogenous series.
- **Retraining behavior per rolling window**: SARIMA/SARIMAX/XGBoost fully re-fit from scratch every 20-day window (cheap enough for these model types). PatchTST/TFT warm-start (continue training existing weights) rather than reinitializing, since full retraining of a neural net at every window would be prohibitively expensive.
- **Scaling**: PatchTST/TFT scale all inputs (`StandardScaler`, fit on training data only). SARIMAX scales only the exogenous regressors, not the target. XGBoost needs no scaling (tree splits are scale-invariant). SARIMA needs no scaling (works directly on price levels). Chronos does its own internal normalization.
- **Training cost profile**: Chronos is the cheapest (no training, just inference). SARIMA/SARIMAX/XGBoost are cheap-to-moderate (each of the ~13×26 refits — 25 trials + 1 final, ×13 rolling windows — takes seconds). PatchTST/TFT are the most expensive by far, since even a "warm-start" update still runs several epochs of backprop at every one of ~13 windows, for every one of 25 HPO trials.

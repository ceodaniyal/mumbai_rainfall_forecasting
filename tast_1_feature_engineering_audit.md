# Feature-Engineering Audit — Why the XGBoost Test RMSE of 58.22 mm Is Not Valid

**Scope:** `rain_forecasting.ipynb`, ML section (Random Forest + XGBoost).
**Question:** Is the reported XGBoost test RMSE of **58.22 mm** (2011–2021) a real result, and is the current "simple lag" feature construction adequate — or should we move to a multi-lag + rolling-window pipeline?

---

## TL;DR

1. **The 58.22 mm RMSE is not mathematically valid. It is target leakage**, caused by feeding `Rainfall_diff` into the model as a predictor. `Rainfall_diff` is a transform of the *target itself*, so the model is handed (a piece of) the answer.
2. **The "simple lag" features (`lag_1 … lag_12`) are *not* the cause of the flaw.** Created with `.shift(i)`, they use only past values and are perfectly legitimate. The leak comes exclusively from `Rainfall_diff`.
3. Once the leak is removed, the honest test RMSE is **≈191 mm (XGBoost)** and **≈195 mm (Random Forest)** — roughly on par with SARIMA (186 mm), not "dramatically better."
4. **We tested the proposed multi-lag + rolling-window pipeline. On this dataset it does *not* improve accuracy — it is slightly worse** (XGB RMSE 214.96). Mumbai rainfall is dominated by the annual monsoon cycle, which `Month` + `lag_12` already capture; extra rolling features mostly add noise. The pipeline is still worth adopting for **engineering hygiene** (it structurally prevents this class of leak), just not as an accuracy lever.

---

## 1. How the features are built today

```python
# Cell 15 — creates the differenced series
df_melt['Rainfall_diff'] = df_melt['Rainfall'].diff()      # Rainfall_diff[t] = Rainfall[t] - Rainfall[t-1]
df_melt.dropna(inplace=True)

# Cell 30 — "simple" lag features (one .shift() per lag)
for i in range(1, 13):
    df_ml[f'lag_{i}'] = df_ml['Rainfall'].shift(i)          # lag_i[t] = Rainfall[t-i]   (past-only, OK)

# Cell 33 — everything except the target becomes X
X_train = train_ml.drop('Rainfall', axis=1)                # <-- silently includes Rainfall_diff
y_train = train_ml['Rainfall']
```

The lag block is the textbook "simple implementation using pandas": shift the target column once per lag. That part is fine. The problem is the line above it and how `X` is assembled.

---

## 2. The actual flaw — target leakage via `Rainfall_diff`

By definition,

```
Rainfall_diff[t] = Rainfall[t] − Rainfall[t−1]
```

and `lag_1[t] = Rainfall[t−1]`. Both are columns in `X`. The model's target is `y[t] = Rainfall[t]`. Therefore:

```
y[t]  =  Rainfall_diff[t]  +  lag_1[t]
```

**The label is an exact sum of two input features.** The model is not forecasting the future — it can algebraically reconstruct the answer from information that already contains it.

This is verified numerically: across the entire dataset,

```
max | Rainfall − (Rainfall_diff + lag_1) |  =  1.1e-13     # i.e. zero, to machine precision
```

### Why the RMSE is 58 and not 0
If a linear model had these features it would score ~0 RMSE. XGBoost and Random Forest are *tree* models: they approximate `Rainfall_diff + lag_1` with piecewise-constant splits rather than computing the sum exactly, so a small residual (~58 mm) remains. The number *looks* like a real error metric, which is exactly what makes the leak dangerous — it is quietly optimistic, not obviously broken.

### Why this invalidates the test score
At genuine forecast time, `Rainfall_diff[t]` **cannot be computed**, because it needs `Rainfall[t]` — the value we are trying to predict. A feature that is unavailable in production cannot legitimately be used to measure production performance. The 58.22 mm therefore does **not** estimate real out-of-sample skill.

### The smoking gun: train/serve skew
The future-forecast cell couldn't use the real `Rainfall_diff` either, so it silently **redefined** it:

```python
row['Rainfall_diff'] = recent_rainfall[-1] - recent_rainfall[-2]   # = lag_1 − lag_2, NOT y − lag_1
```

So the feature means one thing in training/testing (`y − lag_1`) and a **completely different thing** at forecast time (`lag_1 − lag_2`). That inconsistency is the same bug viewed from the serving side, and it confirms the feature was never legitimately usable.

**Verdict: the 58.22 mm RMSE is a logical/statistical artifact of leakage. It is not valid.**

---

## 3. Are the "simple" lag features themselves the problem?

**No.** This is the key nuance. `lag_1 … lag_12` are built with `.shift(i)`, `i > 0`, so every lag value comes strictly from the *past* (`lag_i[t] = Rainfall[t−i]`). None of them contain `Rainfall[t]`. They are legitimate predictors and, on their own, give an honest RMSE of ≈191 mm.

What the leak really exposes is a **process** weakness, not a "lags are too simple" weakness: features were mutated onto the dataframe in place, and `X` was assembled with a blanket `drop('Rainfall')`. That pattern makes it easy for a target-derived column (`Rainfall_diff`) to slip into the feature set unnoticed. The fix for *correctness* is simply:

```python
# Only use values known BEFORE month t. Enumerate features explicitly — never blanket-drop.
feature_cols = ['Month'] + [f'lag_{i}' for i in range(1, 13)]
X_train, X_test = train_ml[feature_cols], test_ml[feature_cols]
```

(`Rainfall_diff` can stay in the dataframe for the SARIMA stationarity / ACF–PACF analysis — it just must never enter `X`.)

---

## 4. The recommended pattern — a leak-safe multi-lag + rolling-window pipeline

Even though it isn't required to fix the metric, a dynamic feature wrapper is better engineering than hand-writing 12 `.shift()` calls: it is parameterized, reproducible, and — most importantly — makes the leak-safety rule explicit in one place.

```python
def build_features(series, lags=(1, 2, 3, 6, 12, 24), windows=(3, 6, 12)):
    """Leak-safe multi-lag + rolling feature builder. Every feature uses PAST data only."""
    f = pd.DataFrame({'Rainfall': series})
    mth = f.index.month
    f['Month']     = mth
    f['month_sin'] = np.sin(2 * np.pi * mth / 12)   # cyclical calendar encoding
    f['month_cos'] = np.cos(2 * np.pi * mth / 12)

    # multiple lag ranges
    for L in lags:
        f[f'lag_{L}'] = f['Rainfall'].shift(L)

    # rolling statistics — CRITICAL: shift(1) first so the window ends at t-1
    # and NEVER includes Rainfall[t]. Rolling on the raw column would re-introduce
    # exactly the same leak as Rainfall_diff.
    past = f['Rainfall'].shift(1)
    for w in windows:
        f[f'rmean_{w}'] = past.rolling(w).mean()
        f[f'rstd_{w}']  = past.rolling(w).std()
        f[f'rmin_{w}']  = past.rolling(w).min()
        f[f'rmax_{w}']  = past.rolling(w).max()

    return f.dropna()
```

> **The single most important rule for rolling features:** compute them on `series.shift(1)`, not on `series`. A rolling mean that includes the current month contains `Rainfall[t]` and leaks the target just like `Rainfall_diff` did. "Multi-lag + rolling" is only an *upgrade* if it is done this way; done naively it is the same bug with more columns.

---

## 5. Benchmark — does richer feature engineering actually help?

All rows below use the **identical** static split (train 1901–2010, test 2011–2021, 132 test months) and the notebook's XGBoost hyperparameters. Numbers are reproducible from the appendix scripts.

| Feature set | RF — RMSE | XGB — MAE | XGB — RMSE | Valid? |
|---|---:|---:|---:|:--:|
| **Leaked**: Month + `Rainfall_diff` + lag₁…₁₂ | 76.05 | 24.53 | **58.22** | ❌ leakage |
| **Corrected simple lags**: Month + lag₁…₁₂ | 195.27 | 98.34 | **191.19** | ✅ |
| **Multi-lag + rolling** (21 features) | 198.48 | 110.71 | **214.96** | ✅ |
| *Reference — Seasonal-naive (lag₁₂)* | — | — | 271.96 | ✅ |
| *Reference — SARIMA* | — | — | 186.27 | ✅ |

*(Leaked XGB RMSE reproduces to ~61 mm on xgboost 3.3.0 vs the notebook's stored 58.22 — same invalid configuration, tiny version difference. RF, being deterministic, matches the notebook's 76.05 exactly.)*

### Interpretation — an honest and slightly counter-intuitive result
Adding multiple lag ranges and rolling statistics did **not** improve accuracy; XGBoost got modestly **worse** (214.96 vs 191.19). The reason is domain-specific:

- Mumbai rainfall is almost entirely driven by the **annual monsoon cycle**. `Month` (or `lag_12`, the same month last year) already explains the overwhelming majority of the variance.
- Once seasonality is captured, month-to-month rainfall is close to noise. Rolling means/std/min/max over a ~1,300-row monthly series add dimensionality and variance without new signal, so a tree model overfits slightly and generalizes a touch worse.
- Note the **corrected models barely beat SARIMA and only modestly beat the seasonal-naive baseline** — the true, honest picture. This is why Tasks 2–3 (walk-forward validation + seasonal-naive comparison) matter: they tell you whether the ML model earns its complexity, and here the answer is "marginally, at best."

---

## 6. Recommendations

1. **Fix the metric (mandatory):** remove `Rainfall_diff` from `X`. Assemble features by an explicit allow-list of past-only columns; never blanket-`drop('Rainfall')`. Retrain → corrected RMSE ≈ **191 mm (XGB)**, **195 mm (RF)**.
2. **Adopt the leak-safe `build_features` wrapper for hygiene, not for accuracy.** Its value is that it centralizes the "past-only / `shift(1)` before rolling" rule and structurally prevents the exact leak we just found. Do not expect it to lower RMSE on this dataset.
3. **Don't chase the 58 mm number.** It was never achievable. Set stakeholder expectations at ~190 mm and validate honestly with walk-forward backtesting against the seasonal-naive baseline.
4. **Regenerate the deliverables** (`rainfall_forecast_next_12_months.csv`, `feature_importance.csv`) *after* the leak fix — the current CSVs were produced by the leaked model and by the inconsistent `lag_1 − lag_2` forecast hack.

---

## Appendix — reproduction

- Leak identity check: `max|Rainfall − (Rainfall_diff + lag_1)| = 1.1e-13`.
- All RMSE/MAE figures produced by the benchmark script comparing the three feature sets on the same 2011–2021 test window (leaked / corrected-simple / multi-lag+rolling), plus seasonal-naive and SARIMA references.
- Environment: pandas 2.2.3, scikit-learn 1.7.1, xgboost 3.3.0. Random Forest and XGBoost use `random_state=42`.

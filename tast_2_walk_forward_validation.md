# Walk-Forward Validation Report — Backtesting the Corrected XGBoost Model

**Scope:** `rain_forecasting.ipynb`, Task 2 (section "20. Walk-Forward Validation").
**Question:** How does the *leak-free* XGBoost model actually perform when evaluated the way it would run in production — retrained periodically and forecasting a full year ahead — instead of on a single static split?

---

## TL;DR

1. A single static 1901–2010 / 2011–2021 split is fragile: it depends on one arbitrary boundary and quietly assumes you always know **last month's actual rainfall** when predicting this month.
2. We replaced it with **walk-forward validation** (expanding window, iterative 12-month-ahead forecasting), retraining once per year from 2011 to 2021.
3. **Backtested XGBoost: overall RMSE = 212.96 mm, MAE = 109.90 mm** across all 132 forecast months.
4. That is **~22 mm worse than the static-split RMSE of 191.19 mm** — and this larger number is the *honest* one. The gap is exactly the optimism the static split was hiding.
5. XGBoost still **beats the seasonal-naive baseline** (RMSE 271.96 / MAE 136.03), so the model adds value — but the margin is modest and year-to-year performance is volatile (RMSE ≈ 130–283 mm).

---

## 1. Why a static split is not enough

The Task 1 evaluation trained once on 1901–2010 and scored all of 2011–2021 in one shot. Two problems:

- **Boundary sensitivity.** All conclusions rest on a single train/test cut. A different cut can give a materially different RMSE, and there is no sense of variance.
- **Hidden 1-step-ahead leakage of realism.** For every test month *t*, the features `lag_1 … lag_12` were filled with **actual** historical values — including `lag_1[t] = Rainfall[t−1]`, the *real* previous month. In production you do **not** know last month's rainfall when you issue a 12-month forecast; you only know data up to the forecast origin. So the static RMSE flatters the model by feeding it information it would not have.

Walk-forward validation fixes both: it retrains on a rolling basis and forces the model to forecast a full year using only information available at the forecast origin.

---

## 2. The protocol implemented

**Expanding window, iterative (recursive) 12-month-ahead.** For each forecast year *F* = 2011 … 2021:

1. **Retrain** XGBoost on *all* actual data up to **December *F − 1***.
2. **Recursively forecast** the 12 months of *F*:
   - Predict January *F* from the last 12 actual months.
   - Append that prediction to the history and use it as `lag_1` for February.
   - Continue through December — later months lean increasingly on the model's own predictions.
   - `lag_12` (same month, previous year) is always an **actual** value, so the dominant seasonal signal remains anchored to reality.
3. **Record** every (actual, predicted) pair.

This yields 11 annual retrains × 12 months = **132 forecast points**, the same span as the static test set, so the two are directly comparable.

```python
feature_cols = ['Month'] + [f'lag_{i}' for i in range(1, 13)]

records = []
for F in range(2011, 2022):                               # forecast years 2011..2021
    train_end = f"{F-1}-12-31"
    train = make_supervised(series)
    train = train[train.index <= train_end]               # expanding window
    model = new_xgb().fit(train[feature_cols], train['Rainfall'])

    history = series[series.index <= train_end].tolist()  # actuals up to Dec (F-1)
    for month in range(1, 13):
        row = {'Month': month, **{f'lag_{i}': history[-i] for i in range(1, 13)}}
        pred = max(float(model.predict(pd.DataFrame([row])[feature_cols])[0]), 0.0)
        history.append(pred)                              # <-- iterative feedback
        date = pd.Timestamp(year=F, month=month, day=1)
        records.append({'Date': date, 'y_true': float(series.get(date)), 'y_pred': pred})
```

> **Design choice — recursive, not "cheating" one-step.** Because the task asks to *"forecast the next 12 months iteratively,"* each month's prediction is fed back as the next month's lag. This is the realistic water-utility scenario (you forecast the year, then wait for it to happen). It is deliberately harder than reusing actual lags month by month.

---

## 3. Results

### Overall (132 months, 2011–2021)

| Metric | **Walk-forward XGBoost** | Static split (Task 1) | Seasonal-naive (lag₁₂) | SARIMA |
|---|---:|---:|---:|---:|
| **RMSE (mm)** | **212.96** | 191.19 | 271.96 | 186.27 |
| **MAE (mm)** | **109.90** | — | 136.03 | — |

### Per-year breakdown

| Forecast year | RMSE (mm) | MAE (mm) |
|---|---:|---:|
| 2011 | 226.52 | 103.75 |
| 2012 | 130.39 | 73.95 |
| 2013 | 231.49 | 114.38 |
| 2014 | 263.61 | 121.41 |
| 2015 | 233.28 | 121.03 |
| 2016 | 145.94 | 72.90 |
| 2017 | 141.72 | 90.18 |
| 2018 | 180.57 | 120.65 |
| 2019 | 282.70 | 137.60 |
| 2020 | 243.89 | 114.87 |
| 2021 | 199.18 | 138.17 |

---

## 4. Interpretation

- **Walk-forward RMSE (212.96) > static-split RMSE (191.19).** This is the expected and desirable direction. The extra ~22 mm is the error the static split hid by handing the model actual lags. The backtested figure is the number to quote to stakeholders and to size dashboards/reservoir buffers against.
- **Errors compound within each year.** January forecasts (all-actual lags) are the most accurate; by mid-year the model is forecasting off its own predictions, so residuals accumulate. This is inherent to recursive multi-step forecasting.
- **Performance is volatile across folds (RMSE ≈ 130–283 mm).** Mumbai rainfall has large inter-annual monsoon variability; a single static number understates how much the model's accuracy swings year to year. Reporting the *distribution*, not just one point estimate, is the main value of backtesting.
- **The model still beats the seasonal-naive baseline** (212.96 vs 271.96 RMSE; 109.90 vs 136.03 MAE). So XGBoost is adding genuine skill over "assume this year = last year," though the improvement is moderate rather than dramatic. A formal statistical comparison against the baseline is Task 3.

---

## 5. Recommendations

1. **Report the backtested metric (≈213 mm RMSE), not the static 191 mm**, as the model's expected production accuracy.
2. **Publish the per-year spread**, not just the aggregate — the ~130–283 mm range is material for planning.
3. **Retrain on a fixed cadence** (this backtest assumes annual retraining on all history). Keep the retraining schedule in production identical to the one validated here.
4. **Keep the seasonal-naive baseline in the monitoring loop** (Task 3). If the live model ever fails to beat lag₁₂, that is a signal to investigate or fall back.
5. Recursive error growth suggests **direct multi-horizon models** (a separate model per lead time) as a possible future improvement, though the seasonality-dominated signal caps how much is achievable.

---

## Appendix — reproduction

- Section "20. Walk-Forward Validation" in `rain_forecasting.ipynb` contains the full loop; the printed output there matches the numbers in this report.
- Same leak-free feature set as the corrected Task 1 pipeline (`Month` + `lag_1 … lag_12`), same XGBoost hyperparameters (`n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42`).
- Environment: pandas 2.2.3, scikit-learn 1.7.1, xgboost 3.3.0.

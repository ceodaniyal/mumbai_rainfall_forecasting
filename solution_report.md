# Solution Report — Issue #12: Fix Suspicious ML Metrics, Walk-Forward Validation, Naive Baseline

**TL;DR:** The XGBoost test RMSE of **58.22 mm was invalid** — a target-leakage artifact, not forecasting skill. After fixing the leak and validating honestly with walk-forward backtesting, XGBoost scores **~213 mm RMSE**. It still beats a seasonal-naive baseline by ~22 % **and the gap is statistically significant**, so the model adds real value — but the true performance is ~3.7× worse than advertised.

Detailed write-ups: [`tast_1_feature_engineering_audit.md`](tast_1_feature_engineering_audit.md), [`tast_2_walk_forward_validation.md`](tast_2_walk_forward_validation.md), [`tast_3_seasonal_naive_baseline.md`](tast_3_seasonal_naive_baseline.md).

---

## 1. Model audit — the 58.22 mm RMSE is not valid (Task 1)

**Root cause: target leakage via `Rainfall_diff`.** The feature `Rainfall_diff` was fed to the models, but by definition

```
Rainfall_diff[t] = Rainfall[t] − Rainfall[t−1] = y[t] − lag_1[t]   ⟹   y[t] = Rainfall_diff[t] + lag_1[t]
```

The target is an exact sum of two input features (verified: `max|y − (Rainfall_diff + lag_1)| = 1.1e-13`). The model was handed the answer. The score isn't 0 only because tree ensembles approximate the sum with discrete splits. Crucially, `Rainfall_diff[t]` needs `Rainfall[t]` — unavailable at forecast time — so the metric never measured real performance. (The future-forecast cell even silently redefined the feature as `lag_1 − lag_2`, a train/serve mismatch that confirms the bug.)

**Fix:** drop `Rainfall_diff` from the feature matrix; use only past-only predictors — `Month` + `lag_1 … lag_12`. Models retrained.

| Model | Reported (leaked) RMSE | **Corrected RMSE** (static split) |
|---|---:|---:|
| Random Forest | 76.05 | **195.27** |
| XGBoost | **58.22** | **191.19** |

*Note:* the `lag_1 … lag_12` features themselves were never leaky (they use `.shift()`, past-only). We also tested a richer multi-lag + rolling-window pipeline; on this monsoon-dominated series it did **not** improve accuracy (XGB 214.96), because `Month` + `lag_12` already capture the dominant seasonal signal.

---

## 2. Walk-forward validation (Task 2)

The static split is fragile and optimistic (it lets every test month use the *actual* previous month as `lag_1`). We replaced it with **expanding-window, iterative 12-month-ahead backtesting**: for each year *F* = 2011…2021, retrain on all data up to Dec *F−1*, then recursively forecast the 12 months of *F* (predictions feed back as lags; `lag_12` stays actual).

| | Static split | **Walk-forward (backtest)** |
|---|---:|---:|
| XGBoost RMSE | 191.19 | **212.96** |
| XGBoost MAE | — | **109.90** |

The backtested RMSE is **higher and more honest** — the ~22 mm gap is the optimism the static split hid. Per-year RMSE is volatile (~130–283 mm), reflecting high inter-annual monsoon variability.

---

## 3. Model comparison & seasonal-naive baseline (Task 3)

**Baseline:** seasonal-naive — rainfall for month *m* of year *Y* = actual rainfall of month *m* of year *Y−1* (`lag_12`), evaluated on the same walk-forward window.

| Model | RMSE (mm) | MAE (mm) | Protocol |
|---|---:|---:|---|
| ~~XGBoost (leaked)~~ | ~~58.22~~ | ~~24.53~~ | ❌ invalid |
| SARIMA | 186.27 | — | static split |
| XGBoost (corrected) | 191.19 | 98.34 | static split |
| Random Forest (corrected) | 195.27 | 96.67 | static split |
| **XGBoost (corrected)** | **212.96** | **109.90** | **walk-forward** |
| Seasonal-naive (`lag_12`) | 271.96 | 136.03 | walk-forward |

**Does ML add statistical value over the naive baseline? Yes.** On the identical walk-forward window, XGBoost improves RMSE by **21.7 %** and MAE by **19.2 %**, and the improvement is significant:

- **Diebold–Mariano test** (HLN-corrected, h = 12): *p* = **0.0033** (squared-error loss), *p* = **0.0115** (absolute-error loss).
- XGBoost wins **9 of 11** forecast years; the 2 losses are years whose rainfall closely repeated the prior year.
- Conservative annual-level **Wilcoxon signed-rank** test: *p* = **0.0337**.

---

## 4. Conclusions & recommendations

1. **Do not deploy on the 58 mm figure** — it was leakage. Realistic performance is **~213 mm RMSE** (walk-forward).
2. **XGBoost earns its place**: statistically significant ~20 % improvement over seasonal-naive. But the margin is moderate and comparable to SARIMA, so set stakeholder expectations accordingly.
3. **Keep the seasonal-naive baseline in production monitoring** as a floor — if the live model stops beating `lag_12`, investigate or fall back.
4. **Report per-year skill, not just the aggregate** — value is uneven across years.
5. **Biggest lever for future gains**: exogenous drivers (ENSO/IOD climate indices) or direct multi-horizon models — endogenous feature tweaks alone won't move the needle much.

---

## Deliverables status

| Deliverable | Status |
|---|---|
| Corrected feature-engineering pipeline in `rain_forecasting.ipynb` | ✅ leak removed, models retrained |
| Walk-forward validation loop (§20) | ✅ |
| Seasonal-naive baseline + significance tests (§21) | ✅ |
| `rainfall_forecast_next_12_months.csv` regenerated from corrected model | ⏳ pending |
| `feature_importance.csv` regenerated from corrected model | ⏳ pending |
| Summary report | ✅ this file |

*Environment: pandas 2.2.3, scikit-learn 1.7.1, xgboost 3.3.0. All models use `random_state=42`.*

# Seasonal-Naive Baseline Report — Does the ML Model Actually Add Value?

**Scope:** `rain_forecasting.ipynb`, Task 3 (section "21. Seasonal-Naive Baseline & Statistical Comparison").
**Question:** Evaluate a seasonal-naive baseline on the same 2011–2021 walk-forward window, and determine whether the corrected XGBoost model adds *statistically significant* value over it.

---

## TL;DR

1. The right baseline for a strongly seasonal series is the **seasonal-naive** forecast: rainfall for month *m* of year *Y* = actual rainfall for month *m* of year *Y − 1* (i.e. **`lag_12`**). No model, no recursion — `lag_12` is always observed at the forecast origin.
2. On the identical walk-forward window: **seasonal-naive RMSE = 271.96 mm, MAE = 136.03 mm**.
3. **XGBoost beats it: RMSE 212.96 mm (−21.7 %), MAE 109.90 mm (−19.2 %).**
4. The improvement is **statistically significant** — Diebold–Mariano test: *p* = 0.0033 (squared-error loss) and *p* = 0.0115 (absolute-error loss); XGBoost wins **9 of 11** years; a conservative annual-level Wilcoxon test gives *p* = 0.0337.
5. **Answer: yes, the ML model adds statistical value** — but the margin is moderate, not dramatic, and the baseline should stay in production as a monitoring floor.

---

## 1. The seasonal-naive baseline

Mumbai rainfall is dominated by the annual monsoon, so the hardest "free" competitor is simply **last year's value for the same month**:

```
naive_forecast[month m, year Y] = actual_rainfall[month m, year Y − 1]   ==   lag_12
```

Why this is the correct baseline (and why it needs no walk-forward machinery):

- It encodes the single strongest pattern in the data (12-month seasonality) with zero parameters.
- For any month in forecast year *F*, `lag_12` falls in year *F − 1*, which is fully observed before the forecast is issued. So the baseline is always available at the forecast origin — unlike `lag_1 … lag_11`, which the ML model has to *predict* recursively.
- If a trained model cannot beat this, it is not learning anything useful.

```python
# lag_12 is always an actual value at the forecast origin -> no recursion required
wf_results['naive'] = series.shift(12).reindex(wf_results['Date']).values
```

---

## 2. Evaluation on the same walk-forward window (2011–2021, 132 months)

| Model | RMSE (mm) | MAE (mm) |
|---|---:|---:|
| Seasonal-naive (`lag_12`) | 271.96 | 136.03 |
| **XGBoost (walk-forward)** | **212.96** | **109.90** |
| **Skill score** (1 − model/naive) | **+21.7 %** | **+19.2 %** |

XGBoost lowers both error metrics by roughly one-fifth. But a point-estimate gap is not proof — the two seasonal-naive "wins" below show the baseline is genuinely competitive in some years, so we test for significance rather than eyeballing the averages.

### Per-year RMSE

| Forecast year | XGBoost | Seasonal-naive | Winner |
|---|---:|---:|:--|
| 2011 | 226.52 | 95.48 | naive |
| 2012 | 130.39 | 264.06 | XGB |
| 2013 | 231.49 | 299.39 | XGB |
| 2014 | 263.61 | 306.55 | XGB |
| 2015 | 233.28 | 397.25 | XGB |
| 2016 | 145.94 | 251.54 | XGB |
| 2017 | 141.72 | 101.52 | naive |
| 2018 | 180.57 | 235.06 | XGB |
| 2019 | 282.70 | 321.12 | XGB |
| 2020 | 243.89 | 264.42 | XGB |
| 2021 | 199.18 | 302.45 | XGB |

**XGBoost wins 9 of 11 years.** The two losses (2011, 2017) are years whose rainfall closely repeated the previous year — exactly the regime where "assume this year = last year" is very hard to beat.

---

## 3. Statistical comparison — is the improvement significant?

A lower average error could still be luck. To answer *"does ML add **statistical** value?"* we use the **Diebold–Mariano (DM) test**, the standard test for comparing the accuracy of two forecasts on the same series. It works on the per-month **loss differential** `d_t = loss(XGB)_t − loss(naive)_t` and, crucially, uses a HAC (long-run) variance so it accounts for the autocorrelation induced by overlapping 12-month-ahead forecasts. We apply the Harvey–Leybourne–Newbold small-sample correction (h = 12).

| Test | Loss | Statistic | one-sided *p* | Result |
|---|---|---:|---:|---|
| Diebold–Mariano | squared error | −2.758 | **0.0033** | XGBoost significantly better (1 %) |
| Diebold–Mariano | absolute error | −2.299 | **0.0115** | XGBoost significantly better (5 %) |
| Per-year win count | RMSE | 9 / 11 | — | XGBoost |
| Wilcoxon signed-rank (annual MAE) | — | — | **0.0337** | XGBoost significantly better (5 %) |

A negative DM statistic means the first forecast (XGBoost) carries lower loss. Both DM tests reject the null of equal accuracy in XGBoost's favour. As a **robustness check** that does not rely on the DM autocorrelation model, we also run a non-parametric **Wilcoxon signed-rank test** on the 11 near-independent *per-year* MAEs (one value per retraining origin); it agrees (*p* = 0.0337).

> Why two tests? Monthly rainfall errors are highly non-normal and autocorrelated within a forecast year. DM handles the autocorrelation via its HAC variance; the annual-level Wilcoxon sidesteps both issues by aggregating to independent yearly numbers. Both pointing the same way makes the conclusion robust.

---

## 4. Verdict

**Yes — the machine-learning model adds statistically significant value over the seasonal-naive baseline.** It reduces RMSE by ~22 % and MAE by ~19 %, wins 9 of 11 years, and clears conventional significance thresholds on two independent tests.

Caveats worth stating honestly:

- **The margin is moderate, not transformational.** Honest walk-forward RMSE (~213 mm) is only about one-fifth below the naive floor and is in the same ballpark as SARIMA's static-split RMSE (186 mm — different protocol, so not directly comparable).
- **The baseline is competitive in "persistent" years.** When a year's rainfall resembles the prior year (2011, 2017), the naive model wins outright. The ML model's advantage comes from the *other* years, where it corrects large year-on-year swings.
- The corrected XGBoost is a defensible production model precisely *because* it was validated this way — against a real baseline, with a real significance test, on a realistic backtest — rather than on the leaky 58 mm figure that started this investigation.

---

## 5. Recommendations

1. **Ship XGBoost**, but report the walk-forward metrics (RMSE ≈ 213 mm, MAE ≈ 110 mm) and the skill scores (~+20 %), not the invalid 58 mm.
2. **Keep the seasonal-naive baseline permanently in production monitoring.** It is free to compute; if the live model ever fails to beat `lag_12` over a rolling window, trigger an investigation or fall back to the baseline.
3. **Track per-year (not just aggregate) skill.** The value is uneven across years; a single number hides the 2011/2017-style regimes.
4. Given the moderate margin, treat further modelling effort (exogenous drivers such as ENSO/IOD indices, direct multi-horizon models) as the path to a *materially* better model — incremental feature tweaks on the endogenous series alone are unlikely to move the needle much (see Task 1's multi-lag/rolling experiment).

---

## Appendix — reproduction

- Section "21. Seasonal-Naive Baseline & Statistical Comparison" in `rain_forecasting.ipynb`; printed output there matches every figure in this report.
- Seasonal-naive = `series.shift(12)` on the same 2011–2021 window used in Task 2.
- DM test: HLN small-sample correction, autocovariances to lag h−1 = 11, one-sided alternative "XGBoost more accurate". Wilcoxon: `scipy.stats.wilcoxon(alternative='less')` on 11 per-year MAEs.
- Environment: pandas 2.2.3, scikit-learn 1.7.1, xgboost 3.3.0, scipy (bundled with scikit-learn).

# Rain Forecasting — Mumbai

A machine-learning project to forecast monthly rainfall for Mumbai, India, using 121 years of historical data (1901 – 2021). Built for **Spinnaker Analytics** as part of the Data Science & AI track.

---

## Project goal

Mumbai receives ~90% of its annual rainfall during the four-month southwest monsoon (June – September). Anticipating monthly rainfall — even a few months ahead — helps the city's water utility plan reservoir releases, schedule infrastructure maintenance, and trigger demand-management protocols before crisis hits.

This project benchmarks five forecasting approaches and ships a working 12-month forward forecast.

---

## Results at a glance

| Rank | Model | MAE (mm) | RMSE (mm) |
|-----:|-------|---------:|----------:|
| 1 | **XGBoost** ★ | **24.53** | **58.22** |
| 2 | Random Forest | 29.36 | 76.05 |
| 3 | SARIMA(1,1,1)(1,1,1,12) | 100.88 | 186.27 |
| 4 | LSTM | 112.61 | 204.43 |
| 5 | Prophet | 171.16 | 284.69 |

XGBoost wins by a wide margin — a ~69% RMSE reduction over the SARIMA baseline. The strongest single feature is `lag_12` (rainfall in the same calendar month a year ago), which carries ~48% of the model's predictive power.

---

## Repository structure

```
Rain Forecasting/
├── rain_forecasting.ipynb              # Full Jupyter notebook (EDA + modeling)
├── rain_forecasting_report.docx        # Final report (Word document)
├── rain_forecasting_presentation.pptx  # Final presentation (PowerPoint)
├── mumbai-monthly-rains.csv            # Source data — Mumbai monthly rainfall
├── rainfall_forecast_next_12_months.csv# 12-month forward forecast (XGBoost)
├── feature_importance.csv              # Ranked feature importances
├── xgboost_rainfall_model.pkl          # Trained XGBoost model
├── sarima_model.pkl                    # Trained SARIMA model
├── forecast_plot.png                   # Forecast visualisation
├── link.txt                            # Source URL for the dataset
└── README.md                           # This file
```

---

## Dataset

- **Source:** OpenCity India — `data.opencity.in/dataset/mumbai-rainfall-data`
- **Period:** 1901 – 2021 (121 years)
- **Frequency:** Monthly
- **Records:** 1,452 monthly observations
- **Target:** Rainfall (mm)
- **Annual mean:** ~2,150 mm

---

## Methodology (in 10 steps)

1. Load the wide-format CSV.
2. Drop the `Total` column; melt to long format (`Date`, `Month`, `Rainfall`).
3. Exploratory analysis — time-series plot, monthly seasonality boxplots.
4. Stationarity check via Augmented Dickey-Fuller (raw + first-differenced).
5. Feature engineering — `lag_1` … `lag_12`, `Month`, `Rainfall_diff`.
6. Time-based train-test split — train 1901 – 2010, test 2011 – 2021.
7. Train SARIMA, Random Forest, XGBoost, Prophet, and an LSTM.
8. Evaluate on the held-out test window using MAE and RMSE.
9. Generate a 12-month iterative forward forecast with the best model.
10. Persist the trained models and forecast outputs to disk.

---

## How to run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost prophet tensorflow
```

### Reproduce

1. Open `rain_forecasting.ipynb` in Jupyter / VS Code / Colab.
2. Run cells top to bottom.
3. The notebook will regenerate `feature_importance.csv`, `rainfall_forecast_next_12_months.csv`, and the `.pkl` model files.

### Use the trained model directly

```python
import pickle
import pandas as pd

with open("xgboost_rainfall_model.pkl", "rb") as f:
    model = pickle.load(f)

# Build the same feature row used during training:
# columns = ['Month', 'Rainfall_diff', 'lag_1', 'lag_2', ..., 'lag_12']
# then call model.predict(X)
```

---

## Limitations

- **Climate non-stationarity** — the model can't extrapolate beyond regimes seen in the 1901-2021 training data.
- **Iterative forecast drift** — long-horizon iterative prediction can shift the seasonal peak slightly.
- **Univariate features** — only past rainfall is used. Adding ENSO / IOD indices and SST anomalies would likely improve skill on anomalous years.
- **Monthly resolution** — flood-warning use cases need a daily model.
- **Single train-test split** — walk-forward cross-validation would give a more robust estimate.

---

## Recommendations

- Use the 12-month forecast monthly to drive reservoir-release schedules across Mumbai's seven supply lakes.
- Auto-trigger demand-management protocols when forecast Jun-Sep volume falls below the 25th historical percentile.
- Concentrate pipeline and treatment-plant maintenance in months with forecast rainfall < 50 mm.
- Productionise the XGBoost model behind a forecasting API with monthly retraining and drift monitoring.
- Build a Power BI / Tableau dashboard showing forecast vs reservoir levels for monthly review meetings.

---

## Author

**Daniyal Khan** — daniyal.khan@growhut.in

Submitted to Spinnaker Analytics — April 2026.

---

## Notes

- Per the project brief, the source dataset is **not to be uploaded publicly** (GitHub / Kaggle / etc.).
- All deliverables (notebook, report, presentation) should be zipped together for final submission.

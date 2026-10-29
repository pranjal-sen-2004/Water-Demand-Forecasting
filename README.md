# Hourly Urban Water Demand Forecasting — SARIMA vs MSTL + ARIMA

Time Series Analysis (24AM7PCTSA) Lab AAT · Domain: Smart City

**Question:** does explicitly modelling daily *and* weekly seasonality improve day-ahead hourly water-demand
forecasts compared with a conventional SARIMA, on the same unseen data?

**Answer (test period 18 Jun 2020 – 1 Apr 2021, 6,912 rolling day-ahead forecasts):**

| Model | MAE (m³/h) | RMSE (m³/h) | MAPE (%) |
|---|---:|---:|---:|
| Seasonal naive, 24 h | 285.6 | 426.9 | 15.9 |
| Seasonal naive, 168 h | 283.4 | 408.1 | 14.7 |
| SARIMA(3,0,1)(0,1,1)₂₄ — baseline | 270.7 | 381.9 | 15.9 |
| MSTL (24 h only) + ARIMA — ablation | 288.8 | 406.9 | 16.8 |
| **MSTL (24 h + 168 h) + ARIMA — enhanced** | **199.1** | **277.7** | **11.2** |

MAE −26.4% (block-bootstrap 95% CI 21.8–31.9%), RMSE −27.3%; Diebold–Mariano p ≈ 2×10⁻²³.
Full numbers: `outputs/metrics/results.json`.

## Data

Weyrauch et al. (2024), *Hourly Water Demand of a Mixed District Recorded by Supplier*, Zenodo,
doi:10.5281/zenodo.11045013, CC-BY-4.0. The three CSVs are in `data/raw/` (MD5 checksums match Zenodo).
Modelling uses `waterconsumption_noNaN_fixedTimechange.csv`. Hours the publisher filled or altered are
flagged by comparison with the raw file.

## Reproduce

Run from this folder (paths are resolved relative to `src/config.py`):

```bash
pip install -r requirements.txt
python src/data_preprocessing.py     # audit + clean series   (~5 s)
python src/exploratory_analysis.py   # EDA figures + tests     (~25 s)
python src/experiments.py            # validation grid + one test run (~15 min on 12 workers)
python src/evaluation.py             # metrics, DM test, bootstrap, diagnostics (~30 s)
python report/build_report.py        # report/Lab_AAT_Report.docx
python presentation/build_slides.py  # presentation/Water_Demand_Forecasting.pptx
```

`python src/experiments.py --reuse-validation` skips the grid search and reuses the saved validation tables.
All models are deterministic; the bootstrap uses seed 42.

## Layout

```
data/raw/            Zenodo CSVs              data/processed/   cleaned hourly series + flags
src/config.py        paths, splits, horizon   src/common.py     loading, splits, metrics, plot style
src/data_preprocessing.py  audit              src/exploratory_analysis.py  EDA, ADF/KPSS, ACF/PACF, MSTL
src/models.py        SARIMA, MSTL-ARIMA, naive   src/experiments.py  validation selection + test run
src/evaluation.py    metrics, DM, bootstrap, plots
outputs/figures/     fig01–fig14              outputs/metrics/  JSON/CSV results   outputs/forecasts/
report/              build_report.py, literature.py, Lab_AAT_Report.docx
presentation/        build_slides.py, Water_Demand_Forecasting.pptx
blog/                blog_post.md, blog_post_blogger.html, blog_example.py, figures
SUBMISSION_GUIDE.md  viva Q&A, rubric audit, checklist, Blogger steps
```

## How to study this project

1. Read the question and results above, then the [report](report/Lab_AAT_Report.docx) (§7 EDA, §8–9 models, §11 results).
2. Look at the figures in order: `outputs/figures/fig01` → `fig14`. Each one is produced by one script.
3. Read the code in pipeline order: `config.py` → `data_preprocessing.py` → `exploratory_analysis.py` →
   `models.py` (the two forecasting methods) → `experiments.py` (validation, then one test run) → `evaluation.py`.
4. For a quick start, run [`blog/blog_example.py`](blog/blog_example.py): one self-contained file that builds
   both forecasts for one day.
5. Try extending it: add holidays, add a TBATS or double-seasonal Holt-Winters baseline, or evaluate prediction intervals.

Key ideas: multiple seasonality (24 h and 168 h), rolling-origin day-ahead evaluation without leakage,
identical validation rules for every model, Diebold–Mariano test, and an ablation study.

## Protocol in one paragraph

Chronological split in whole days: train 2016-01-01 → 2019-09-03, validation → 2020-06-17, test → 2021-04-01.
Every day at 00:00 each model forecasts the next 24 hours using only data up to 23:00 of the previous day.
SARIMA parameters are estimated on the last 4 weeks before the forecast period, and the Kalman filter is then
updated daily. MSTL-ARIMA refits on a trailing 12-week window each day. Both configurations were chosen by
lowest validation MAE, and the test set was evaluated once.

Library versions: Python 3.13.5, pandas 2.2.3, numpy 2.1.3, statsmodels 0.14.4, matplotlib 3.10.0, scipy 1.15.3, joblib 1.4.2.

## Licence

- **Data:** CC-BY-4.0, © the dataset authors. See [data/README.md](data/README.md) for attribution.
- **Code and documents:** add a licence of your choice before publishing (e.g. MIT for code). Without one,
  others may read the code but have no explicit permission to reuse it.

## Citation

If you use this work, please cite the dataset (Weyrauch et al., 2024, doi:10.5281/zenodo.11045013) and the
methods: Bandara, Hyndman & Bergmeir (2021) for MSTL, and statsmodels (Seabold & Perktold, 2010).

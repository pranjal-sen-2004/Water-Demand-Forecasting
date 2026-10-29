"""Verified literature matrix (checked 29 Oct 2026 against publisher,
repository or arXiv pages). 'Results' only states what the accessed source
shows; where only the abstract was accessible this is said explicitly."""

PAPERS = [
    {
        "key": "Weyrauch2024",
        "paper": "ReCycle: Fast and Efficient Long Time Series Forecasting with Residual Cyclic Transformers",
        "authors": "Weyrauch, Steens, Taubert, Hanke, Eqbal, Götz E., Streit, Götz M., Debus (2024)",
        "source": "arXiv:2405.03429 (IEEE CAI 2024); https://arxiv.org/abs/2405.03429",
        "objective": "Cut the compute cost of Transformer forecasters for long, cyclic series",
        "dataset": "Same Zenodo water dataset (hourly, 2016–2021) plus electricity/ETT benchmarks; "
                   "21 days input → 7 days (168 h) output; 6:2:2 split",
        "methods": "Transformer, FEDformer, PatchTST, NHiTS with/without ReCycle; RHP (recent historic profile)",
        "metrics": "MSE, MAPE",
        "results": "Water data (full text, Table I): RHP MAPE 13.9 %, best deep model PatchTST+ReCycle 13.8 %, "
                   "plain FEDformer lowest MSE; authors call the water series 'very noisy'",
        "limitations": "Deep models barely beat a simple profile on this data; no MAE/RMSE; no classical "
                       "statistical baseline (SARIMA/ETS) reported",
        "relevance": "Shows simple cycle-based structure is hard to beat on this exact dataset → motivates a "
                     "transparent statistical multi-seasonal approach",
    },
    {
        "key": "Arandia2016",
        "paper": "Tailoring Seasonal Time Series Models to Forecast Short-Term Water Demand",
        "authors": "Arandia, Ba, Eck, McKenna (2016)",
        "source": "J. Water Resour. Plann. Manage. 142(3), 04015067; doi:10.1061/(ASCE)WR.1943-5452.0000591",
        "objective": "Short-term urban water demand forecasting with SARIMA + data assimilation",
        "dataset": "Utility demand at 15-min, hourly and daily resolution",
        "methods": "SARIMA with daily and weekly periodicities; offline re-estimation vs online Kalman-filter updating",
        "metrics": "Normalised forecast deviation (abstract)",
        "results": "Abstract: errors ≈4.2 % to 1.3 %, falling with coarser granularity; weekly-seasonal models "
                   "remove autocorrelation more efficiently than daily-seasonal ones; online updating helps",
        "limitations": "Single utility; SARIMA handles one seasonal period per model (full text not accessed)",
        "relevance": "Direct precedent for our SARIMA baseline with Kalman-filter updating; its weekly-vs-daily "
                     "finding motivates modelling both cycles together",
    },
    {
        "key": "Pacchin2019",
        "paper": "A Comparison of Short-Term Water Demand Forecasting Models",
        "authors": "Pacchin, Gagliardi, Alvisi, Franchini (2019)",
        "source": "Water Resour. Manage. 33(4):1481–1497; doi:10.1007/s11269-019-02213-y",
        "objective": "Compare six short-term water demand models across real networks",
        "dataset": "Seven district-metered areas / networks, two years of hourly data",
        "methods": "ANN, pattern-based, two moving-window pattern models, Markov chain (probabilistic), naïve benchmark",
        "metrics": "Accuracy indices (abstract does not list them; full text not accessed)",
        "results": "Abstract: different techniques reach comparable medium–high accuracy; moving-window models "
                   "most robust and easiest to set up",
        "limitations": "Italian DMAs only; numerical scores not verified here",
        "relevance": "Supports a 24-h-ahead hourly protocol, a naïve benchmark and a moving (trailing) window design",
    },
    {
        "key": "Herrera2010",
        "paper": "Predictive Models for Forecasting Hourly Urban Water Demand",
        "authors": "Herrera, Torgo, Izquierdo, Pérez-García (2010)",
        "source": "J. Hydrology 387(1–2):141–150; doi:10.1016/j.jhydrol.2010.04.005",
        "objective": "Compare machine-learning predictors for hourly urban demand",
        "dataset": "Hourly consumption, hydraulic sector of a city in south-eastern Spain",
        "methods": "ANN, projection pursuit regression, MARS, random forests, SVR, weighted demand-profile model",
        "metrics": "Not stated in abstract (full text restricted)",
        "results": "Abstract: data highly non-linear; accuracy judged adequate for operational management; "
                   "best model not verified here",
        "limitations": "Single sector; ML models need feature engineering and give limited interpretability",
        "relevance": "Establishes hourly demand forecasting as an operational problem and the value of "
                     "demand-profile (seasonal pattern) information",
    },
    {
        "key": "Bandara2021",
        "paper": "MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns",
        "authors": "Bandara, Hyndman, Bergmeir (2021)",
        "source": "arXiv:2107.13462; https://arxiv.org/abs/2107.13462",
        "objective": "Decompose series with several seasonal cycles efficiently",
        "dataset": "Synthetic series and a perturbed real-world dataset",
        "methods": "MSTL (iterated STL per period) vs other decomposition benchmarks",
        "metrics": "Decomposition accuracy and computation time",
        "results": "Abstract: competitive accuracy with lower computational cost than benchmarks",
        "limitations": "A decomposition, not a forecaster by itself; seasonal components must be extrapolated",
        "relevance": "Method used for our enhancement (statsmodels MSTL implementation)",
    },
    {
        "key": "Taylor2003",
        "paper": "Short-Term Electricity Demand Forecasting Using Double Seasonal Exponential Smoothing",
        "authors": "Taylor (2003)",
        "source": "J. Oper. Res. Soc. 54(8):799–805; doi:10.1057/palgrave.jors.2601589",
        "objective": "Model intraday and intraweek cycles in utility demand",
        "dataset": "Half-hourly electricity demand (daily and weekly cycles)",
        "methods": "Double-seasonal Holt-Winters with AR error correction vs standard Holt-Winters and "
                   "double-seasonal multiplicative ARIMA",
        "metrics": "Forecast accuracy over lead times (per abstract)",
        "results": "Abstract: double-seasonal Holt-Winters outperformed both standard Holt-Winters and the "
                   "double-seasonal ARIMA",
        "limitations": "Electricity rather than water; parametric two-cycle structure",
        "relevance": "Classic evidence that modelling two seasonal cycles improves short-term utility demand forecasts",
    },
    {
        "key": "Donkor2014",
        "paper": "Urban Water Demand Forecasting: Review of Methods and Models",
        "authors": "Donkor, Mazzuchi, Soyer, Roberson (2014)",
        "source": "J. Water Resour. Plann. Manage. 140(2):146–159; doi:10.1061/(ASCE)WR.1943-5452.0000314",
        "objective": "Review urban water demand forecasting literature 2000–2010",
        "dataset": "Literature review",
        "methods": "Time-series, regression, econometric, ANN, hybrid models",
        "metrics": "—",
        "results": "ANNs common for short-term forecasting; econometric/scenario models for long-term planning; "
                   "calls for more probabilistic forecasting",
        "limitations": "Covers literature only up to 2010",
        "relevance": "Frames forecast horizons and model families; supports noting uncertainty as a limitation",
    },
]

METHOD_REFS = [
    "Box, G.E.P., Jenkins, G.M., Reinsel, G.C., Ljung, G.M. (2015). Time Series Analysis: Forecasting and Control, 5th ed. Wiley.",
    "Cleveland, R.B., Cleveland, W.S., McRae, J.E., Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. Journal of Official Statistics, 6(1), 3–73.",
    "Diebold, F.X., Mariano, R.S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.",
    "Hyndman, R.J., Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd ed. OTexts. https://otexts.com/fpp3/",
    "Seabold, S., Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. Proc. 9th Python in Science Conference.",
    "Weyrauch, A., Taubert, O., Götz, E., Streit, A., Götz, M., Debus, C., Wolter, F. (2024). Hourly Water Demand of a Mixed District Recorded by Supplier [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.11045013 (CC-BY-4.0).",
    "United Nations (2015). Transforming our world: the 2030 Agenda for Sustainable Development (A/RES/70/1). Goal 6: Clean Water and Sanitation.",
]

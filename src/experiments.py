"""Phases 5-6: model selection on validation, then a single test evaluation.

Selection rule (identical for both families): lowest validation MAE.
The test period is touched exactly once, after both configurations are frozen.

Outputs
- outputs/metrics/validation_sarima.csv, validation_mstl.csv
- outputs/forecasts/test_forecasts.csv, validation_forecasts.csv
- outputs/metrics/selected_configs.json, sarima_summary.txt
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from common import load_clean, metrics, split_index
from models import mstl_arima_rolling, sarima_rolling, seasonal_naive, fit_sarima

SARIMA_ORDERS = [((1, 0, 0), (0, 1, 1, 24)), ((2, 0, 0), (0, 1, 1, 24)), ((1, 0, 1), (0, 1, 1, 24)),
                 ((2, 0, 1), (0, 1, 1, 24)), ((3, 0, 1), (0, 1, 1, 24)), ((2, 0, 0), (1, 1, 1, 24))]
SARIMA_WEEKS = [4, 8, 12]
MSTL_WEEKS = [4, 8, 12]
MSTL_ORDERS = [((1, 1, 1), "n"), ((2, 1, 2), "n"), ((2, 0, 1), "c")]


def run_sarima_cfg(y, start, n_days, order, sorder, weeks):
    t = time.time()
    f, res, warns = sarima_rolling(y, start, n_days, order, sorder, weeks)
    return {"order": str(order), "seasonal_order": str(sorder), "est_weeks": weeks,
            "aic": float(res.aic), "converged_warnings": len(warns), "seconds": round(time.time() - t, 1)}, f


def run_mstl_cfg(y, start, n_days, weeks, order, trend, periods=(C.DAILY, C.WEEKLY)):
    t = time.time()
    f, fails = mstl_arima_rolling(y, start, n_days, weeks, order, trend, periods)
    return {"window_weeks": weeks, "arima_order": str(order), "trend": trend, "periods": str(periods),
            "arima_fallbacks": fails, "seconds": round(time.time() - t, 1)}, f


def main():
    df = load_clean()
    y = df[C.TARGET]
    train, val, test = split_index(df)
    v0, t0 = len(train), len(train) + len(val)
    yv, yt = val[C.TARGET].to_numpy(), test[C.TARGET].to_numpy()
    par = Parallel(n_jobs=12, verbose=5)
    # --reuse-validation: take the frozen selections from the saved validation
    # tables instead of re-running the grid (results are deterministic).
    if "--reuse-validation" in sys.argv:
        best_s = pd.read_csv(C.METRIC_DIR / "validation_sarima.csv").iloc[0]
        best_m = pd.read_csv(C.METRIC_DIR / "validation_mstl.csv").iloc[0]
    else:
        best_s, best_m = run_validation(par, y, v0, val, yv)
    test_stage(par, y, t0, test, yt, best_s, best_m)


def run_validation(par, y, v0, val, yv):
    # ---- Validation: SARIMA ------------------------------------------------
    jobs = [(o, s, w) for (o, s) in SARIMA_ORDERS for w in SARIMA_WEEKS]
    res = par(delayed(run_sarima_cfg)(y, v0, C.VAL_DAYS, o, s, w) for o, s, w in jobs)
    rows = [{**info, **metrics(yv, f)} for info, f in res]
    vs = pd.DataFrame(rows).sort_values("MAE")
    vs.to_csv(C.METRIC_DIR / "validation_sarima.csv", index=False)
    print(vs.to_string())
    best_s = vs.iloc[0]

    # ---- Validation: MSTL-ARIMA ---------------------------------------------
    jobs = [(w, o, tr) for w in MSTL_WEEKS for (o, tr) in MSTL_ORDERS]
    res_m = par(delayed(run_mstl_cfg)(y, v0, C.VAL_DAYS, w, o, tr) for w, o, tr in jobs)
    rows = [{**info, **metrics(yv, f)} for info, f in res_m]
    vm = pd.DataFrame(rows).sort_values("MAE")
    vm.to_csv(C.METRIC_DIR / "validation_mstl.csv", index=False)
    print(vm.to_string())
    best_m = vm.iloc[0]

    # Keep the selected configurations' validation forecasts for plots.
    sel_s = [f for (info, f) in res if info["order"] == best_s.order and
             info["seasonal_order"] == best_s.seasonal_order and info["est_weeks"] == best_s.est_weeks][0]
    sel_m = [f for (info, f) in res_m if info["window_weeks"] == best_m.window_weeks and
             info["arima_order"] == best_m.arima_order and info["trend"] == best_m.trend][0]
    pd.DataFrame({"actual": yv, "SARIMA": sel_s, "MSTL-ARIMA": sel_m,
                  "Seasonal naive (168h)": seasonal_naive(y, v0, C.VAL_DAYS, 168)},
                 index=val.index).to_csv(C.FORECAST_DIR / "validation_forecasts.csv")
    return best_s, best_m


def test_stage(par, y, t0, test, yt, best_s, best_m):
    # ---- Freeze configs, evaluate once on test ----------------------------
    s_order, s_sorder = eval(best_s.order), eval(best_s.seasonal_order)
    s_weeks, m_weeks = int(best_s.est_weeks), int(best_m.window_weeks)
    m_order, m_trend = eval(best_m.arima_order), best_m.trend
    # SARIMA parameters are re-estimated on the s_weeks just before the test
    # period (validation data is allowed pre-test history).
    s_res, s_warn = fit_sarima(y, t0, s_order, s_sorder, s_weeks)
    with open(C.METRIC_DIR / "sarima_summary.txt", "w") as fh:
        fh.write(str(s_res.summary()))

    test_jobs = [
        # The worker re-estimates the same pre-test fit (fitted results objects
        # cannot be sent between processes); estimation is deterministic.
        delayed(sarima_rolling)(y, t0, C.TEST_DAYS, s_order, s_sorder, s_weeks),
        delayed(mstl_arima_rolling)(y, t0, C.TEST_DAYS, m_weeks, m_order, m_trend, (C.DAILY, C.WEEKLY)),
        delayed(mstl_arima_rolling)(y, t0, C.TEST_DAYS, m_weeks, m_order, m_trend, (C.DAILY,)),
    ]
    (f_s, _, _), (f_m, fails_m), (f_m24, fails_m24) = par(test_jobs)
    fc = pd.DataFrame({"actual": yt, "SARIMA": f_s, "MSTL-ARIMA": f_m, "MSTL-24 only": f_m24,
                       "Seasonal naive (24h)": seasonal_naive(y, t0, C.TEST_DAYS, 24),
                       "Seasonal naive (168h)": seasonal_naive(y, t0, C.TEST_DAYS, 168),
                       "publisher_modified": test.publisher_modified.to_numpy()},
                      index=test.index)
    fc.to_csv(C.FORECAST_DIR / "test_forecasts.csv")

    selected = {
        "selection_rule": "lowest validation MAE over the listed grid; test used once",
        "protocol": f"rolling day-ahead: origin 00:00 daily, horizon {C.HORIZON} h, "
                    f"{C.TEST_DAYS} test days ({C.TEST_DAYS * 24} forecasts)",
        "SARIMA": {"order": s_order, "seasonal_order": s_sorder, "est_weeks": s_weeks,
                   "test_estimation_window": [str(y.index[t0 - s_weeks * 168]), str(y.index[t0 - 1])],
                   "test_fit_aic": float(s_res.aic), "test_fit_convergence_warnings": s_warn,
                   "params": {k: float(v) for k, v in s_res.params.items()},
                   "validation_MAE": float(best_s.MAE)},
        "MSTL-ARIMA": {"periods": [C.DAILY, C.WEEKLY], "window_weeks": m_weeks, "arima_order": m_order,
                       "trend": m_trend, "seasonal_extension": "repeat last cycle of each component",
                       "validation_MAE": float(best_m.MAE), "test_arima_fallbacks": int(fails_m)},
        "MSTL-24 only (ablation)": {"periods": [C.DAILY], "window_weeks": m_weeks, "arima_order": m_order,
                                    "trend": m_trend, "test_arima_fallbacks": int(fails_m24)},
    }
    with open(C.METRIC_DIR / "selected_configs.json", "w") as fh:
        json.dump(selected, fh, indent=2, default=str)
    print(json.dumps(selected, indent=1, default=str))
    for col in ["SARIMA", "MSTL-ARIMA", "MSTL-24 only", "Seasonal naive (24h)", "Seasonal naive (168h)"]:
        print(col, {k: round(v, 2) for k, v in metrics(fc.actual, fc[col]).items()})


if __name__ == "__main__":
    main()

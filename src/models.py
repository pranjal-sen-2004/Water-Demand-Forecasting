"""Forecasting models, each run with the same rolling day-ahead protocol.

At every origin (00:00 of a forecast day) a model may use observations
strictly before the origin and must output the next HORIZON (24) hours.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config as C

H = C.HORIZON


def origins(start, n_days):
    """Integer positions of forecast origins (one per day)."""
    return [start + H * k for k in range(n_days)]


# --------------------------------------------------------------------------
# Reference benchmarks
# --------------------------------------------------------------------------
def seasonal_naive(y, start, n_days, period):
    """Forecast each hour with the value one period earlier."""
    v = y.to_numpy()
    out = []
    for o in origins(start, n_days):
        out.append(np.array([v[o + h - period * (1 + h // period)] for h in range(H)]))
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Baseline: SARIMA(p,d,q)(P,D,Q)_24
# --------------------------------------------------------------------------
def fit_sarima(y, end, order, seasonal_order, est_weeks):
    """Estimate parameters on the est_weeks immediately before position `end`."""
    hist = y.iloc[end - est_weeks * 168:end]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = SARIMAX(hist, order=order, seasonal_order=seasonal_order).fit(disp=False, maxiter=200)
    conv = [str(x.message) for x in w if issubclass(x.category, ConvergenceWarning)]
    return res, conv


def sarima_rolling(y, start, n_days, order, seasonal_order, est_weeks, res=None):
    """Parameters are fixed after estimation; each day the Kalman filter state
    is updated with that day's 24 observations (res.extend) before the next
    forecast, so no information from after an origin is used."""
    warns = []
    if res is None:
        res, warns = fit_sarima(y, start, order, seasonal_order, est_weeks)
    fitted = res
    out = []
    for o in origins(start, n_days):
        out.append(np.asarray(res.forecast(H)))
        res = res.extend(y.iloc[o:o + H])
    return np.concatenate(out), fitted, warns


# --------------------------------------------------------------------------
# Enhancement: MSTL decomposition + ARIMA on the seasonally adjusted series
# --------------------------------------------------------------------------
def mstl_arima_one(hist, periods, order, trend):
    """One forecast from one origin.

    1. MSTL splits the trailing window into trend + one seasonal component per
       period + remainder.
    2. Each seasonal component is extended by repeating its most recent cycle
       (seasonal naive on the component).
    3. ARIMA forecasts the seasonally adjusted series (trend + remainder).
    4. Forecast = ARIMA forecast + sum of extended seasonal components.
    """
    res = MSTL(hist, periods=periods, stl_kwargs={"seasonal_deg": 0}).fit()
    seas = res.seasonal if res.seasonal.ndim == 2 else res.seasonal.to_frame()
    s_fc = np.zeros(H)
    for i, p in enumerate(periods):
        comp = seas.iloc[:, i].to_numpy()
        s_fc += np.array([comp[len(comp) - p + (h % p)] for h in range(H)])
    adj = hist.to_numpy() - seas.sum(axis=1).to_numpy()
    failed = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ARIMA(adj, order=order, trend=trend).fit()
        a_fc = np.asarray(a.forecast(H))
        if not np.all(np.isfinite(a_fc)):
            raise ValueError("non-finite forecast")
    except Exception:
        # Fallback keeps the run going; the count is reported.
        a_fc, failed = np.repeat(adj[-1], H), True
    return a_fc + s_fc, failed


def mstl_arima_rolling(y, start, n_days, window_weeks, order, trend="n", periods=(C.DAILY, C.WEEKLY)):
    out, fails = [], 0
    w = window_weeks * 168
    for o in origins(start, n_days):
        f, failed = mstl_arima_one(y.iloc[o - w:o], periods, order, trend)
        out.append(f)
        fails += failed
    return np.concatenate(out), fails

"""Phase 7: test-set evaluation, uncertainty analysis and diagnostics.

Reads the frozen test forecasts written by experiments.py; never refits on
test data. The SARIMA re-estimation below repeats the pre-test fit (same
window, same order) only to obtain in-sample residuals for diagnostics.
"""
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

import config as C
from common import COLORS, INK2, load_clean, metrics, savefig, split_index
from models import fit_sarima

MODELS = ["SARIMA", "MSTL-ARIMA", "MSTL-24 only", "Seasonal naive (24h)", "Seasonal naive (168h)"]
RNG = np.random.default_rng(C.SEED)


def dm_test(loss_a, loss_b, lag=7):
    """Diebold-Mariano test on daily mean losses (one value per forecast day,
    so each day's 24-hour forecast is one observation). Newey-West long-run
    variance with `lag` days captures weekly dependence between days.
    H0: equal expected loss. Positive statistic: model b has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T, dbar = len(d), d.mean()
    dc = d - dbar
    lrv = np.sum(dc * dc) / T
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / T
    stat = dbar / np.sqrt(lrv / T)
    p = 2 * stats.t.sf(abs(stat), df=T - 1)  # sf avoids 1 - cdf underflow
    return {"mean_loss_diff": float(dbar), "DM_stat": float(stat), "p_value": float(p), "days": int(T),
            "nw_lag_days": lag}


def block_bootstrap_reduction(err_a, err_b, block=7, n_boot=2000):
    """95% CI of the % MAE and RMSE reduction (a -> b) via moving-block bootstrap of whole days."""
    ea, eb = err_a.reshape(-1, 24), err_b.reshape(-1, 24)
    n = len(ea)
    starts_max = n - block + 1
    mae_red, rmse_red = [], []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in RNG.integers(0, starts_max, n // block + 1)])[:n]
        a, b = ea[idx].ravel(), eb[idx].ravel()
        mae_red.append(100 * (np.abs(a).mean() - np.abs(b).mean()) / np.abs(a).mean())
        rmse_red.append(100 * (np.sqrt((a ** 2).mean()) - np.sqrt((b ** 2).mean())) / np.sqrt((a ** 2).mean()))
    return {"MAE_reduction_pct_CI95": [float(np.percentile(mae_red, 2.5)), float(np.percentile(mae_red, 97.5))],
            "RMSE_reduction_pct_CI95": [float(np.percentile(rmse_red, 2.5)), float(np.percentile(rmse_red, 97.5))],
            "block_days": block, "resamples": n_boot}


def main():
    fc = pd.read_csv(C.FORECAST_DIR / "test_forecasts.csv", parse_dates=["timestamp"], index_col="timestamp")
    sel = json.load(open(C.METRIC_DIR / "selected_configs.json"))
    y = fc["actual"]
    out = {"test_period": [str(fc.index[0]), str(fc.index[-1])], "n_forecasts": len(fc),
           "protocol": sel["protocol"],
           "metric_definitions": {
               "MAE": "mean(|y - f|), m^3/h", "RMSE": "sqrt(mean((y - f)^2)), m^3/h",
               "MAPE": "100 * mean(|(y - f) / y|), %", "WAPE": "100 * sum|y - f| / sum|y|, %"},
           "min_actual_in_test": float(y.min()),
           "hours_actual_below_100": int((y < 100).sum())}

    # ---- Metric tables ------------------------------------------------------
    keep = ~fc.publisher_modified.astype(bool)
    out["metrics_all_hours"] = {m: metrics(y, fc[m]) for m in MODELS}
    out["metrics_observed_hours_only"] = {m: metrics(y[keep], fc[m][keep]) for m in MODELS}
    out["n_observed_hours"] = int(keep.sum())
    b, e = out["metrics_all_hours"]["SARIMA"], out["metrics_all_hours"]["MSTL-ARIMA"]
    out["comparison"] = {k: {"SARIMA": b[k], "MSTL-ARIMA": e[k], "difference": e[k] - b[k],
                             "pct_reduction": 100 * (b[k] - e[k]) / b[k]} for k in b}
    a24 = out["metrics_all_hours"]["MSTL-24 only"]
    out["ablation_weekly_component"] = {k: {"MSTL-24 only": a24[k], "MSTL-24+168": e[k],
                                            "pct_reduction": 100 * (a24[k] - e[k]) / a24[k]} for k in b}

    # ---- Uncertainty: DM tests + block bootstrap ----------------------------
    err = {m: (y - fc[m]).to_numpy() for m in MODELS}
    day_abs = {m: np.abs(err[m]).reshape(-1, 24).mean(1) for m in MODELS}
    day_sq = {m: (err[m] ** 2).reshape(-1, 24).mean(1) for m in MODELS}
    out["dm_tests"] = {
        "SARIMA_vs_MSTL-ARIMA_abs": dm_test(day_abs["SARIMA"], day_abs["MSTL-ARIMA"]),
        "SARIMA_vs_MSTL-ARIMA_sq": dm_test(day_sq["SARIMA"], day_sq["MSTL-ARIMA"]),
        "MSTL-24only_vs_MSTL-ARIMA_abs": dm_test(day_abs["MSTL-24 only"], day_abs["MSTL-ARIMA"]),
        "SNaive168_vs_MSTL-ARIMA_abs": dm_test(day_abs["Seasonal naive (168h)"], day_abs["MSTL-ARIMA"]),
        "SNaive168_vs_SARIMA_abs": dm_test(day_abs["Seasonal naive (168h)"], day_abs["SARIMA"]),
    }
    out["bootstrap_SARIMA_to_MSTL"] = block_bootstrap_reduction(err["SARIMA"], err["MSTL-ARIMA"])
    out["days_MSTL_better_MAE"] = int((day_abs["MSTL-ARIMA"] < day_abs["SARIMA"]).sum())

    # ---- Breakdowns ---------------------------------------------------------
    ab = pd.DataFrame({m: np.abs(err[m]) for m in MODELS}, index=fc.index)
    by_hour = ab.groupby(ab.index.hour).mean()
    by_dow = ab.groupby(ab.index.dayofweek).mean()
    by_month = ab.groupby(ab.index.to_period("M")).mean()
    out["MAE_by_hour"] = by_hour[["SARIMA", "MSTL-ARIMA"]].round(1).to_dict()
    out["MAE_by_dow"] = by_dow[["SARIMA", "MSTL-ARIMA"]].round(1).to_dict()
    out["MAE_by_month"] = {str(k): v for k, v in by_month[["SARIMA", "MSTL-ARIMA"]].round(1).T.to_dict().items()}
    out["MAE_weekday_vs_weekend"] = {
        m: {"weekday": float(ab[m][ab.index.dayofweek < 5].mean()),
            "weekend": float(ab[m][ab.index.dayofweek >= 5].mean())} for m in ["SARIMA", "MSTL-ARIMA", "MSTL-24 only"]}
    by_hour.to_csv(C.METRIC_DIR / "test_mae_by_hour.csv")
    by_dow.to_csv(C.METRIC_DIR / "test_mae_by_dow.csv")
    largest = ab["MSTL-ARIMA"].nlargest(5)
    out["largest_errors_MSTL"] = {str(k): {"actual": float(y[k]), "MSTL": float(fc["MSTL-ARIMA"][k]),
                                           "SARIMA": float(fc["SARIMA"][k])} for k in largest.index}

    # ---- Residual diagnostics ----------------------------------------------
    df = load_clean()
    tr, va, te = split_index(df)
    s = sel["SARIMA"]
    res, _ = fit_sarima(df[C.TARGET], len(tr) + len(va), tuple(s["order"]), tuple(s["seasonal_order"]),
                        s["est_weeks"])
    resid = res.resid.iloc[24 + 1:]  # drop the burn-in of the seasonal difference
    lb = acorr_ljungbox(resid, lags=[24, 168])
    out["sarima_insample_ljungbox"] = {int(k): {"stat": float(r.lb_stat), "p": float(r.lb_pvalue)}
                                       for k, r in lb.iterrows()}
    for m in ["SARIMA", "MSTL-ARIMA"]:
        lbe = acorr_ljungbox(err[m], lags=[24, 168])
        out[f"forecast_error_ljungbox_{m}"] = {int(k): {"stat": float(r.lb_stat), "p": float(r.lb_pvalue)}
                                               for k, r in lbe.iterrows()}
        out[f"forecast_error_mean_{m}"] = float(err[m].mean())
    out["error_acf"] = {m: {str(k): float(v) for k, v in zip((1, 24, 144, 168),
                                                             acf(err[m], nlags=168)[[1, 24, 144, 168]])}
                        for m in ["SARIMA", "MSTL-ARIMA"]}

    fig = res.plot_diagnostics(figsize=(11, 7), lags=48)
    fig.suptitle(f"Figure 12. SARIMA in-sample residual diagnostics ({s['est_weeks']}-week pre-test estimation window)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    savefig(fig, "fig12_sarima_residual_diagnostics.png")

    # ---- Figures --------------------------------------------------------------
    win = slice("2020-09-07", "2020-09-20")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(y[win].index, y[win], color=COLORS["actual"], lw=1.8, label="Actual")
    for m in ["SARIMA", "MSTL-ARIMA"]:
        ax.plot(fc[win].index, fc[m][win], color=COLORS[m], lw=1.3, label=m)
    for t in pd.date_range("2020-09-07", "2020-09-20", freq="D"):
        if t.dayofweek >= 5:
            ax.axvspan(t, t + pd.Timedelta(hours=24), color="#e4e3df", alpha=0.6, lw=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.set(title="Figure 9. Day-ahead forecasts vs actual, 7–20 Sep 2020 (test period; weekends shaded)",
           ylabel="Consumption (m³/h)")
    ax.legend(loc="upper left", ncol=3)
    savefig(fig, "fig09_test_forecasts_two_weeks.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for m in ["SARIMA", "MSTL-ARIMA", "MSTL-24 only"]:
        axes[0].plot(by_hour.index, by_hour[m], marker="o", ms=4, color=COLORS[m], label=m)
        axes[1].plot(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], by_dow[m], marker="o", ms=7,
                     color=COLORS[m], label=m)
    axes[0].set(title="(a) MAE by hour of day (= lead time, origin 00:00)", xlabel="Hour", ylabel="MAE (m³/h)",
                xticks=range(0, 24, 3), ylim=(0, None))
    axes[1].set(title="(b) MAE by day of week", ylabel="MAE (m³/h)", ylim=(0, None))
    axes[0].legend(loc="upper left")
    fig.suptitle("Figure 10. Where the test errors occur", fontweight="bold", y=1.03)
    savefig(fig, "fig10_error_breakdown.png")

    daily = pd.DataFrame({m: day_abs[m] for m in ["SARIMA", "MSTL-ARIMA"]},
                         index=pd.date_range(fc.index[0], periods=len(day_abs["SARIMA"]), freq="D"))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for m in daily:
        ax.plot(daily.index, daily[m].rolling(7, center=True).mean(), color=COLORS[m], label=f"{m} (7-day mean)")
    ax.set(title="Figure 11. Daily MAE across the test period", ylabel="MAE (m³/h)", ylim=(0, None))
    ax.legend(loc="upper right")
    savefig(fig, "fig11_daily_mae_test.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for i, m in enumerate(["SARIMA", "MSTL-ARIMA"]):
        plot_acf(err[m], lags=200, ax=axes[i, 0], markersize=2, title=f"({'ab'[i]}) ACF of {m} test errors")
        for k in (24, 168):
            axes[i, 0].axvline(k, color=INK2, lw=0.8, ls=":")
        axes[i, 1].hist(err[m], bins=80, color=COLORS[m], edgecolor="white", linewidth=0.3)
        axes[i, 1].set(title=f"({'cd'[i]}) {m} error distribution (mean {err[m].mean():.0f} m³/h)",
                       xlabel="Actual − forecast (m³/h)", ylabel="Hours")
        axes[i, 0].set_xlabel("Lag (hours)")
    fig.suptitle("Figure 13. Test forecast-error diagnostics", fontweight="bold", y=1.0)
    fig.tight_layout()
    savefig(fig, "fig13_forecast_error_diagnostics.png")

    # Comparison bar chart of headline metrics (bars start at zero)
    labels = ["Seasonal naive (24h)", "Seasonal naive (168h)", "SARIMA", "MSTL-24 only", "MSTL-ARIMA"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"wspace": 0.75})
    for ax, k in zip(axes, ["MAE", "RMSE"]):
        vals = [out["metrics_all_hours"][m][k] for m in labels]
        cols = [COLORS.get(m, "#b9b8b2") for m in labels]
        bars = ax.barh(labels, vals, color=cols, height=0.6)
        for bb, v in zip(bars, vals):
            ax.text(v, bb.get_y() + bb.get_height() / 2, f" {v:.0f}", va="center", fontsize=9)
        ax.set(title=f"{k} on test set (m³/h)", xlim=(0, max(vals) * 1.15))
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 14. Test-set accuracy of all models (lower is better)", fontweight="bold", y=1.03)
    savefig(fig, "fig14_metric_comparison.png")

    with open(C.METRIC_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: out[k] for k in ["metrics_all_hours", "comparison", "ablation_weekly_component",
                                          "dm_tests", "bootstrap_SARIMA_to_MSTL", "days_MSTL_better_MAE",
                                          "MAE_weekday_vs_weekend", "sarima_insample_ljungbox",
                                          "metrics_observed_hours_only", "min_actual_in_test"]},
                     indent=1, default=str))


if __name__ == "__main__":
    main()

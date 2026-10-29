"""Phase 4: exploratory data analysis.

Descriptive plots use the full series. Every statistic that informs model
design (ADF/KPSS, ACF/PACF, MSTL seasonal strength) uses the training split
only, so nothing from validation/test leaks into model choices.
"""
import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller, kpss

import config as C
from common import COLORS, INK2, load_clean, savefig, split_index

warnings.filterwarnings("ignore", category=FutureWarning)


def adf_kpss(x, label):
    x = pd.Series(x).dropna()
    a = adfuller(x, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when p is outside its table
        k = kpss(x, regression="c", nlags="auto")
    return {"series": label, "n": int(len(x)),
            "adf_stat": round(a[0], 3), "adf_p": float(a[1]), "adf_lags": int(a[2]),
            "adf_nobs": int(a[3]), "adf_crit_5pct": round(a[4]["5%"], 3),
            "kpss_stat": round(k[0], 3), "kpss_p": float(k[1]), "kpss_lags": int(k[2])}


def seasonal_strength(seasonal, remainder):
    """Hyndman & Athanasopoulos: F_s = max(0, 1 - Var(R) / Var(S + R))."""
    return float(max(0.0, 1 - np.var(remainder) / np.var(seasonal + remainder)))


def main():
    df = load_clean()
    y = df[C.TARGET]
    train, val, test = split_index(df)
    out = {"splits": {k: {"start": str(d.index[0]), "end": str(d.index[-1]), "hours": len(d),
                          "publisher_modified_hours": int(d.publisher_modified.sum())}
                      for k, d in (("train", train), ("validation", val), ("test", test))}}

    # ---- A. Descriptive statistics ----------------------------------------
    d = y.describe()
    out["descriptive"] = {k: round(float(v), 2) for k, v in d.items()}
    out["descriptive"].update({"cv_pct": round(100 * y.std() / y.mean(), 2),
                               "iqr": round(float(d["75%"] - d["25%"]), 2),
                               "skew": round(float(y.skew()), 3),
                               "missing_after_publisher_fill": int(y.isna().sum())})
    out["descriptive_by_split"] = {k: {"mean": round(float(s[C.TARGET].mean()), 1),
                                       "std": round(float(s[C.TARGET].std()), 1),
                                       "min": float(s[C.TARGET].min()), "max": float(s[C.TARGET].max()),
                                       "hours_below_100": int((s[C.TARGET] < 100).sum())}
                                   for k, s in (("train", train), ("validation", val), ("test", test))}

    # ---- B. Full time-series plot with splits and imputed block ----------
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(y.index, y, color="#b9b8b2", lw=0.3, label="Hourly")
    ax.plot(y.resample("D").mean(), color=COLORS["actual"], lw=1.0, label="Daily mean")
    for s, lab in ((val, "Validation"), (test, "Test")):
        ax.axvline(s.index[0], color=INK2, ls="--", lw=1)
        ax.text(s.index[0], y.max() * 0.98, f" {lab} →", color=INK2, va="top", fontsize=9)
    imp = df.index[df.imputed_missing]
    gap = imp[(imp >= "2017-06-01") & (imp <= "2017-07-01")]
    ax.axvspan(gap.min(), gap.max(), color="#eb6834", alpha=0.25, label="Publisher-filled gap (502 h)")
    ax.set(title="Figure 1. Hourly water consumption, Jan 2016 – Apr 2021",
           ylabel="Consumption (m³/h)", xlabel="Date")
    ax.legend(loc="upper left", ncol=3)
    savefig(fig, "fig01_full_series.png")

    # Two-week zoom showing daily and weekly cycles
    z = y["2019-05-06":"2019-05-19"]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(z.index, z, color=COLORS["actual"])
    for t in pd.date_range(z.index[0], z.index[-1], freq="D"):
        if t.dayofweek >= 5:
            ax.axvspan(t, t + pd.Timedelta(hours=24), color="#e4e3df", alpha=0.6, lw=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.set(title="Figure 2. Two weeks of hourly demand (6–19 May 2019, weekends shaded)",
           ylabel="Consumption (m³/h)")
    savefig(fig, "fig02_two_week_zoom.png")

    # ---- C. Trend and seasonality profiles (training only) ----------------
    tr = train
    by_hour = tr.groupby("hour")[C.TARGET].mean()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = tr.groupby("dayofweek")[C.TARGET].mean()
    heat = tr.pivot_table(index="dayofweek", columns="hour", values=C.TARGET, aggfunc="mean")
    out["profiles"] = {"hour_mean": by_hour.round(1).to_dict(),
                       "dow_mean": {dow_names[i]: round(v, 1) for i, v in by_dow.items()},
                       "peak_hour": int(by_hour.idxmax()), "trough_hour": int(by_hour.idxmin()),
                       "weekday_mean": round(float(tr[tr.weekend == 0][C.TARGET].mean()), 1),
                       "weekend_mean": round(float(tr[tr.weekend == 1][C.TARGET].mean()), 1)}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 0.8, 1.4]})
    axes[0].plot(by_hour.index, by_hour, marker="o", ms=4, color=COLORS["SARIMA"])
    axes[0].set(title="(a) Mean by hour of day", xlabel="Hour", ylabel="m³/h", xticks=range(0, 24, 3))
    # Dots, not bars: the differences are small relative to the level, and a
    # bar axis that does not start at zero would exaggerate them.
    axes[1].plot(dow_names, by_dow, marker="o", ms=8, lw=0, color=COLORS["SARIMA"])
    axes[1].axhline(tr[C.TARGET].mean(), color=INK2, ls=":", lw=1, label="Overall mean")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set(title="(b) Mean by day of week", ylabel="m³/h")
    im = axes[2].imshow(heat, aspect="auto", cmap="Blues")
    axes[2].set(title="(c) Hour × weekday mean", xlabel="Hour", yticks=range(7), yticklabels=dow_names)
    axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], label="m³/h")
    fig.suptitle("Figure 3. Daily and weekly demand profiles (training period)", fontweight="bold", y=1.03)
    savefig(fig, "fig03_seasonal_profiles.png")

    monthly = y.resample("MS").mean()
    yearly = y.resample("YS").mean()
    out["monthly_mean"] = {str(k.date()): round(v, 1) for k, v in monthly.items()}
    out["annual_mean"] = {str(k.year): round(v, 1) for k, v in yearly.items()}
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(monthly.index, monthly, marker="o", ms=4, color=COLORS["actual"])
    ax.set(title="Figure 4. Monthly mean consumption (long-term level and annual cycle)",
           ylabel="Mean consumption (m³/h)")
    savefig(fig, "fig04_monthly_mean.png")

    # ---- G. Distribution and outliers -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].hist(y, bins=80, color=COLORS["SARIMA"], edgecolor="white", linewidth=0.3)
    axes[0].set(title="(a) Distribution of hourly consumption", xlabel="m³/h", ylabel="Hours")
    data = [tr[tr.hour == h][C.TARGET].values for h in range(24)]
    axes[1].boxplot(data, positions=range(24), widths=0.6, flierprops={"ms": 2, "alpha": 0.4},
                    medianprops={"color": COLORS["MSTL-ARIMA"]})
    axes[1].set(title="(b) By hour of day (training)", xlabel="Hour", ylabel="m³/h", xticks=range(0, 24, 3))
    fig.suptitle("Figure 5. Distribution and outlier screening", fontweight="bold", y=1.03)
    savefig(fig, "fig05_distribution.png")

    # Outlier screen: robust z-score within each hour-of-day (training)
    g = tr.groupby("hour")[C.TARGET]
    med, mad = g.transform("median"), g.transform(lambda s: (s - s.median()).abs().median())
    rz = 0.6745 * (tr[C.TARGET] - med) / mad
    out["outliers_train"] = {"robust_z_gt_5": int((rz.abs() > 5).sum()),
                             "examples": {str(k): round(float(v), 1)
                                          for k, v in tr[C.TARGET][rz.abs() > 5].head(10).items()}}

    # ---- D. MSTL decomposition (training only) ----------------------------
    mstl = MSTL(train[C.TARGET], periods=(C.DAILY, C.WEEKLY), stl_kwargs={"seasonal_deg": 0}).fit()
    s24, s168 = mstl.seasonal.iloc[:, 0], mstl.seasonal.iloc[:, 1]
    out["mstl_train"] = {
        "strength_daily_24": round(seasonal_strength(s24, mstl.resid), 3),
        "strength_weekly_168": round(seasonal_strength(s168, mstl.resid), 3),
        "trend_strength": round(float(max(0, 1 - np.var(mstl.resid) / np.var(mstl.trend + mstl.resid))), 3),
        "amplitude_daily_p5_p95": [round(float(np.percentile(s24, q)), 1) for q in (5, 95)],
        "amplitude_weekly_p5_p95": [round(float(np.percentile(s168, q)), 1) for q in (5, 95)],
        "resid_std": round(float(mstl.resid.std()), 1),
    }
    win = slice("2019-06-03", "2019-06-30")
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    parts = [(train[C.TARGET][win], "Observed"), (mstl.trend[win], "Trend"),
             (s24[win], "Daily seasonal (24 h)"), (s168[win], "Weekly seasonal (168 h)"),
             (mstl.resid[win], "Remainder")]
    for ax, (s, lab) in zip(axes, parts):
        ax.plot(s.index, s, color=COLORS["actual"] if lab == "Observed" else COLORS["SARIMA"], lw=1)
        ax.set_ylabel(lab, fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle("Figure 6. MSTL decomposition of training data (periods 24 h and 168 h); June 2019 shown",
                 fontweight="bold", y=0.995)
    savefig(fig, "fig06_mstl_decomposition.png")

    # ---- E. Stationarity (training only) ----------------------------------
    yt = train[C.TARGET]
    out["stationarity"] = [adf_kpss(yt, "level"),
                           adf_kpss(yt.diff(24), "seasonal difference (lag 24)"),
                           adf_kpss(yt.diff(24).diff(), "lag-24 then lag-1 difference")]

    # ---- F. ACF / PACF (training only) -------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    plot_acf(yt, lags=24 * 8, ax=axes[0, 0], markersize=2, title="(a) ACF – level")
    plot_pacf(yt, lags=72, ax=axes[0, 1], method="ywm", markersize=3, title="(b) PACF – level")
    yd = yt.diff(24).dropna()
    plot_acf(yd, lags=24 * 8, ax=axes[1, 0], markersize=2, title="(c) ACF – after lag-24 difference")
    plot_pacf(yd, lags=72, ax=axes[1, 1], method="ywm", markersize=3, title="(d) PACF – after lag-24 difference")
    for ax in axes[:, 0]:
        for k in (24, 168):
            ax.axvline(k, color=COLORS["MSTL-ARIMA"], lw=0.8, ls=":")
    for ax in axes.flat:
        ax.set_xlabel("Lag (hours)")
    fig.suptitle("Figure 7. Autocorrelation structure of the training series (dotted: lags 24 and 168)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    savefig(fig, "fig07_acf_pacf.png")
    from statsmodels.tsa.stattools import acf
    a_lvl, a_d = acf(yt, nlags=336), acf(yd, nlags=336)
    out["acf_values"] = {"level": {k: round(float(a_lvl[k]), 3) for k in (1, 12, 24, 48, 144, 167, 168, 336)},
                         "diff24": {k: round(float(a_d[k]), 3) for k in (1, 2, 24, 48, 168, 336)}}

    # Exogenous check: daily consumption vs temperature (training only)
    dd = train.resample("D").agg({C.TARGET: "sum", "temperature": "mean"})
    out["temperature_corr_daily_train"] = round(float(dd.corr().iloc[0, 1]), 3)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(dd.temperature, dd[C.TARGET] / 1000, s=8, alpha=0.5, color=COLORS["SARIMA"])
    ax.set(title=f"Figure 8. Daily demand vs temperature (r = {out['temperature_corr_daily_train']})",
           xlabel="Daily mean temperature (°C)", ylabel="Daily consumption (1000 m³)")
    savefig(fig, "fig08_temperature.png")

    with open(C.METRIC_DIR / "eda_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

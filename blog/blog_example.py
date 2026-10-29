"""Standalone worked example for the blog post.

Usage:  python blog_example.py path/to/waterconsumption_noNaN_fixedTimechange.csv [output_dir]
Produces blog_fig1_decomposition.png and blog_fig2_one_day_forecast.png.
"""
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
path = sys.argv[1] if len(sys.argv) > 1 else "waterconsumption_noNaN_fixedTimechange.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "."

# 1. Load: build an hourly timestamp from Date + the start of the Time interval
df = pd.read_csv(path)
start_hour = df["Time"].str.strip().str[:2].astype(int)
df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y") + pd.to_timedelta(start_hour, unit="h")
y = df["Consumption"].asfreq("h")          # m^3 per hour, regular hourly grid

# 2. Decompose 12 weeks of history into trend + daily + weekly + remainder
origin = pd.Timestamp("2020-09-14 00:00")   # a Monday in the test period
hist = y[origin - pd.Timedelta(weeks=12):origin - pd.Timedelta(hours=1)]
res = MSTL(hist, periods=(24, 168), stl_kwargs={"seasonal_deg": 0}).fit()

fig = res.plot()
fig.set_size_inches(9, 7)
fig.suptitle("MSTL decomposition: 12 weeks of hourly water demand", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(out, "blog_fig1_decomposition.png"), dpi=150)

# 3. Enhanced forecast for the next 24 h
s24, s168 = res.seasonal["seasonal_24"].to_numpy(), res.seasonal["seasonal_168"].to_numpy()
season_next = s24[-24:] + s168[-168:-144]   # repeat the latest daily and weekly cycle
adjusted = hist - res.seasonal.sum(axis=1)  # trend + remainder
arima = ARIMA(adjusted.to_numpy(), order=(1, 1, 1)).fit()
mstl_fc = arima.forecast(24) + season_next

# 4. Baseline: SARIMA(3,0,1)(0,1,1)_24 estimated on the last 4 weeks
sarima = SARIMAX(hist[-4 * 168:], order=(3, 0, 1), seasonal_order=(0, 1, 1, 24)).fit(disp=False)
sarima_fc = sarima.forecast(24).to_numpy()

actual = y[origin:origin + pd.Timedelta(hours=23)].to_numpy()
for name, f in [("SARIMA", sarima_fc), ("MSTL + ARIMA", mstl_fc)]:
    print(f"{name:13s} MAE for this one day = {np.mean(np.abs(actual - f)):.0f} m^3/h")

hours = np.arange(24)
plt.figure(figsize=(9, 3.6))
plt.plot(hours, actual, color="#3a3936", lw=2, label="Actual")
plt.plot(hours, sarima_fc, color="#2a78d6", label="SARIMA(3,0,1)(0,1,1)24")
plt.plot(hours, mstl_fc, color="#eb6834", label="MSTL + ARIMA (24 h & 168 h)")
plt.xlabel("Hour of day, Monday 14 Sep 2020"); plt.ylabel("Consumption (m³/h)")
plt.title("One day-ahead forecast"); plt.legend(frameon=False); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out, "blog_fig2_one_day_forecast.png"), dpi=150)

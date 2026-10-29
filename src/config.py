"""Central configuration: paths, split sizes and model settings.

Every script imports from here so a single change propagates everywhere.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("WDF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Deep Windows folders can exceed the 260-character MAX_PATH limit; the
# extended-length prefix lifts it.
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
METRIC_DIR = ROOT / "outputs" / "metrics"
FORECAST_DIR = ROOT / "outputs" / "forecasts"

RAW_FILE = DATA_RAW / "waterconsumption_rawdata.csv"
NONAN_FILE = DATA_RAW / "waterconsumption_noNaN.csv"
FIXED_FILE = DATA_RAW / "waterconsumption_noNaN_fixedTimechange.csv"
CLEAN_FILE = DATA_PROCESSED / "water_hourly_clean.csv"

TARGET = "consumption"
FREQ = "h"

# Chronological split in whole days (1918 days in total): ~70 / 15 / 15 %.
TRAIN_DAYS = 1342
VAL_DAYS = 288
TEST_DAYS = 288

# Forecasting protocol: every day at 00:00 forecast the next 24 hours
# (day-ahead), using only observations up to 23:00 of the previous day.
HORIZON = 24

# Seasonal periods examined (hours): daily and weekly cycles.
DAILY, WEEKLY = 24, 168

# SARIMA parameters are estimated on the most recent block of history before
# the forecast period (full 30k-hour estimation with s=24 is impractical on a
# laptop). The Kalman-filter state is then updated with every new observation.
SARIMA_EST_WEEKS = 8

SEED = 42

for _d in (DATA_PROCESSED, FIG_DIR, METRIC_DIR, FORECAST_DIR):
    _d.mkdir(parents=True, exist_ok=True)

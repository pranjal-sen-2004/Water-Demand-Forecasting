"""Shared helpers: data loading, splits, metrics and figure styling."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

# Fixed categorical order (validated palette): actual data is neutral ink,
# models take series slots 1, 2, 3 in a fixed order and keep them everywhere.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"actual": "#3a3936", "SARIMA": "#2a78d6", "MSTL-ARIMA": "#eb6834",
          "MSTL-24 only": "#1baf7a", "Seasonal naive (168h)": "#6250d6"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "legend.frameon": False,
})


def load_clean():
    df = pd.read_csv(C.CLEAN_FILE, parse_dates=["timestamp"], index_col="timestamp")
    df = df.asfreq(C.FREQ)
    assert df.index.is_monotonic_increasing and df[C.TARGET].notna().all()
    return df


def split_index(df):
    """Return (train, val, test) DataFrames split on whole days."""
    h = 24
    a, b = C.TRAIN_DAYS * h, (C.TRAIN_DAYS + C.VAL_DAYS) * h
    assert b + C.TEST_DAYS * h == len(df), "split does not cover the data"
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def metrics(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    e = y - f
    return {"MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE": float(np.mean(np.abs(e / y)) * 100),
            # WAPE (sum|e| / sum y) stays stable when single hours are small.
            "WAPE": float(np.sum(np.abs(e)) / np.sum(np.abs(y)) * 100)}


def savefig(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path

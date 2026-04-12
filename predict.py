"""
predict.py — Inference-only script for GRU-based stock price forecasting.
Designed for use in a FastAPI backend (Render / Vercel deployment).

Expected files alongside this script (or paths passed via env vars):
    - gru_v4.keras
    - feature_scaler.joblib
    - target_scaler.joblib

Usage:
    from predict import run_forecast
    result = run_forecast(n_days=5)
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.impute import SimpleImputer


# ── Paths (override via environment variables if needed) ──────────────────────
MODEL_PATH          = os.getenv("MODEL_PATH",          "model_files/gru_v4.keras")
FEATURE_SCALER_PATH = os.getenv("FEATURE_SCALER_PATH", "model_files/feature_scaler.joblib")
TARGET_SCALER_PATH  = os.getenv("TARGET_SCALER_PATH",  "model_files/target_scaler.joblib")

#Use URL or direct path to the CSV data (the URL is used here to ensure the same data as training, but can be overridden if needed)
DATA_URL = (
    "https://raw.githubusercontent.com/SusmitSekharBhakta/"
    "Stock-market-price-prediction/main/final_data_adj.csv"
)

WINDOW    = 40
SPLIT_IDX_RATIO = 0.85  # kept to reproduce the exact scaler-fit split


# ── Custom loss (required to load the .keras model) ───────────────────────────
def huber_directional_loss(delta: float = 0.05, direction_weight: float = 0.25):
    def loss(y_true, y_pred):
        err     = y_true - y_pred
        abs_err = tf.abs(err)
        huber   = tf.where(
            abs_err <= delta,
            0.5 * tf.square(err),
            delta * (abs_err - 0.5 * delta),
        )
        dir_wrong = tf.maximum(0.0, -tf.sign(y_true) * tf.sign(y_pred))
        return tf.reduce_mean(huber) + direction_weight * tf.reduce_mean(dir_wrong)

    loss.__name__ = "huber_directional"
    return loss


# ── Module-level singletons (loaded once per worker process) ──────────────────
_model          = None
_feature_scaler = None
_target_scaler  = None


def _load_artifacts():
    global _model, _feature_scaler, _target_scaler
    if _model is None:
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"huber_directional": huber_directional_loss()},
        )
    if _feature_scaler is None:
        _feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    if _target_scaler is None:
        _target_scaler = joblib.load(TARGET_SCALER_PATH)


def _load_metrics():
    metric_map = {
        "R²": "r2",
        "MSE": "mse",
        "RMSE": "rmse",
        "MAE": "mae",
        "Dir Acc": "dir_acc",
        "Dir Acc (Large Moves)": "dir_acc_large_moves",
    }

    open_ret = {}
    close_ret = {}
    open_px = {}
    close_px = {}

    with open("model_files/metrics.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Metric") or line.startswith("─"):
                continue

            match = re.match(
                r"^(.*?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(?:\s+(-?\d+(?:\.\d+)?))?(?:\s+(-?\d+(?:\.\d+)?))?$",
                line,
            )
            if not match:
                continue

            label = match.group(1).strip()
            key = metric_map.get(label)
            if key is None:
                continue

            values = [float(v) for v in match.groups()[1:] if v is not None]
            if len(values) >= 2:
                open_ret[key] = values[0]
                close_ret[key] = values[1]
            if len(values) >= 4:
                open_px[key] = values[2]
                close_px[key] = values[3]

    metrics = {
        "open_ret": open_ret,
        "close_ret": close_ret,
        "open_px": open_px,
        "close_px": close_px,
        "r2": close_px.get("r2"),
        "mse": close_px.get("mse"),
        "rmse": close_px.get("rmse"),
        "mae": close_px.get("mae"),
        "dir_acc": close_ret.get("dir_acc"),
        "dir_acc_large_moves": close_ret.get("dir_acc_large_moves"),
    }
    return metrics


# ── Data helpers ──────────────────────────────────────────────────────────────
def _load_and_preprocess():
    """
    Reproduces the exact preprocessing pipeline from training:
      1. Load CSV, parse dates.
      2. Impute NaNs.
      3. Compute log-returns for Open & Close.
      4. Drop the first row (NaN from shift).
    Returns df (log-returns), dates, raw_prices.
    """
    df_raw = pd.read_csv(DATA_URL)
    dates  = pd.to_datetime(df_raw["Date"])
    df_raw.drop(columns=["Date"], inplace=True)

    imputer = SimpleImputer(missing_values=np.nan)
    df = pd.DataFrame(
        imputer.fit_transform(df_raw), columns=df_raw.columns
    ).reset_index(drop=True)

    raw_prices = df[["Open", "Close"]].copy()

    for col in ["Open", "Close"]:
        df[col] = np.log(df[col] / df[col].shift(1))

    df.dropna(inplace=True)
    dates      = dates.iloc[1:].reset_index(drop=True)
    raw_prices = raw_prices.iloc[1:].reset_index(drop=True)
    df         = df.reset_index(drop=True)

    return df, dates, raw_prices


def _scale_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply feature + target scaling exactly as in training."""
    s = pd.DataFrame(
        _feature_scaler.transform(raw.values), columns=raw.columns
    )
    s[["Open", "Close"]] = _target_scaler.transform(
        raw[["Open", "Close"]].values
    )
    return s.astype(float)


# ── Public API ────────────────────────────────────────────────────────────────
def run_forecast(n_days: int = 5) -> dict:
    """
    Run recursive multi-step forecast.

    Parameters
    ----------
    n_days : int
        Number of trading days to forecast ahead (default 5).
        Accuracy degrades beyond ~5 steps.

    Returns
    -------
    dict with keys:
        "forecast_dates"   : list[str]  — ISO-format date strings
        "forecast_open"    : list[float]
        "forecast_close"   : list[float]
        "last_known_open"  : float
        "last_known_close" : float
        "forecast_prices"  : list[list[float]]
        "metrics"          : dict
    """
    _load_artifacts()

    # ── 1. Preprocess data ────────────────────────────────────────
    df, dates, raw_prices = _load_and_preprocess()

    # ── 2. Scale full dataset with the pre-fitted scalers ─────────
    full_scaled = _scale_df(df)

    # Column indices needed for window updates
    open_col  = list(df.columns).index("Open")
    close_col = list(df.columns).index("Close")

    # ── 3. Seed window & anchor price ────────────────────────────
    seed_window      = full_scaled.values[-WINDOW:].copy()   # (WINDOW, n_features)
    last_known_price = raw_prices.iloc[-1].values            # [open, close]
    last_known_date  = dates.iloc[-1]

    # ── 4. Recursive forecasting ──────────────────────────────────
    forecast_prices = []
    forecast_dates  = []

    current_window = seed_window.copy()
    current_price  = last_known_price.copy()
    current_date   = last_known_date

    for step in range(n_days):
        x_input = current_window[np.newaxis, :, :].astype(np.float32)  # (1, WINDOW, F)

        pred_scaled = _model.predict(x_input, verbose=0)[0]                    # (2,)
        pred_return = _target_scaler.inverse_transform([pred_scaled])[0]       # actual log-return

        next_price = current_price * np.exp(pred_return)
        forecast_prices.append(next_price)

        # Advance date by 1 trading day, skipping weekends
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        forecast_dates.append(next_date)
        current_date = next_date

        # Update window: carry last row forward, overwrite Open/Close
        new_row = current_window[-1].copy()
        new_row[open_col]  = pred_scaled[0]
        new_row[close_col] = pred_scaled[1]

        current_window = np.vstack([current_window[1:], new_row])
        current_price  = next_price

    forecast_prices = np.array(forecast_prices)   # (n_days, 2)

    metrics = _load_metrics()

    return {
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "forecast_open": forecast_prices[:, 0].tolist(),
        "forecast_close": forecast_prices[:, 1].tolist(),
        "forecast_prices": forecast_prices.tolist(),
        "last_known_open": float(last_known_price[0]),
        "last_known_close": float(last_known_price[1]),
        "metrics": metrics,
    }


if __name__ == "__main__":
    import json
    result = run_forecast()
    with open("model_files/latest_forecast.json", "w") as f:
        json.dump(result, f)
    print("Forecast saved to model_files/latest_forecast.json")
"""
Short-Term Electrical Load Forecasting for a Commercial Building (24 h ahead)
Author: Farrukh Hussain

MEP consultants size transformers and generators from connected-load estimates,
but operators need to know tomorrow's hourly demand to schedule generators,
avoid maximum-demand (MDI) penalties and plan load shedding. This project
forecasts hourly kW for a Karachi-style commercial building using weather,
calendar and lag features with Gradient Boosting, and compares it to a naive
"same hour last week" baseline.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape

RNG = np.random.default_rng(5)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)


def simulate_building(days=730):
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    doy, hr, dow = idx.dayofyear.values, idx.hour.values, idx.dayofweek.values
    temp = 27 + 7 * np.sin(2 * np.pi * (doy - 110) / 365) + 4 * np.sin(2 * np.pi * (hr - 9) / 24) + RNG.normal(0, 1.2, len(idx))
    humidity = np.clip(60 + 15 * np.sin(2 * np.pi * (doy - 180) / 365) + RNG.normal(0, 6, len(idx)), 20, 100)
    occupied = ((hr >= 9) & (hr <= 18) & (dow < 5)) | ((hr >= 10) & (hr <= 14) & (dow == 5))
    base = 180                                                    # 24/7 loads: servers, lifts standby, lighting
    hvac = np.clip(temp - 22, 0, None) * (22 + 18 * occupied)     # cooling load grows with temperature
    office = 320 * occupied                                       # lighting + small power
    load = base + hvac + office + RNG.normal(0, 15, len(idx))
    holidays = RNG.choice(np.unique(idx.normalize()), 20, replace=False)
    load[np.isin(idx.normalize(), holidays)] *= .55
    return pd.DataFrame({"kW": load, "temp_C": temp, "humidity": humidity}, index=idx)


def make_features(df):
    f = df.copy()
    f["hour"], f["dow"], f["month"] = f.index.hour, f.index.dayofweek, f.index.month
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * f.hour / 24), np.cos(2 * np.pi * f.hour / 24)
    f["lag_24h"] = f.kW.shift(24)
    f["lag_168h"] = f.kW.shift(168)
    f["roll_mean_24h"] = f.kW.shift(24).rolling(24).mean()
    f["cooling_degree"] = np.clip(f.temp_C - 22, 0, None)
    return f.dropna()


def main():
    df = make_features(simulate_building())
    split = df.index[-24 * 60]                                    # last 60 days = test
    tr, te = df[df.index < split], df[df.index >= split]
    X_cols = [c for c in df.columns if c != "kW"]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=.05, random_state=0)
    model.fit(tr[X_cols], tr.kW)
    pred = model.predict(te[X_cols])

    print("Naive (last week) MAPE : %.2f %%" % (100 * mape(te.kW, te.lag_168h)))
    print("Gradient Boosting MAPE : %.2f %%" % (100 * mape(te.kW, pred)))
    print("Forecast peak demand   : %.0f kW  (actual %.0f kW)" % (pred.max(), te.kW.max()))

    wk = slice(0, 24 * 7)
    plt.figure(figsize=(12, 4))
    plt.plot(te.index[wk], te.kW.values[wk], label="Actual", color="#0ea5e9")
    plt.plot(te.index[wk], pred[wk], "--", label="Forecast", color="#c8a84b")
    plt.ylabel("Demand (kW)"); plt.title("Hourly building load - 1 week of test data"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "forecast_week.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()

"""
Solar PV Power Forecasting + Performance Monitoring
Author: Farrukh Hussain

A 100 kWp rooftop PV plant (Karachi, lat 24.9 N). Physics first, ML second:
    1. Clear-sky irradiance from solar geometry (declination, hour angle)
    2. Cloud cover reduces irradiance; module temperature reduces efficiency
       (-0.4 %/C, NOCT model) - this is what the real plant does
    3. A Random Forest learns AC power from weather forecast inputs
    4. Performance Ratio (PR) is tracked and low-PR days flagged
       (soiling / inverter trip / string fault) - dust is a big issue in Sindh
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

LAT, KWP = np.radians(24.9), 100
RNG = np.random.default_rng(21)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)


def clear_sky_ghi(idx):
    doy = idx.dayofyear.values; hr = idx.hour.values + idx.minute.values / 60
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.radians(15 * (hr - 12))
    cos_z = np.sin(LAT) * np.sin(decl) + np.cos(LAT) * np.cos(decl) * np.cos(ha)
    return np.clip(1100 * cos_z, 0, None), np.clip(cos_z, 0, None)


def simulate(days=365):
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    ghi_cs, cos_z = clear_sky_ghi(idx)
    daily_cloud = np.repeat(RNG.beta(1.2, 4, days), 24)
    monsoon = ((idx.month >= 7) & (idx.month <= 8)) * .3
    cloud = np.clip(daily_cloud + monsoon + RNG.normal(0, .05, len(idx)), 0, 1)
    ghi = ghi_cs * (1 - .75 * cloud)
    t_amb = 26 + 6 * np.sin(2 * np.pi * (idx.dayofyear.values - 110) / 365) + 5 * np.sin(2 * np.pi * (idx.hour.values - 9) / 24)
    t_mod = t_amb + ghi / 800 * (45 - 20)                       # NOCT 45 C
    soiling = 1 - .002 * (idx.dayofyear.values % 30)                 # cleaned monthly
    dc = KWP * ghi / 1000 * (1 - .004 * (t_mod - 25)) * soiling
    ac = np.asarray(np.clip(dc * .965, 0, 95), dtype=float).copy()                            # inverter eff + clipping
    fault_days = RNG.choice(days, 6, replace=False)
    for d in fault_days:                                       # string fuse / inverter trip
        ac[d * 24:(d + 1) * 24] *= .6
    return pd.DataFrame({"ghi_cs": ghi_cs, "cos_zenith": cos_z, "cloud": cloud, "t_amb": t_amb,
                         "hour": idx.hour, "doy": idx.dayofyear, "ac_kw": ac + RNG.normal(0, .5, len(idx)).clip(0)},
                        index=idx), ghi, fault_days


def main():
    df, ghi, fault_days = simulate()
    feats = ["ghi_cs", "cos_zenith", "cloud", "t_amb", "hour", "doy"]
    tr, te = df[df.index.month <= 10], df[df.index.month > 10]
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0).fit(tr[feats], tr.ac_kw)
    pred = rf.predict(te[feats])
    day = te.cos_zenith > 0
    print("Day-time MAE: %.2f kW   R2: %.3f" % (mean_absolute_error(te.ac_kw[day], pred[day]), r2_score(te.ac_kw[day], pred[day])))

    daily_e = df.ac_kw.resample("D").sum()
    daily_h = pd.Series(ghi, index=df.index).resample("D").sum() / 1000
    pr = daily_e / (KWP * daily_h)
    flagged = pr[pr < pr.rolling(15, center=True, min_periods=5).median() * .8]
    print("Annual energy yield: %.0f MWh  (specific yield %.0f kWh/kWp)" % (daily_e.sum() / 1000, daily_e.sum() / KWP))
    print("Low-PR days flagged :", [d.strftime("%d-%b") for d in flagged.index])
    true_days = sorted(pd.Timestamp("2025-01-01") + pd.to_timedelta(int(d), "D") for d in fault_days)
    print("True fault days     :", [d.strftime("%d-%b") for d in true_days])

    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    s = slice(0, 24 * 5)
    ax[0].plot(te.index[s], te.ac_kw.values[s], label="Actual AC kW", color="#0ea5e9")
    ax[0].plot(te.index[s], pred[s], "--", label="Forecast", color="#c8a84b"); ax[0].legend(); ax[0].set_title("5-day PV forecast")
    ax[1].plot(pr.index, pr, color="#34d399", lw=.8); ax[1].scatter(flagged.index, flagged, color="red", zorder=3, label="Flagged")
    ax[1].set_title("Daily Performance Ratio"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pv_forecast_pr.png"), dpi=110); plt.close()


if __name__ == "__main__":
    main()

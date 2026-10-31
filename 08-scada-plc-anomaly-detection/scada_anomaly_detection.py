"""
SCADA / PLC Process Anomaly Detection for a Film Extrusion Line
Author: Farrukh Hussain

Based on my years maintaining a continuous BOPP film plant: a line has
extruder motor current, melt temperature, melt pressure, line speed, DC drive
current and film thickness tags logged by the PLC every second. Failures
rarely appear in one tag - they show up as a broken *relationship* between
tags (e.g. pressure rising while speed is constant = screen pack blocking).

Method (unsupervised - plants rarely have labelled faults):
    1. Learn normal multivariate behaviour with PCA (Hotelling T2 + SPE/Q statistic)
    2. Isolation Forest as a second opinion
    3. Raise alarm only if the anomaly persists (debounce, like a PLC timer)
    4. Contribution plot tells the engineer WHICH tag caused the alarm
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

RNG = np.random.default_rng(4)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)
TAGS = ["extruder_amp", "melt_temp_C", "melt_press_bar", "line_speed_mpm", "dc_drive_amp", "thickness_um"]


def simulate(n=8000, faults=True):
    speed = 250 + np.cumsum(RNG.normal(0, .05, n)).clip(-15, 15)
    amp = 180 + .5 * (speed - 250) + RNG.normal(0, 1.5, n)
    temp = 245 + .02 * (amp - 180) + RNG.normal(0, .4, n)
    press = 160 + .3 * (speed - 250) - .8 * (temp - 245) + RNG.normal(0, 1, n)
    drive = 95 + .35 * (speed - 250) + RNG.normal(0, .8, n)
    thick = 20 + .02 * (press - 160) - .01 * (speed - 250) + RNG.normal(0, .05, n)
    X = np.c_[amp, temp, press, speed, drive, thick]
    label = np.zeros(n, dtype=int)
    if faults:
        # Fault 1: screen-pack blockage - pressure ramps up, everything else normal
        s = slice(3000, 3400); X[s, 2] += np.linspace(0, 18, 400); label[s] = 1
        # Fault 2: heater band failure - melt temp drifts down, pressure rises
        s = slice(5200, 5600); r = np.linspace(0, 8, 400); X[s, 1] -= 1.5 * r; X[s, 2] += .8 * r; label[s] = 1
        # Fault 3: DC drive bearing - drive current rises with no speed change
        s = slice(7000, 7300); X[s, 4] += np.linspace(0, 10, 300); label[s] = 1
    return pd.DataFrame(X, columns=TAGS), label


def main():
    train, _ = simulate(faults=False)
    test, label = simulate()
    sc = StandardScaler().fit(train)
    Ztr, Zte = sc.transform(train), sc.transform(test)
    pca = PCA(n_components=3).fit(Ztr)

    def spe(Z):
        rec = pca.inverse_transform(pca.transform(Z)); return ((Z - rec) ** 2).sum(1), (Z - rec) ** 2
    def t2(Z):
        return ((pca.transform(Z) ** 2) / pca.explained_variance_).sum(1)
    spe_tr, _ = spe(Ztr); spe_te, _ = spe(Zte)
    # combined index: SPE catches broken correlations, T2 catches large moves inside the model
    stat_tr = spe_tr / np.percentile(spe_tr, 99) + t2(Ztr) / np.percentile(t2(Ztr), 99)
    stat_te = spe_te / np.percentile(spe_tr, 99) + t2(Zte) / np.percentile(t2(Ztr), 99)
    limit = np.percentile(stat_tr, 99.5)
    contrib = Zte ** 2                                                   # per-tag deviation for root cause
    spe_te = stat_te

    raw = spe_te > limit
    alarm = pd.Series(raw).rolling(20).sum().fillna(0).values >= 15        # 15 of last 20 s -> alarm
    iso = IsolationForest(contamination=.005, random_state=0).fit(Ztr)
    iso_flag = iso.predict(Zte) == -1

    tp = (alarm & (label == 1)).sum(); fp = (alarm & (label == 0)).sum()
    print("PCA (T2+SPE) alarm: detected %.1f%% of fault samples, false alarms %d s out of %d s normal"
          % (100 * tp / label.sum(), fp, (label == 0).sum()))
    print("IsolationForest flagged %.1f%% of fault samples" % (100 * (iso_flag & (label == 1)).sum() / label.sum()))
    for name, s in [("Screen-pack blockage", slice(3000, 3400)), ("Heater band failure", slice(5200, 5600)), ("DC drive bearing", slice(7000, 7300))]:
        first = np.argmax(alarm[s]); top = pd.Series(contrib[s].mean(0), index=TAGS).idxmax()
        print(f"  {name:22s} alarm after {first:3d} s | root-cause tag -> {top}")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(test.melt_press_bar, lw=.6, color="#0ea5e9"); ax[0].set_ylabel("melt pressure (bar)")
    ax[1].plot(spe_te, lw=.6, color="#c8a84b"); ax[1].axhline(limit, color="red", ls="--", label="99.5% limit")
    ax[1].fill_between(range(len(alarm)), 0, spe_te.max(), where=alarm, color="red", alpha=.15, label="ALARM")
    ax[1].set_ylabel("combined T2 + SPE index"); ax[1].legend(); ax[1].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "scada_anomaly.png"), dpi=110); plt.close()


if __name__ == "__main__":
    main()

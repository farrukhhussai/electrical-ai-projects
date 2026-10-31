"""
Electricity Theft / Non-Technical Loss Detection from Smart Meter Data
Author: Farrukh Hussain

Distribution companies in Pakistan lose a large share of energy to theft
(meter tampering, direct hooking, bypass). Smart meters (AMI) produce daily
kWh readings; theft patterns look different from honest consumption:
    * sudden permanent drop after a "normal" period (bypass installed)
    * readings reduced by a constant factor (meter tampering / CT shunting)
    * zero readings on random days
    * consumption only at night (hooking during day)
Pipeline: feature engineering on 1-year daily profiles -> Gradient Boosting
(supervised) + Isolation Forest (unsupervised, for unlabelled feeders).
Output: ranked inspection list for field teams.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

RNG = np.random.default_rng(9)
DAYS = 365
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)


def honest_profile():
    base = RNG.uniform(5, 30)                                   # kWh/day
    season = 1 + .45 * np.sin(2 * np.pi * (np.arange(DAYS) - 100) / 365)   # summer AC load
    weekly = 1 + .08 * (np.arange(DAYS) % 7 >= 5)
    return base * season * weekly * RNG.lognormal(0, .15, DAYS)


def theft_profile(p):
    kind = RNG.integers(4)
    p = p.copy()
    if kind == 0:   # bypass from day X
        p[RNG.integers(60, 300):] *= RNG.uniform(.1, .5)
    elif kind == 1:  # tampering factor
        p *= RNG.uniform(.3, .7)
    elif kind == 2:  # random zero days
        p[RNG.random(DAYS) < RNG.uniform(.1, .4)] = 0
    else:            # flattening - reads only a capped value
        p = np.minimum(p, np.percentile(p, RNG.uniform(10, 40)))
    return p


def features(p):
    m = p.reshape(-1, 73).mean(axis=1)                          # 5 periods of 73 days
    return {
        "mean": p.mean(), "std": p.std(), "cv": p.std() / (p.mean() + 1e-6),
        "zero_days": (p < .1).sum(), "min": p.min(), "max": p.max(),
        "max_drop_ratio": (m[1:] / (m[:-1] + 1e-6)).min(),
        "summer_winter_ratio": p[150:240].mean() / (p[0:60].mean() + 1e-6),
        "plateau_ratio": (np.abs(np.diff(p)) < 1e-3).mean(),
        "last_vs_first_q": p[-90:].mean() / (p[:90].mean() + 1e-6),
        "autocorr_7": pd.Series(p).autocorr(7),
    }


def main():
    n, theft_rate = 6000, .12
    profiles, labels = [], []
    for _ in range(n):
        p = honest_profile(); thief = RNG.random() < theft_rate
        profiles.append(theft_profile(p) if thief else p); labels.append(int(thief))
    X = pd.DataFrame([features(p) for p in profiles]).fillna(0); y = np.array(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)

    gb = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X_tr, y_tr)
    prob = gb.predict_proba(X_te)[:, 1]
    print("Supervised GBM   ROC-AUC: %.3f" % roc_auc_score(y_te, prob))
    top = np.argsort(-prob)[: int(.12 * len(prob))]              # inspect top 12 % of customers
    hit = y_te[top].mean()
    print("Inspect top 12%% -> hit rate %.1f%% (random inspection: %.1f%%)" % (100 * hit, 100 * y_te.mean()))
    yp = (prob > .5).astype(int)
    print("Precision %.2f  Recall %.2f" % (precision_score(y_te, yp), recall_score(y_te, yp)))

    iso = IsolationForest(contamination=theft_rate, random_state=0).fit(X_tr)
    print("Unsupervised IsolationForest ROC-AUC: %.3f" % roc_auc_score(y_te, -iso.score_samples(X_te)))

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    i_h, i_t = np.where(y == 0)[0][0], np.where(y == 1)[0][:3]
    ax[0].plot(profiles[i_h], color="#34d399"); ax[0].set_title("Honest consumer - daily kWh")
    for i in i_t:
        ax[1].plot(profiles[i], lw=.8)
    ax[1].set_title("Suspected theft patterns"); ax[1].set_xlabel("day of year")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "consumption_patterns.png"), dpi=110); plt.close()


if __name__ == "__main__":
    main()

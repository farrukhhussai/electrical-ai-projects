"""
Power Quality Disturbance Classification (IEEE 1159) using Signal Features + ML
Author: Farrukh Hussain

Poor power quality trips VFDs, overheats transformers and damages electronics.
Power quality analysers record thousands of events; this project classifies them
automatically into IEEE 1159 categories:
    NORMAL, SAG, SWELL, INTERRUPTION, HARMONICS, TRANSIENT, FLICKER, SAG+HARMONICS
Features: RMS profile, THD, crest factor, wavelet-like energy bands (via FFT),
then a Random Forest classifier.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

F0, FS, CYCLES = 50, 3200, 10           # 50 Hz grid (Pakistan / Europe), 64 samples/cycle
t = np.arange(0, CYCLES / F0, 1 / FS)
SPC = FS // F0
RNG = np.random.default_rng(3)
CLASSES = ["NORMAL", "SAG", "SWELL", "INTERRUPTION", "HARMONICS", "TRANSIENT", "FLICKER", "SAG+HARMONICS"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)


def window():
    t1 = RNG.uniform(0.04, 0.09); t2 = t1 + RNG.uniform(0.04, 0.09)
    return ((t >= t1) & (t <= t2)).astype(float)


def harmonics():
    return (RNG.uniform(.01, .12) * np.sin(2 * np.pi * 3 * F0 * t) +
            RNG.uniform(.01, .12) * np.sin(2 * np.pi * 5 * F0 * t) +
            RNG.uniform(.01, .12) * np.sin(2 * np.pi * 7 * F0 * t))


def simulate(kind):
    v = np.sin(2 * np.pi * F0 * t)
    if kind == "SAG":
        v *= 1 - RNG.uniform(.03, .9) * window()
    elif kind == "SWELL":
        v *= 1 + RNG.uniform(.03, .8) * window()
    elif kind == "INTERRUPTION":
        v *= 1 - RNG.uniform(.9, 1.0) * window()
    elif kind == "HARMONICS":
        v += harmonics()
    elif kind == "TRANSIENT":
        t0 = RNG.uniform(.04, .15)
        v += RNG.uniform(.5, 1.5) * np.exp(-(t - t0) / .002) * np.sin(2 * np.pi * RNG.uniform(900, 1300) * t) * (t >= t0)
    elif kind == "FLICKER":
        v *= 1 + RNG.uniform(.01, .2) * np.sin(2 * np.pi * RNG.uniform(8, 25) * t)
    elif kind == "SAG+HARMONICS":
        v = (v + harmonics()) * (1 - RNG.uniform(.03, .9) * window())
    return v + RNG.normal(0, .03, len(t))


def features(v):
    rms = np.sqrt(np.mean(v.reshape(-1, SPC) ** 2, axis=1)) * np.sqrt(2)  # per-cycle RMS (pu)
    spec = np.abs(np.fft.rfft(v)) / len(v)
    f = np.fft.rfftfreq(len(v), 1 / FS)
    fund = spec[np.argmin(abs(f - F0))]
    harm = np.sqrt(sum(spec[np.argmin(abs(f - h * F0))] ** 2 for h in range(2, 16)))
    return {
        "rms_min": rms.min(), "rms_max": rms.max(), "rms_std": rms.std(),
        "thd": harm / fund, "crest": np.abs(v).max() / np.sqrt(np.mean(v ** 2)),
        "hf_energy": (spec[f > 800] ** 2).sum(), "lf_mod": rms.std() / rms.mean(),
        "rms_range": rms.max() - rms.min(), "diff_max": np.abs(np.diff(v)).max(),
    }


def main():
    df = pd.DataFrame([{**features(simulate(c)), "label": c} for c in CLASSES for _ in range(300)])
    X, y = df.drop(columns="label"), df.label
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    print("5-fold CV accuracy: %.3f" % cross_val_score(clf, X, y, cv=5).mean())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, stratify=y, random_state=0)
    clf.fit(X_tr, y_tr); p = clf.predict(X_te)
    print("Hold-out accuracy : %.3f" % accuracy_score(y_te, p))

    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)
    for ax, c in zip(axes.ravel(), CLASSES):
        ax.plot(t * 1000, simulate(c), lw=.8, color="#0ea5e9"); ax.set_title(c, fontsize=9); ax.set_ylim(-2, 2)
    fig.supxlabel("time (ms)"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "disturbance_waveforms.png"), dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y_te, p, ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=110); plt.close()


if __name__ == "__main__":
    main()

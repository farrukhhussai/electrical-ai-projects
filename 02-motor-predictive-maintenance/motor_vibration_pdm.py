"""
Induction Motor Predictive Maintenance - Vibration Signal Analysis + ML
Author: Farrukh Hussain

On a production plant, an unplanned motor failure stops the line. Each mechanical
fault leaves a fingerprint in the vibration spectrum:
    * Unbalance        -> high amplitude at 1x running speed
    * Misalignment     -> strong 2x (and 3x) running speed
    * Bearing defect   -> high-frequency impacts at bearing fault frequency (BPFO)
    * Looseness        -> many harmonics + sub-harmonics
This project simulates accelerometer signals, extracts time + FFT features
(the same ones a vibration analyst reads), and trains a classifier.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

FS = 5000            # sampling frequency (Hz)
T = 1.0              # 1-second capture
RPM = 1480           # 4-pole motor, 50 Hz supply
F_RUN = RPM / 60     # ~24.7 Hz
BPFO = 3.58 * F_RUN  # outer-race bearing defect frequency (typical 6205 bearing)
RNG = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)
t = np.arange(0, T, 1 / FS)
CLASSES = ["HEALTHY", "UNBALANCE", "MISALIGNMENT", "BEARING_FAULT", "LOOSENESS"]


def simulate(condition):
    s = 0.2 * np.sin(2 * np.pi * F_RUN * t)
    s += 0.05 * np.sin(2 * np.pi * 2 * F_RUN * t)
    sev = RNG.uniform(0.1, 1.4)          # early-stage faults are hard to see
    if condition == "UNBALANCE":
        s += sev * 0.8 * np.sin(2 * np.pi * F_RUN * t + RNG.uniform(0, 6))
    elif condition == "MISALIGNMENT":
        s += sev * 0.6 * np.sin(2 * np.pi * 2 * F_RUN * t) + sev * 0.25 * np.sin(2 * np.pi * 3 * F_RUN * t)
    elif condition == "BEARING_FAULT":
        impacts = (np.sin(2 * np.pi * BPFO * t) > 0.995).astype(float)
        ring = np.exp(-t[:200] * 900) * np.sin(2 * np.pi * 1800 * t[:200])
        s += sev * 2.0 * np.convolve(impacts, ring, mode="same")
    elif condition == "LOOSENESS":
        for k in range(1, 8):
            s += sev * 0.15 * np.sin(2 * np.pi * k * F_RUN * t + RNG.uniform(0, 6))
        s += sev * 0.15 * np.sin(2 * np.pi * 0.5 * F_RUN * t)
    return s + RNG.normal(0, 0.25, len(t))


def band_amp(freqs, mag, f0, bw=2.0):
    m = (freqs > f0 - bw) & (freqs < f0 + bw)
    return mag[m].max() if m.any() else 0.0


def extract_features(sig):
    freqs = np.fft.rfftfreq(len(sig), 1 / FS)
    mag = np.abs(np.fft.rfft(sig)) * 2 / len(sig)
    return {
        "rms": np.sqrt(np.mean(sig ** 2)),
        "peak": np.max(np.abs(sig)),
        "crest_factor": np.max(np.abs(sig)) / np.sqrt(np.mean(sig ** 2)),
        "kurtosis": kurtosis(sig),
        "skewness": skew(sig),
        "amp_0.5x": band_amp(freqs, mag, 0.5 * F_RUN),
        "amp_1x": band_amp(freqs, mag, F_RUN),
        "amp_2x": band_amp(freqs, mag, 2 * F_RUN),
        "amp_3x": band_amp(freqs, mag, 3 * F_RUN),
        "amp_BPFO": band_amp(freqs, mag, BPFO),
        "hf_energy": mag[freqs > 1000].sum(),
        "harmonic_sum_4to7x": sum(band_amp(freqs, mag, k * F_RUN) for k in range(4, 8)),
    }


def main():
    rows = []
    for c in CLASSES:
        for _ in range(250):
            f = extract_features(simulate(c)); f["condition"] = c; rows.append(f)
    df = pd.DataFrame(rows)
    X, y = df.drop(columns="condition"), df.condition
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    clf = GradientBoostingClassifier(random_state=1).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    print(f"Fault classification accuracy: {accuracy_score(y_te, pred):.3f}\n")
    print(classification_report(y_te, pred))

    # Plot: waveform + spectrum for each condition
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(12, 11))
    for i, c in enumerate(CLASSES):
        sig = simulate(c)
        freqs = np.fft.rfftfreq(len(sig), 1 / FS); mag = np.abs(np.fft.rfft(sig)) * 2 / len(sig)
        axes[i, 0].plot(t[:1000], sig[:1000], lw=.7); axes[i, 0].set_ylabel(c, fontsize=8)
        axes[i, 1].plot(freqs[freqs < 300], mag[freqs < 300], color="#c8a84b")
    axes[0, 0].set_title("Vibration waveform (0.2 s)"); axes[0, 1].set_title("FFT spectrum (0-300 Hz)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "vibration_signatures.png"), dpi=110); plt.close()

    new = extract_features(simulate("BEARING_FAULT"))
    print("Live sensor reading diagnosis ->", clf.predict(pd.DataFrame([new]))[0])


if __name__ == "__main__":
    main()

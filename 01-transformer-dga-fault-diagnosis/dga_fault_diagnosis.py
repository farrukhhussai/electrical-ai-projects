"""
Transformer Fault Diagnosis using Dissolved Gas Analysis (DGA) + Machine Learning
Author: Farrukh Hussain

Power transformers fail slowly. Before a failure, the insulating oil breaks down
and releases gases (H2, CH4, C2H6, C2H4, C2H2). The ratio of these gases tells
an engineer what kind of fault is developing. Classical methods (IEC 60599
ratios, Duval Triangle) are rule-based; here a Random Forest learns the same
patterns from data and is compared against the IEC ratio method.

Fault classes (IEC 60599):
    NORMAL, PD (partial discharge), D1 (low-energy discharge),
    D2 (high-energy discharge / arcing), T1 (<300C), T2 (300-700C), T3 (>700C)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
RNG = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# Typical gas signature (ppm) per fault type - based on IEC 60599 / IEEE C57.104 patterns
SIGNATURES = {
    "NORMAL": [40, 20, 30, 15, 0.5],
    "PD":     [900, 60, 15, 5, 1],
    "D1":     [300, 40, 10, 60, 120],
    "D2":     [400, 120, 20, 300, 350],
    "T1":     [60, 150, 180, 40, 1],
    "T2":     [80, 300, 90, 350, 3],
    "T3":     [150, 400, 80, 900, 15],
}


def generate_dataset(n_per_class=400):
    """Create a realistic DGA dataset with log-normal measurement scatter."""
    rows = []
    for fault, base in SIGNATURES.items():
        base = np.array(base, dtype=float)
        for _ in range(n_per_class):
            sample = base * RNG.lognormal(0, 0.75, size=5)
            rows.append(list(sample) + [fault])
    return pd.DataFrame(rows, columns=GASES + ["fault"])


def add_ratio_features(df):
    """IEC 60599 key gas ratios used by protection engineers."""
    eps = 1e-6
    df = df.copy()
    df["R1_C2H2_C2H4"] = df.C2H2 / (df.C2H4 + eps)
    df["R2_CH4_H2"] = df.CH4 / (df.H2 + eps)
    df["R3_C2H4_C2H6"] = df.C2H4 / (df.C2H6 + eps)
    total = df[["CH4", "C2H4", "C2H2"]].sum(axis=1) + eps   # Duval triangle %
    df["duval_CH4"] = 100 * df.CH4 / total
    df["duval_C2H4"] = 100 * df.C2H4 / total
    df["duval_C2H2"] = 100 * df.C2H2 / total
    df["TDCG"] = df[GASES].sum(axis=1)                       # total dissolved combustible gas
    for g in GASES:
        df["log_" + g] = np.log10(df[g] + 1)
    return df


def iec_ratio_method(row):
    """Simplified rule-based IEC 60599 interpretation (baseline)."""
    r1, r2, r3 = row.R1_C2H2_C2H4, row.R2_CH4_H2, row.R3_C2H4_C2H6
    if row.TDCG < 200:
        return "NORMAL"
    if r2 < 0.1 and r3 < 0.2:
        return "PD"
    if r1 > 1 and r3 > 1 and r2 < 0.5:
        return "D1"
    if 0.6 <= r1 <= 2.5 and r3 > 2:
        return "D2"
    if r3 < 1 and r2 > 1:
        return "T1"
    if r1 < 0.1 and 1 <= r3 <= 4:
        return "T2"
    if r1 < 0.2 and r3 > 4:
        return "T3"
    return "UNDETERMINED"


def main():
    df = add_ratio_features(generate_dataset())
    features = [c for c in df.columns if c != "fault"]
    X_tr, X_te, y_tr, y_te = train_test_split(df[features], df.fault, test_size=0.25,
                                              stratify=df.fault, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    iec_pred = X_te.apply(iec_ratio_method, axis=1)
    print(f"IEC ratio method accuracy : {accuracy_score(y_te, iec_pred):.3f}")
    print(f"Random Forest accuracy    : {accuracy_score(y_te, pred):.3f}\n")
    print(classification_report(y_te, pred))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Transformer DGA Fault Diagnosis - Random Forest")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120); plt.close()

    imp = pd.Series(model.feature_importances_, index=features).sort_values().tail(10)
    imp.plot.barh(figsize=(7, 5), color="#c8a84b", title="Top 10 diagnostic features")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=120); plt.close()

    # Diagnose a new oil sample (ppm)
    sample = pd.DataFrame([[350, 110, 25, 280, 300]], columns=GASES)
    sample = add_ratio_features(sample)[features]
    print("New oil sample diagnosis ->", model.predict(sample)[0])


if __name__ == "__main__":
    main()

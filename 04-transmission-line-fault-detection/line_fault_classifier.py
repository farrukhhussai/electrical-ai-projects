"""
Transmission Line Fault Detection, Classification & Location using ML
Author: Farrukh Hussain

A 132 kV line is modelled with lumped source + line impedance. For every fault
type (AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC) at a random distance and fault
resistance, the relay-point 3-phase voltage & current phasors are computed using
symmetrical components (sequence networks). Two models are trained:
    1. Classifier  -> fault type (what a distance relay must decide in < 20 ms)
    2. Regressor   -> fault distance in km (helps line crew find the fault)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

RNG = np.random.default_rng(11)
a = np.exp(2j * np.pi / 3)
A = np.array([[1, 1, 1], [1, a * a, a], [1, a, a * a]])          # sequence -> phase
V_PH = 132e3 / np.sqrt(3)
LINE_KM = 100
Z1_KM, Z0_KM = 0.03 + 0.38j, 0.12 + 1.2j                          # ohm/km (typical ACSR)
ZS1, ZS0 = 2 + 12j, 3 + 20j                                       # source impedance
FAULTS = ["NO_FAULT", "AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)

# rotation so any phase-A based formula works for B / C faults
ROT = {"A": 0, "B": 1, "C": 2}


def fault_phasors(ftype, d_km, rf):
    Z1 = ZS1 + Z1_KM * d_km; Z2 = Z1; Z0 = ZS0 + Z0_KM * d_km
    E = V_PH
    if ftype == "NO_FAULT":
        load = V_PH / (300 + 60j) * RNG.uniform(.5, 1.2)
        I = np.array([load, load * a * a, load * a]); V = np.array([E, E * a * a, E * a]) * .98
        return V, I
    phases = ftype.replace("G", "")
    ground = ftype.endswith("G") or len(phases) == 1
    if len(phases) == 1:                                           # single line to ground
        I1 = E / (Z1 + Z2 + Z0 + 3 * rf); I2 = I0 = I1
        ref = phases
    elif len(phases) == 2 and not ground:                          # line-line
        I1 = E / (Z1 + Z2 + rf); I2 = -I1; I0 = 0
        ref = ({"A", "B", "C"} - set(phases)).pop()
    elif len(phases) == 2:                                         # double line to ground
        Zp = Z2 * (Z0 + 3 * rf) / (Z2 + Z0 + 3 * rf)
        I1 = E / (Z1 + Zp); I2 = -I1 * (Z0 + 3 * rf) / (Z2 + Z0 + 3 * rf); I0 = -I1 - I2
        ref = ({"A", "B", "C"} - set(phases)).pop()
    else:                                                          # three phase
        I1 = E / (Z1 + rf); I2 = I0 = 0; ref = "A"
    Iseq = np.array([I0, I1, I2])
    Vseq = np.array([-ZS0 * I0, E - ZS1 * I1, -ZS1 * I2])         # voltage at relay (source bus)
    I = A @ Iseq; V = A @ Vseq
    k = ROT[ref]                                                   # rotate to the faulted phase
    return np.roll(V, k), np.roll(I, k)


def build(n=600):
    rows = []
    for f in FAULTS:
        for _ in range(n):
            d = RNG.uniform(1, LINE_KM); rf = RNG.uniform(0.01, 150)
            V, I = fault_phasors(f, d, rf)
            V = V * (1 + RNG.normal(0, .04, 3)); I = I * (1 + RNG.normal(0, .04, 3))   # CT/VT error
            r = {f"V{p}_mag": abs(V[i]) / V_PH for i, p in enumerate("abc")}
            r.update({f"I{p}_mag": abs(I[i]) / 1000 for i, p in enumerate("abc")})
            r.update({f"I{p}_ang": np.angle(I[i]) for i, p in enumerate("abc")})
            r["I0_mag"] = abs(I.sum()) / 3000
            r["Z_apparent"] = np.min(np.abs(V) / (np.abs(I) + 1e-3))
            r["fault"], r["distance_km"] = f, (d if f != "NO_FAULT" else 0)
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = build()
    X = df.drop(columns=["fault", "distance_km"]); y = df.fault
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(X, y, df.distance_km, test_size=.25, random_state=0, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)
    print("Fault-type classification accuracy: %.3f" % accuracy_score(y_te, clf.predict(X_te)))

    m_tr, m_te = (y_tr != "NO_FAULT").values, (y_te != "NO_FAULT").values
    reg = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_tr[m_tr], d_tr[m_tr])
    pred_d = reg.predict(X_te[m_te])
    print("Fault location MAE: %.2f km on a %d km line" % (mean_absolute_error(d_te[m_te], pred_d), LINE_KM))

    plt.figure(figsize=(6, 6))
    plt.scatter(d_te[m_te], pred_d, s=6, alpha=.5, color="#0ea5e9")
    plt.plot([0, LINE_KM], [0, LINE_KM], "--", color="#c8a84b")
    plt.xlabel("Actual fault distance (km)"); plt.ylabel("Predicted (km)"); plt.title("Fault location estimate")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fault_location.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()

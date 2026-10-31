"""
Li-ion Battery State-of-Charge (SoC) Estimation: Coulomb Counting vs Kalman Filter vs ML
Author: Farrukh Hussain

UPS, solar inverters and EVs all need to know how much charge is left.
    * Coulomb counting integrates current - simple, but drifts with sensor bias
    * Extended Kalman Filter (EKF) fuses current with a 1-RC equivalent-circuit
      model and the measured terminal voltage - the industry standard in a BMS
    * A neural network (MLP) learns SoC from voltage, current, temperature history
A 1-RC Thevenin model of an NMC cell generates the 'true' data with realistic
sensor noise and a current-sensor offset.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RNG = np.random.default_rng(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT, exist_ok=True)
Q_AH, R0, R1, C1, DT = 2.5, 0.05, 0.03, 2000.0, 1.0


def ocv(soc):                                     # NMC open-circuit voltage curve
    soc = np.clip(soc, 1e-3, 1)
    return 3.0 + 1.2 * soc - 0.25 * np.exp(-20 * soc) + 0.15 * np.log(soc + 1)


def docv(soc, h=1e-4):
    return (ocv(soc + h) - ocv(soc - h)) / (2 * h)


def drive_cycle(n, seed):
    r = np.random.default_rng(seed); i = np.zeros(n); k = 0
    while k < n:
        dur = r.integers(30, 300); i[k:k + dur] = r.choice([0, 0.5, 1.0, 2.0, -1.0, 2.5]); k += dur
    return i                                      # +discharge / -charge (A)


def simulate(i, soc0=0.95):
    soc, v1, out, i_act = soc0, 0.0, [], []
    a = np.exp(-DT / (R1 * C1))
    for ik in i:
        if (soc <= 0.05 and ik > 0) or (soc >= 0.99 and ik < 0):
            ik = 0.0                               # BMS cut-off: no discharge below 5 %, no charge above 99 %
        soc -= ik * DT / (3600 * Q_AH); v1 = a * v1 + R1 * (1 - a) * ik
        out.append((soc, ocv(soc) - v1 - R0 * ik)); i_act.append(ik)
    s, v = map(np.array, zip(*out)); i_act = np.array(i_act)
    return s, v + RNG.normal(0, .005, len(v)), i_act + 0.05 + RNG.normal(0, .02, len(i_act))  # 50 mA sensor bias


def coulomb(i_meas, soc0):
    return soc0 - np.cumsum(i_meas) * DT / (3600 * Q_AH)


def ekf(i_meas, v_meas, soc0=0.7):
    a = np.exp(-DT / (R1 * C1)); x = np.array([soc0, 0.]); P = np.diag([.1, .01])
    Q, R, est = np.diag([1e-7, 1e-6]), 1e-4, []
    for ik, vk in zip(i_meas, v_meas):
        x = np.array([x[0] - ik * DT / (3600 * Q_AH), a * x[1] + R1 * (1 - a) * ik])
        F = np.diag([1, a]); P = F @ P @ F.T + Q
        H = np.array([docv(x[0]), -1.0]); S = H @ P @ H + R; K = P @ H / S
        x = x + K * (vk - (ocv(x[0]) - x[1] - R0 * ik)); P = (np.eye(2) - np.outer(K, H)) @ P
        est.append(x[0])
    return np.array(est)


def ml_features(v, i):
    def ma(x, w):                                  # causal moving average (only past samples)
        c = np.cumsum(np.r_[np.zeros(w), x]); return (c[w:] - c[:-w]) / w
    v_ir = v + R0 * i                              # IR-compensated voltage ~ OCV - V_RC
    return np.c_[v, i, v_ir, ma(v_ir, 60), ma(v_ir, 600), ma(i, 60), ma(i, 600)]


def main():
    n = 9000
    Xs, ys = [], []
    for seed in range(20):                          # training cycles
        s, v, im = simulate(drive_cycle(n, seed), RNG.uniform(.6, 1))
        Xs.append(ml_features(v, im)); ys.append(s)
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0))
    mlp.fit(np.vstack(Xs), np.concatenate(ys))

    s, v, im = simulate(drive_cycle(n, 99), 0.95)  # unseen test cycle
    cc = np.clip(coulomb(im, 0.85), 0, 1)                         # wrong initial SoC (unknown after rest)
    kf = ekf(im, v, 0.70)                          # EKF starts even further off
    nn = np.clip(mlp.predict(ml_features(v, im)), 0, 1)
    for name, e in [("Coulomb counting", cc), ("Extended Kalman Filter", kf), ("Neural network (MLP)", nn)]:
        print(f"{name:24s} RMSE = {100 * np.sqrt(np.mean((e - s) ** 2)):.2f} % SoC")

    plt.figure(figsize=(11, 4.5))
    plt.plot(s * 100, "k", lw=2, label="True SoC")
    plt.ylim(-5, 105)
    plt.plot(cc * 100, label="Coulomb counting"); plt.plot(kf * 100, label="EKF"); plt.plot(nn * 100, alpha=.7, label="MLP")
    plt.xlabel("time (s)"); plt.ylabel("SoC (%)"); plt.legend(); plt.title("Battery SoC estimation on an unseen drive cycle")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "soc_estimation.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()

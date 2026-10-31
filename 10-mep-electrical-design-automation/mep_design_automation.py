"""
MEP Electrical Design Automation - Load Schedule -> Breaker, Cable Sizing, Voltage Drop, Transformer & BOQ
Author: Farrukh Hussain

In 6 years of MEP consulting, the most repetitive job was the same loop for every
circuit: design current -> breaker -> cable size with derating -> voltage-drop
check -> upsize -> BOQ. This tool automates that loop per IEC 60364-5-52 and
writes a ready-to-review Excel load schedule + BOQ, plus transformer and
generator sizing with diversity. A senior engineer still reviews the output -
the tool removes the hours of manual table look-ups and arithmetic errors.

Usage:
    python mep_design_automation.py load_schedule.csv
"""
import os
import sys
import math
import pandas as pd

V_LL, V_LN = 400, 230
RHO_CU = 0.0225               # ohm.mm2/m at operating temperature
X_PER_M = 0.08e-3             # ohm/m cable reactance (typical multicore)
SIZES = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240, 300]
# IEC 60364-5-52 Table B.52.4 / B.52.5 - copper, reference method C (clipped direct / tray), 3 loaded conductors
IZ = {"PVC":  [17.5, 24, 32, 41, 57, 76, 96, 119, 144, 184, 223, 259, 299, 341, 403, 464],
      "XLPE": [22, 30, 40, 51, 70, 94, 119, 148, 180, 232, 282, 328, 379, 434, 514, 593]}
# Table B.52.14 ambient temperature correction (air), Table B.52.17 grouping
K_TEMP = {"PVC": {30: 1.0, 35: .94, 40: .87, 45: .79, 50: .71}, "XLPE": {30: 1.0, 35: .96, 40: .91, 45: .87, 50: .82}}
K_GROUP = {1: 1.0, 2: .85, 3: .79, 4: .75, 5: .73, 6: .72, 7: .72, 8: .71, 9: .70}
BREAKERS = [6, 10, 16, 20, 25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250]
TX_KVA = [100, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000]
VD_LIMIT = {"lighting": 3.0, "power": 5.0, "motor": 5.0}


def size_circuit(r):
    v = V_LL if r.phases == 3 else V_LN
    ib = r.load_kw * 1000 / ((math.sqrt(3) if r.phases == 3 else 1) * v * r.pf)
    in_ = next(b for b in BREAKERS if b >= ib * (1.25 if r.type == "motor" else 1.0))
    k = K_TEMP[r.insulation][int(r.ambient_c)] * K_GROUP[min(int(r.grouped_circuits), 9)]
    sin_phi = math.sqrt(1 - r.pf ** 2)
    for i, s in enumerate(SIZES):
        iz = IZ[r.insulation][i] * k
        if iz < in_:                                          # IEC: Ib <= In <= Iz
            continue
        z = RHO_CU / s * r.pf + X_PER_M * sin_phi
        vd = (math.sqrt(3) if r.phases == 3 else 2) * ib * r.length_m * z
        vd_pct = 100 * vd / v
        if vd_pct <= VD_LIMIT[r.type]:
            return pd.Series({"Ib_A": round(ib, 1), "breaker_A": in_, "derating_k": round(k, 3),
                              "cable_mm2": s, "Iz_derated_A": round(iz, 1), "VD_%": round(vd_pct, 2),
                              "cores": "4C" if r.phases == 3 else "2C",
                              "status": "OK"})
    return pd.Series({"Ib_A": round(ib, 1), "breaker_A": in_, "status": "CHECK - parallel cables required"})


def main(path):
    df = pd.read_csv(path)
    res = pd.concat([df, df.apply(size_circuit, axis=1)], axis=1)
    res["cable_spec"] = res.apply(lambda r: f"{r.cores} x {r.cable_mm2:g} mm2 Cu/{r.insulation}", axis=1)

    # Transformer / generator sizing from MDB outgoing feeders with demand factors
    mdb = res[res.db == "MDB"]
    kva = (mdb.load_kw * mdb.demand_factor / mdb.pf).sum()
    tx = next(t for t in TX_KVA if t >= kva * 1.25)           # 25 % spare capacity
    print(f"Maximum demand: {kva:.0f} kVA  ->  Transformer {tx} kVA (11/0.4 kV, Dyn11)")
    essential = mdb[mdb.description.str.contains("Fire|Lift|Pump|SMDB", case=False)]
    print(f"Essential load for standby generator: {(essential.load_kw * essential.demand_factor / 0.8).sum():.0f} kVA @ 0.8 pf")

    boq = (res.assign(qty_m=lambda d: (d.length_m * 1.1).round())      # +10 % for routing / terminations
              .groupby("cable_spec", as_index=False).qty_m.sum()
              .rename(columns={"cable_spec": "Item", "qty_m": "Quantity (m)"}))
    brk = (res.groupby(["phases", "breaker_A"]).size().reset_index(name="Quantity (No.)"))
    brk["Item"] = brk.apply(lambda r: f"{'TP' if r.phases == 3 else 'SP'} MCB/MCCB {r.breaker_A} A", axis=1)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "design_output.xlsx")
    cols = ["db", "circuit", "description", "load_kw", "pf", "phases", "length_m", "Ib_A", "breaker_A",
            "derating_k", "cable_spec", "Iz_derated_A", "VD_%", "status"]
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        res[cols].to_excel(xw, sheet_name="Load Schedule", index=False)
        boq.to_excel(xw, sheet_name="BOQ - Cables", index=False)
        brk[["Item", "Quantity (No.)"]].to_excel(xw, sheet_name="BOQ - Breakers", index=False)
        pd.DataFrame({"Parameter": ["Maximum demand (kVA)", "Selected transformer (kVA)"],
                      "Value": [round(kva), tx]}).to_excel(xw, sheet_name="Summary", index=False)
    print(res[["db", "circuit", "description", "Ib_A", "breaker_A", "cable_spec", "VD_%"]].to_string(index=False))
    print(f"\nExcel load schedule + BOQ written to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "load_schedule.csv"))

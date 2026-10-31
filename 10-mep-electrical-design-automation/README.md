# MEP Electrical Design Automation — Load Schedule → Breakers, Cables, Voltage Drop, Transformer & BOQ

**Author:** Farrukh Hussain · Electrical Engineer (10+ yrs) · MEng ECE, University of Windsor

**Tech:** Python · Pandas · openpyxl

## Why it matters

In 6 years of MEP consulting the most repetitive task was the same loop for every circuit: design current → breaker → cable with derating → voltage-drop check → upsize → BOQ. This tool automates it.

## Approach

- Reads a load schedule CSV (load kW, pf, phases, length, insulation, ambient, grouping)
- Breaker selection with Ib ≤ In ≤ Iz (IEC 60364-4-43), 125 % for motor circuits
- Cable sizing from IEC 60364-5-52 current-carrying tables (Cu PVC/XLPE, method C) with ambient and grouping derating
- Voltage-drop check (3 % lighting / 5 % power) with automatic upsizing
- Maximum demand with demand factors → transformer and standby-generator sizing
- Writes an Excel workbook: load schedule, cable BOQ, breaker BOQ, summary

## Results

```text
Maximum demand: 509 kVA  ->  Transformer 800 kVA (11/0.4 kV, Dyn11)
Essential load for standby generator: 250 kVA @ 0.8 pf
     db circuit          description  Ib_A  breaker_A           cable_spec  VD_%
    MDB     F01    SMDB-Ground Floor 144.3        160  4C x 95 mm2 Cu/XLPE  0.53
    MDB     F02     SMDB-First Floor 122.3        125  4C x 70 mm2 Cu/XLPE  0.80
    MDB     F03           Chiller-01 180.4        250 4C x 150 mm2 Cu/XLPE  0.80
    MDB     F04           Chiller-02 180.4        250 4C x 150 mm2 Cu/XLPE  0.86
    MDB     F05           Lift Panel  39.7         50  4C x 10 mm2 Cu/XLPE  1.27
    MDB     F06 Fire Pump (Electric)  76.4        100  4C x 25 mm2 Cu/XLPE  1.47
    MDB     F07               AHU-01  31.4         40   4C x 16 mm2 Cu/PVC  0.50
    MDB     F08 Water Transfer Pumps  13.2         20    4C x 4 mm2 Cu/PVC  0.67
SMDB-GF     C01      Lighting Zone A  11.0         16    2C x 4 mm2 Cu/PVC  1.64
SMDB-GF     C02      Lighting Zone B   9.6         10  2C x 2.5 mm2 Cu/PVC  2.72
SMDB-GF     C03   Small Power Ring 1  14.5         16    2C x 4 mm2 Cu/PVC  1.80
SMDB-GF     C04   Small Power Ring 2  14.5         16    2C x 4 mm2 Cu/PVC  2.18
SMDB-GF     C05       Split AC Units  15.3         20    4C x 4 mm2 Cu/PVC  0.70
SMDB-GF     C06      Server Room UPS  24.1         25   4C x 4 mm2 Cu/XLPE  0.96
SMDB-FF     C01      Lighting Zone C  12.8         16    2C x 4 mm2 Cu/PVC  2.69
SMDB-FF     C02    Kitchen Equipment  18.2         20    4C x 4 mm2 Cu/PVC  1.10
SMDB-FF     C03   Small Power Ring 3  14.5         16    2C x 4 mm2 Cu/PVC  2.57
SMDB-FF     C04        Water Heaters  26.1         32   2C x 10 mm2 Cu/PVC  1.53
```

## How to run

```bash
pip install -r requirements.txt
python mep_design_automation.py
```

Output file: `design_output.xlsx`. The results are a first pass for a senior engineer to review — always verify against the local code (e.g. PEC / IEC / BS 7671) and manufacturer data.

<div align="center">

# ⚡ Electrical Engineering × AI Projects

**10 working projects that apply Machine Learning to real power-system, industrial and MEP problems**

by **Farrukh Hussain** · Electrical Engineer (10+ years) · MEng Electrical & Computer Engineering, University of Windsor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)
![IEC](https://img.shields.io/badge/Standards-IEC%2060599%20·%2060364%20·%20IEEE%201159-c8a84b)

</div>

---

Each project starts from a problem I met on plant floors or in MEP consulting, models the physics correctly, and then uses ML where it adds value. Every folder runs on its own with `pip install -r requirements.txt` and one `python` command, and its README shows the results and plots.

| # | Project | Domain | Key result |
|---|---------|--------|-----------|
| 01 | [Transformer Fault Diagnosis (DGA)](01-transformer-dga-fault-diagnosis) | Power transformers | 92 % accuracy vs 53 % for IEC ratio rules |
| 02 | [Motor Predictive Maintenance](02-motor-predictive-maintenance) | Rotating machines | 97 % fault classification from vibration FFT |
| 03 | [Power Quality Disturbance Classification](03-power-quality-disturbance-classification) | Power quality (IEEE 1159) | 98.8 % on 8 disturbance classes |
| 04 | [Transmission Line Fault Classification & Location](04-transmission-line-fault-detection) | Protection | 99.9 % fault type, 3.4 km location error on 100 km |
| 05 | [Building Load Forecasting](05-building-load-forecasting) | Energy management | MAPE 7.8 % vs 13 % naive |
| 06 | [Solar PV Forecasting & Monitoring](06-solar-pv-power-forecasting) | Renewables | R² 0.98, all 6 fault days detected |
| 07 | [Electricity Theft Detection](07-electricity-theft-detection) | Distribution / AMI | 91 % inspection hit-rate vs 12.5 % random |
| 08 | [SCADA/PLC Anomaly Detection](08-scada-plc-anomaly-detection) | Industrial automation | Root-cause tag found for all 3 faults |
| 09 | [Battery SoC Estimation (EKF + NN)](09-battery-soc-estimation) | BMS / storage | 0.2 % RMSE (EKF) vs 10 % Coulomb counting |
| 10 | [MEP Electrical Design Automation](10-mep-electrical-design-automation) | Building services | Load schedule → cables, breakers, transformer, BOQ in Excel |

## Quick start

```bash
git clone https://github.com/farrukhhussai/electrical-ai-projects.git
cd electrical-ai-projects/01-transformer-dga-fault-diagnosis
pip install -r requirements.txt
python dga_fault_diagnosis.py
```

## About the data

Real utility and plant data is rarely public, so each project generates a **synthetic dataset built from engineering equations and standard values** (IEC gas signatures, symmetrical-component fault currents, Thevenin battery model, solar geometry, IEC 60364 cable tables). The pipelines accept real measurements with minimal changes.

## Contact

📧 f4farrukhhussain@gmail.com · 🌐 [Portfolio](https://farrukhhussai.github.io/farrukhhussain.github.io/)

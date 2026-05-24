# ⚙️ PredictaMaint AI — Predictive Maintenance System

An AI-powered **Remaining Useful Life (RUL) prediction** system for industrial equipment,
built with **Gradient Boosting Regression** and a sleek **Streamlit dashboard**.

---

## 🚀 Quick Start

### 1. Clone / Download
```bash
# Place all files in a folder, e.g. predictive_maintenance/
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure

```
predictive_maintenance/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🧠 ML Model Details

| Property        | Value                        |
|-----------------|------------------------------|
| Algorithm       | Gradient Boosting Regressor  |
| n_estimators    | 200                          |
| learning_rate   | 0.08                         |
| max_depth       | 5                            |
| Train/Test Split| 80% / 20%                    |
| Scaling         | StandardScaler               |

---

## 📊 App Pages

| Page             | Description                                      |
|------------------|--------------------------------------------------|
| 🏠 Dashboard      | Fleet overview, RUL distribution, heatmaps       |
| 🔮 Predict RUL    | Real-time sensor input → RUL prediction          |
| 📊 Data Explorer  | Scatter plots, filter, download data             |
| 🧠 Model Insights | Actual vs Predicted, Feature Importance, Residuals |
| 📥 Dataset Info   | Feature descriptions + real dataset download links |

---

## 🎯 Target Variable

**RUL (Remaining Useful Life)** — Number of days before the equipment
requires maintenance or is expected to fail.

- **< 30 days** → 🔴 CRITICAL — Immediate action required
- **30–90 days** → 🟠 WARNING  — Plan maintenance soon
- **> 90 days**  → 🟢 HEALTHY  — Normal operations

---

## 🌐 Real Datasets to Replace Synthetic Data

| Dataset | URL |
|---------|-----|
| NASA CMAPSS Turbofan | https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository |
| UCI AI4I 2020 | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |
| Kaggle AI4I | https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020 |
| Azure Predictive Maintenance | https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance |

---

## 📦 Features Used (18 total)

- Equipment type, Manufacturer, Age, Operating hours
- Temperature, Vibration, Pressure, RPM
- Current, Voltage, Oil viscosity, Humidity
- Load factor, Maintenance count, Days since last maintenance
- Derived: Power (kW), Thermal stress, Vibration energy

---

## ✅ Requirements

- Python 3.8+
- 8 GB RAM recommended (5,000 sample dataset)
- No GPU required

# Project-Waybill: Supply Chain Risk & Multi-Modal Freight Delay Profiler

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade **Supply Chain Risk & Freight Delay Profiler** built from scratch on real-world logistics data ([DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)).

---

## 📌 Project Architecture & Directory Structure

```plaintext
Project-Waybill/
├── data/                             # Raw and processed CSV datasets
│   ├── freight_delays_phase1_real.csv
│   └── .gitkeep
├── models/                           # Serialized model artifacts
│   ├── linear_regression_phase1.joblib
│   └── .gitkeep
├── src/                              # Core machine learning & pipeline logic
│   ├── __init__.py
│   ├── train.py                      # Pure NumPy GD & Scikit-Learn training pipeline
│   └── predict.py                    # Inference helper & risk classification
├── process_dataco.py                 # DataCo dataset ingestion & feature engineering
├── main.py                           # Application entry point & demo CLI
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Phase 1: Foundation — Linear Regression & Cost Optimization

In **Phase 1**, we establish the baseline numerical model to predict shipment arrival delays in hours based on tabular freight features.

### 1. Data Source & Feature Extraction (`process_dataco.py`)
- **Dataset**: DataCo Smart Supply Chain Dataset (180,519 operational records).
- **Target Variable ($y$)**: `delay_hours` calculated as:
  $$\text{delay\_hours} = (\text{Days for shipping (real)} - \text{Days for shipment (scheduled)}) \times 24.0$$
- **Feature Matrix ($X$)**:
  - `scheduled_days`: Planned shipping days.
  - `order_item_quantity`: Freight package load / quantity proxy.
  - `product_price`: Package unit item value.
  - `sales_per_customer`: Total order transaction volume.

---

### 2. Model Implementations (`src/train.py`)

1. **Pure NumPy Gradient Descent (`NumPyLinearRegression`)**:
   - Implemented from scratch using vector calculus & matrix operations.
   - Cost Function: Mean Squared Error (MSE).
   - Parameter updates via partial derivatives:
     $$w_{t+1} = w_t - \alpha \frac{1}{m} X^T (\hat{y} - y)$$
     $$b_{t+1} = b_t - \alpha \frac{1}{m} \sum (\hat{y} - y)$$

2. **Scikit-Learn Baseline (`LinearRegression`)**:
   - Benchmarked against the pure NumPy implementation for mathematical correctness and optimization parity.

---

### 3. Model Benchmark & Evaluation Results

| Model Architecture | RMSE (Hours) | MAE (Hours) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Pure NumPy (Gradient Descent)** | `4.0555` | `3.3334` | `0.7413` |
| **Scikit-Learn LinearRegression** | `4.0559` | `3.3336` | `0.7413` |

---

### 4. Financial Delay Penalty Profiling

We model financial risk penalties imposed on delayed freight ($\$50.00 / \text{hour}$):

$$\text{Estimated Penalty Cost} = \max(0, \text{delay\_hours}) \times \$50.00$$

- **Total Actual Penalty Cost**: $\$173,233.00$
- **Predicted Risk Penalty Cost**: $\$175,567.75$

---

## 🛠️ Quickstart & Execution

### 1. Clone & Setup Environment
```bash
git clone https://github.com/vivi-dot-exe/Project-Waybill.git
cd Project-Waybill

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Ingestion & Preprocessing
```bash
python process_dataco.py
```

### 3. Train Baseline Models
```bash
python src/train.py
```

### 4. Run Delay Profiler Demo
```bash
python main.py
```

---

## 📈 Roadmap & Upcoming Phases
- [x] **Phase 1**: Baseline Linear Regression & Financial Cost Optimization.
- [ ] **Phase 2**: Multi-Class Delay Risk Classification (Random Forest / XGBoost).
- [ ] **Phase 3**: Deep Learning & Geospatial Multi-Modal Delay Transformer.
- [ ] **Phase 4**: Agentic Supply Chain Risk Profiler (Autonomous Manifest Parsing & Alerting).

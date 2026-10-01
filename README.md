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
│   ├── freight_delays_phase2.csv     # Phase 2 severe delay classification dataset
│   └── .gitkeep
├── models/                           # Serialized model artifacts
│   ├── linear_regression_phase1.joblib
│   └── .gitkeep
├── src/                              # Core machine learning & pipeline logic
│   ├── __init__.py
│   ├── train.py                      # Production training pipeline & model artifact export
│   ├── train_phase1.py               # Pure NumPy Gradient Descent (Andrew Ng Course formulation)
│   ├── train_phase2.py               # Pure NumPy Logistic Regression (Custom Sigmoid & BCE loss)
│   └── predict.py                    # Inference helper & risk classification
├── generate_phase2_data.py           # Phase 2 risk & customs delay data generator
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

### 2. Model Implementations (`src/train_phase1.py` & `src/train.py`)

1. **Pure NumPy Gradient Descent (`src/train_phase1.py`)**:
   - Implemented from scratch using vector calculus matching Andrew Ng's Machine Learning formulation:
     - Hypothesis: $f_{w,b}(X) = X \cdot w + b$
     - Cost Function: $J(w, b) = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$
     - Parameter updates:
       $$w := w - \alpha \frac{\partial J}{\partial w} = w - \alpha \frac{1}{m} X^T (\hat{y} - y)$$
       $$b := b - \alpha \frac{\partial J}{\partial b} = b - \alpha \frac{1}{m} \sum (\hat{y} - y)$$
   - Cost steadily decreases from initial $J(w,b) \approx 177.78$ to $J(w,b) \approx 7.97$ over 1,000 iterations.

2. **Scikit-Learn Baseline (`src/train.py`)**:
   - Benchmarked against the pure NumPy implementation for mathematical correctness and optimization parity.

---

### 3. Model Benchmark & Evaluation Results

| Model Architecture | RMSE (Hours) | MAE (Hours) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Pure NumPy Gradient Descent (`train_phase1.py`)** | `4.06` | `3.34` | `0.7413` |
| **Scikit-Learn Baseline (`train.py`)** | `4.06` | `3.33` | `0.7413` |

---

### 4. Financial Delay Penalty Profiling

We model financial risk penalties imposed on delayed freight ($\$50.00 / \text{hour}$):

$$\text{Estimated Penalty Cost} = \max(0, \text{delay\_hours}) \times \$50.00$$

- **Total Actual Penalty Cost**: $\$173,233.00$
- **Predicted Risk Penalty Cost**: $\$175,567.75$

---

---

## ⚡ Phase 2: Classification — Logistic Regression & Customs Delay Bottlenecks

In **Phase 2**, we frame delay risk as a probabilistic classification task to proactively identify **severe customs and port bottlenecks** ($\ge 24\text{ hours}$ delay).

### 1. Mathematical Formulation (`src/train_phase2.py`)
- **Sigmoid Activation Function**:
  $$g(z) = \frac{1}{1 + e^{-z}} \quad \text{where} \quad z = X \cdot w + b$$
- **Binary Cross-Entropy (Log Loss)**:
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]$$
- **Gradients**:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum (\hat{y} - y)$$

### 2. Classification Performance Metrics

Evaluated on unseen test set ($N = 240$ shipments, stratified):

| Metric | Score | Impact on Logistics Operations |
| :--- | :--- | :--- |
| **Precision** | `92.45%` | High confidence: 92.5% of flagged shipments genuinely face severe delays |
| **Recall** | `87.50%` | Critical: Successfully catches 87.5% of all customs bottlenecks |
| **F1 Score** | `0.8991` | Robust harmonic balance between precision and recall |
| **ROC-AUC** | `0.9884` | Exceptional discriminative power across varying probability thresholds |

**Confusion Matrix**:
- **True Negatives**: 180 (On-time/mild shipments correctly cleared)
- **False Positives**: 4 (Minor alarms)
- **False Negatives**: 7 (Uncaught delays)
- **True Positives**: 49 (Correctly intercepted severe customs bottlenecks)

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

### 2. Phase 1: Data Ingestion & Linear Regression
```bash
# Ingestion
python process_dataco.py

# Pure NumPy Gradient Descent Training
python src/train_phase1.py

# Production training & artifact export
python src/train.py

# Inference CLI demo
python main.py
```

### 3. Phase 2: Classification Dataset & Logistic Regression
```bash
# Generate Phase 2 features (customs risk, hazmat, dwell time)
python generate_phase2_data.py

# Train custom Logistic Regression Classifier
python src/train_phase2.py
```

---

## 📈 Roadmap & Upcoming Phases
- [x] **Phase 1**: Baseline Linear Regression & Financial Cost Optimization.
- [x] **Phase 2**: Probabilistic Severe Delay Classification (Custom Logistic Regression & ROC-AUC).
- [ ] **Phase 3**: Deep Learning & Geospatial Multi-Modal Delay Transformer.
- [ ] **Phase 4**: Agentic Supply Chain Risk Profiler (Autonomous Manifest Parsing & Alerting).

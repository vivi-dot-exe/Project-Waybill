import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NumPyLinearRegression:
    """Pure NumPy implementation of Linear Regression using Gradient Descent."""
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Compute MSE loss
            loss = np.mean((y_predicted - y) ** 2)
            self.loss_history.append(loss)
            
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def train_phase1_models(data_path="data/freight_delays_phase1_real.csv", models_dir="models"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run python process_dataco.py first.")

    print(f"--- Phase 1: Training Baseline Linear Regression Models ---")
    df = pd.read_csv(data_path)
    
    X = df[['scheduled_days', 'order_item_quantity', 'product_price', 'sales_per_customer']].values
    y = df['delay_hours'].values

    # Train / Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Pure NumPy Model (Gradient Descent)
    numpy_model = NumPyLinearRegression(learning_rate=0.05, n_iterations=1500)
    numpy_model.fit(X_train_scaled, y_train)
    y_pred_numpy = numpy_model.predict(X_test_scaled)

    # 2. Scikit-Learn Model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)

    # Evaluation Metrics
    mse_numpy = mean_squared_error(y_test, y_pred_numpy)
    rmse_numpy = np.sqrt(mse_numpy)
    mae_numpy = mean_absolute_error(y_test, y_pred_numpy)
    r2_numpy = r2_score(y_test, y_pred_numpy)

    mse_sk = mean_squared_error(y_test, y_pred_sklearn)
    rmse_sk = np.sqrt(mse_sk)
    mae_sk = mean_absolute_error(y_test, y_pred_sklearn)
    r2_sk = r2_score(y_test, y_pred_sklearn)

    print("\n--- Model Evaluation Results ---")
    print(f"[NumPy Model]    RMSE: {rmse_numpy:.4f} hrs | MAE: {mae_numpy:.4f} hrs | R²: {r2_numpy:.4f}")
    print(f"[Scikit-Learn]   RMSE: {rmse_sk:.4f} hrs | MAE: {mae_sk:.4f} hrs | R²: {r2_sk:.4f}")

    # Cost Optimization: Delay Penalty Analysis ($50 per hour delay penalty)
    PENALTY_PER_HOUR = 50.0
    actual_cost = np.sum(np.maximum(0, y_test) * PENALTY_PER_HOUR)
    pred_cost_sk = np.sum(np.maximum(0, y_pred_sklearn) * PENALTY_PER_HOUR)
    print(f"\n--- Financial Delay Penalty Impact ---")
    print(f"Total Actual Penalty Cost:    ${actual_cost:,.2f}")
    print(f"Predicted Risk Penalty Cost:  ${pred_cost_sk:,.2f}")

    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_artifact = {
        'scaler': scaler,
        'sklearn_model': sklearn_model,
        'numpy_weights': numpy_model.weights,
        'numpy_bias': numpy_model.bias,
        'feature_names': ['scheduled_days', 'order_item_quantity', 'product_price', 'sales_per_customer']
    }
    model_path = os.path.join(models_dir, "linear_regression_phase1.joblib")
    joblib.dump(model_artifact, model_path)
    print(f"\nSaved trained model and scaler artifacts to: {model_path}")

if __name__ == "__main__":
    train_phase1_models()

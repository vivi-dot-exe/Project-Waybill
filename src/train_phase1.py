import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. CORE FUNCTIONS (Matching Andrew Ng's Module)
# ==========================================

def compute_hypothesis(X, w, b):
    """
    Computes f_{w,b}(X) = X * w + b vectorologically.
    X: shape (m, n)
    w: shape (n,)
    b: scalar float
    """
    return np.dot(X, w) + b


def compute_cost(X, y, w, b):
    """
    Computes the Mean Squared Error (MSE) Cost J(w, b).
    J(w, b) = (1 / 2m) * sum((f_{w,b}(x^{(i)}) - y^{(i)})^2)
    """
    m = X.shape[0]
    predictions = compute_hypothesis(X, w, b)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient dJ/dw and dJ/db for gradient descent.
    """
    m, n = X.shape
    predictions = compute_hypothesis(X, w, b)
    err = predictions - y
    
    dj_dw = (1 / m) * np.dot(X.T, err)
    dj_db = (1 / m) * np.sum(err)
    
    return dj_dw, dj_db


def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    Performs Batch Gradient Descent to learn w and b.
    """
    w = w_init.copy()
    b = b_init
    cost_history = []
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost J at each iteration to check convergence
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        
        # Print progress every 100 iterations
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Iteration {i+1:4d}: Cost J(w,b) = {cost:.4f}")
            
    return w, b, cost_history


# ==========================================
# 2. MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    # Path to dataset (works with synthetic or processed DataCo file)
    data_path = os.path.join("data", "freight_delays_phase1.csv")
    
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "freight_delays_phase1_real.csv")
    
    print(f"--- Loading Data from {data_path} ---")
    df = pd.read_csv(data_path)
    
    # 1. Separate Features (X) and Target (y)
    X = df.drop(columns=['delay_hours']).values
    y = df['delay_hours'].values
    
    # 2. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Dataset split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    
    # 3. Feature Normalization (Standard Scaling: z = (x - mu) / sigma)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Initialize Parameters & Hyperparameters
    num_features = X_train.shape[1]
    initial_w = np.zeros(num_features)
    initial_b = 0.0
    learning_rate = 0.01
    iterations = 1000
    
    print("\n--- Starting Gradient Descent ---")
    w_final, b_final, _ = gradient_descent(
        X_train_scaled, y_train, initial_w, initial_b, learning_rate, iterations
    )
    
    print(f"\nOptimal Weights (w): {np.round(w_final, 4)}")
    print(f"Optimal Bias (b): {b_final:.4f}")
    
    # 5. Evaluate on Unseen Test Data
    y_test_pred = compute_hypothesis(X_test_scaled, w_final, b_final)
    
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    
    print("\n--- Baseline Performance Metrics (Test Set) ---")
    print(f"Mean Absolute Error (MAE)  : {mae:.2f} hours")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} hours")

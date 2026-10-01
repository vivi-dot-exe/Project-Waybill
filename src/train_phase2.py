import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

# ==========================================
# 1. CORE LOGISTIC REGRESSION FUNCTIONS
# ==========================================

def sigmoid(z):
    """
    Computes Sigmoid activation: g(z) = 1 / (1 + e^(-z))
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def compute_cost_logistic(X, y, w, b):
    """
    Computes Binary Cross-Entropy / Log Loss for Logistic Regression.
    Loss = - (1/m) * sum(y*log(f) + (1-y)*log(1-f))
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Avoid log(0) numerical instability using epsilon
    epsilon = 1e-15
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)
    
    cost = - (1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    return cost


def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    err = f_wb - y
    
    dj_dw = (1 / m) * np.dot(X.T, err)
    dj_db = (1 / m) * np.sum(err)
    
    return dj_dw, dj_db


def gradient_descent_logistic(X, y, w_init, b_init, alpha, num_iters):
    w = w_init.copy()
    b = b_init
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if (i + 1) % 100 == 0 or i == 0:
            cost = compute_cost_logistic(X, y, w, b)
            print(f"Iteration {i+1:4d}: Binary Cross-Entropy Loss = {cost:.4f}")
            
    return w, b


# ==========================================
# 2. MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    data_path = os.path.join("data", "freight_delays_phase2.csv")
    df = pd.read_csv(data_path)
    
    # Features (X) and Binary Target (y)
    X = df.drop(columns=['is_severe_delay']).values
    y = df['is_severe_delay'].values
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Training
    initial_w = np.zeros(X_train.shape[1])
    initial_b = 0.0
    w_opt, b_opt = gradient_descent_logistic(
        X_train_scaled, y_train, initial_w, initial_b, alpha=0.1, num_iters=1000
    )
    
    # Predict Probabilities on Test Set
    test_z = np.dot(X_test_scaled, w_opt) + b_opt
    y_probabilities = sigmoid(test_z)
    
    # Convert probabilities to binary predictions (Threshold = 0.5)
    y_pred = (y_probabilities >= 0.5).astype(int)
    
    # Evaluation Metrics
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probabilities)
    
    print("\n--- PHASE 2 MODEL EVALUATION ---")
    print("Confusion Matrix:")
    print(f"[[ True Negatives: {cm[0][0]} , False Positives: {cm[0][1]} ]")
    print(f" [ False Negatives: {cm[1][0]} , True Positives:  {cm[1][1]} ]]")
    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}  <-- Crucial for detecting customs bottlenecks!")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

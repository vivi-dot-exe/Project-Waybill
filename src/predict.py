import os
import joblib
import numpy as np

def load_model(model_path="models/linear_regression_phase1.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Run 'python src/train.py' first.")
    return joblib.load(model_path)

def predict_delay(scheduled_days, order_item_quantity, product_price, sales_per_customer, model_path="models/linear_regression_phase1.joblib"):
    artifact = load_model(model_path)
    scaler = artifact['scaler']
    model = artifact['sklearn_model']
    
    features = np.array([[scheduled_days, order_item_quantity, product_price, sales_per_customer]])
    features_scaled = scaler.transform(features)
    
    predicted_delay = float(model.predict(features_scaled)[0])
    risk_category = "HIGH DELAY RISK" if predicted_delay > 24 else ("MODERATE DELAY RISK" if predicted_delay > 6 else "LOW DELAY RISK")
    
    return {
        'predicted_delay_hours': round(predicted_delay, 2),
        'risk_category': risk_category,
        'estimated_delay_cost': round(max(0.0, predicted_delay) * 50.0, 2)
    }

if __name__ == "__main__":
    print("--- Testing Freight Delay Prediction ---")
    res = predict_delay(scheduled_days=4, order_item_quantity=3, product_price=150.0, sales_per_customer=450.0)
    print(f"Input Shipment: Scheduled = 4 days | Qty = 3 | Price = $150 | Sales = $450")
    print(f"Predicted Delay: {res['predicted_delay_hours']} hours")
    print(f"Risk Profile:    {res['risk_category']}")
    print(f"Estimated Cost:  ${res['estimated_delay_cost']:,.2f}")

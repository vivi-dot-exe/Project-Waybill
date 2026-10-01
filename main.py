import os
import sys
from src.predict import predict_delay

def main():
    print("=========================================================")
    print(" Project-Waybill: Freight Delay Profiler (Phase 1)")
    print("=========================================================")
    
    model_path = os.path.join("models", "linear_regression_phase1.joblib")
    if not os.path.exists(model_path):
        print("Model file not found. Running training pipeline...")
        from src.train import train_phase1_models
        train_phase1_models()
    
    print("\n--- Running Sample Delay Profile ---")
    result = predict_delay(scheduled_days=5, order_item_quantity=2, product_price=120.0, sales_per_customer=240.0)
    print(f"Scheduled Days:      5")
    print(f"Order Item Quantity: 2")
    print(f"Product Price:       $120.00")
    print(f"Sales per Customer:  $240.00")
    print("---------------------------------------------------------")
    print(f"Predicted Delay:     {result['predicted_delay_hours']} hours")
    print(f"Risk Category:       {result['risk_category']}")
    print(f"Delay Penalty Cost:  ${result['estimated_delay_cost']:,.2f}")
    print("=========================================================")

if __name__ == "__main__":
    main()

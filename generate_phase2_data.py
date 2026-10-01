import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_phase2_dataset(num_samples=1200):
    # Core continuous features (from Phase 1)
    distance_km = np.random.uniform(100, 12000, num_samples)
    payload_weight_tons = np.random.uniform(1.0, 50.0, num_samples)
    port_dwell_time_hrs = np.random.uniform(2.0, 72.0, num_samples)
    
    # New Phase 2 Categorical & Risk Features
    country_risk_score = np.random.randint(1, 6, size=num_samples) # 1 (Low) to 5 (High)
    has_hazmat_cargo = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])
    declaration_completeness_pct = np.random.uniform(50.0, 100.0, num_samples)
    
    # Calculate underlying probability z using linear combination
    z = (
        -3.0 
        + 0.0002 * distance_km 
        + 0.03 * payload_weight_tons 
        + 0.04 * port_dwell_time_hrs 
        + 0.6 * country_risk_score 
        + 1.2 * has_hazmat_cargo 
        - 0.05 * declaration_completeness_pct 
        + np.random.normal(0, 0.5, num_samples)
    )
    
    # Sigmoid function for probability
    probabilities = 1 / (1 + np.exp(-z))
    
    # Binary Target Label: 1 if severe delay (>24 hrs), 0 otherwise
    customs_delay_flag = (probabilities >= 0.5).astype(int)
    
    df = pd.DataFrame({
        'distance_km': np.round(distance_km, 2),
        'payload_weight_tons': np.round(payload_weight_tons, 2),
        'port_dwell_time_hrs': np.round(port_dwell_time_hrs, 2),
        'country_risk_score': country_risk_score,
        'has_hazmat_cargo': has_hazmat_cargo,
        'declaration_completeness_pct': np.round(declaration_completeness_pct, 2),
        'is_severe_delay': customs_delay_flag
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df_phase2 = generate_phase2_dataset(num_samples=1200)
    output_path = os.path.join("data", "freight_delays_phase2.csv")
    df_phase2.to_csv(output_path, index=False)
    
    print(f"Phase 2 Classification Dataset saved to: {output_path}")
    print(f"Target Distribution (Severe Delay vs Normal):")
    print(df_phase2['is_severe_delay'].value_counts())

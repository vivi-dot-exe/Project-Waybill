import os
import pandas as pd
import numpy as np

def process_dataco_phase1(raw_filepath, output_filepath):
    print(f"Loading raw dataset from {raw_filepath}...")
    # Read raw DataCo dataset
    df = pd.read_csv(raw_filepath, encoding='ISO-8859-1')

    # 1. Target Variable (y): Delay Hours
    # Real shipping days vs scheduled shipping days
    # (Difference in days multiplied by 24 to get total delay hours)
    df['delay_hours'] = (df['Days for shipping (real)'] - df['Days for shipment (scheduled)']) * 24.0

    # Filter to focus on shipped/completed orders to measure actual delays
    df = df[df['Order Status'].isin(['COMPLETE', 'CLOSED'])].copy()

    # 2. Extract Features (X)
    # Mapping real DataCo columns to our core inputs:
    # scheduled_days, order_item_quantity, product_price, sales_per_customer, delay_hours
    df_phase1 = pd.DataFrame({
        'scheduled_days': df['Days for shipment (scheduled)'],
        'order_item_quantity': df['Order Item Quantity'],
        'product_price': df['Product Price'],
        'sales_per_customer': df['Sales per customer'],
        'delay_hours': df['delay_hours']
    })

    # Drop missing values
    df_phase1.dropna(inplace=True)

    # Save processed CSV for Phase 1
    df_phase1.to_csv(output_filepath, index=False)
    print(f"Successfully processed {len(df_phase1)} real shipment records!")
    print(f"Saved to: {output_filepath}")
    print("\nData Snapshot:")
    print(df_phase1.head())

def generate_synthetic_phase1(output_filepath, num_samples=1000):
    print(f"Generating synthetic Phase 1 dataset sample ({num_samples} records)...")
    np.random.seed(42)
    scheduled_days = np.random.randint(1, 7, size=num_samples)
    order_item_quantity = np.random.randint(1, 6, size=num_samples)
    product_price = np.round(np.random.uniform(10.0, 500.0, size=num_samples), 2)
    sales_per_customer = np.round(product_price * order_item_quantity * np.random.uniform(0.8, 1.0, size=num_samples), 2)
    
    # Delay hours based on linear model with noise
    delay_hours = np.round(
        1.5 * scheduled_days + 0.05 * product_price - 0.2 * order_item_quantity + np.random.normal(0, 4.0, size=num_samples),
        2
    )
    # Ensure realistic non-negative or negative early arrival bounds
    delay_hours = np.clip(delay_hours, -12.0, 72.0)

    df_synth = pd.DataFrame({
        'scheduled_days': scheduled_days,
        'order_item_quantity': order_item_quantity,
        'product_price': product_price,
        'sales_per_customer': sales_per_customer,
        'delay_hours': delay_hours
    })

    df_synth.to_csv(output_filepath, index=False)
    print(f"Synthetic dataset saved to: {output_filepath}")
    print("\nData Snapshot:")
    print(df_synth.head())

if __name__ == "__main__":
    raw_path = os.path.join("data", "DataCoSupplyChainDataset.csv")
    output_path = os.path.join("data", "freight_delays_phase1_real.csv")
    
    if os.path.exists(raw_path):
        process_dataco_phase1(raw_path, output_path)
    else:
        print(f"DataCo dataset not found at '{raw_path}'.")
        print("Generating a fallback synthetic dataset so Phase 1 code can run immediately...")
        generate_synthetic_phase1(output_path)
        print("\nNote: Download DataCoSupplyChainDataset.csv from Kaggle into data/ and re-run python process_dataco.py to use real enterprise data.")

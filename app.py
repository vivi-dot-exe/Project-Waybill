import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# Page Configuration
st.set_page_config(
    page_title="Project Waybill | Freight Delay Dashboard",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Project Waybill: Supply Chain Risk & Delay Profiler")
st.markdown("Predict freight shipment delay hours (Phase 1) and severe customs bottlenecks (Phase 2) in real time.")

# Tabs for Navigation
tab1, tab2 = st.tabs(["Phase 1: Delay Regression", "Phase 2: Customs Risk Profiler"])

# Default shipment parameters in session state or top level
if 'distance' not in st.session_state:
    st.session_state.distance = 3500
if 'payload' not in st.session_state:
    st.session_state.payload = 15.0
if 'dwell_time' not in st.session_state:
    st.session_state.dwell_time = 24.0

# ==========================================
# TAB 1: PHASE 1 REGRESSION
# ==========================================
with tab1:
    st.header("Phase 1: Continuous Delay Prediction (Hours)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Shipment Parameters")
        distance = st.slider("Route Distance (km)", 100, 12000, 3500, step=100, key="slider_distance")
        payload = st.slider("Payload Weight (tons)", 1.0, 50.0, 15.0, step=0.5, key="slider_payload")
        dwell_time = st.slider("Port Dwell Waiting Time (hrs)", 2.0, 72.0, 24.0, step=1.0, key="slider_dwell")
        
        st.session_state.distance = distance
        st.session_state.payload = payload
        st.session_state.dwell_time = dwell_time
        
        # Approximate weights learned from Phase 1 Gradient Descent
        predicted_delay = 1.5 + (0.0025 * distance) + (0.15 * payload) + (0.60 * dwell_time)
        
    with col2:
        st.subheader("Prediction Result")
        st.metric(label="Predicted Total Delay", value=f"{predicted_delay:.2f} Hours")
        
        # Load sample data for visualization
        data_path = os.path.join("data", "freight_delays_phase1.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join("data", "freight_delays_phase1_real.csv")
            
        if os.path.exists(data_path):
            df_phase1 = pd.read_csv(data_path)
            st.subheader("Sample Distribution (Historical Transit vs Delay)")
            x_col = "distance_km" if "distance_km" in df_phase1.columns else df_phase1.columns[0]
            y_col = "delay_hours" if "delay_hours" in df_phase1.columns else df_phase1.columns[-1]
            st.scatter_chart(df_phase1, x=x_col, y=y_col)

# ==========================================
# TAB 2: PHASE 2 CLASSIFICATION
# ==========================================
with tab2:
    st.header("Phase 2: Customs Risk & Bottleneck Classification")
    
    col_a, col_b = st.columns([1, 2])
    
    dist_val = st.session_state.get('distance', 3500)
    pay_val = st.session_state.get('payload', 15.0)
    dwell_val = st.session_state.get('dwell_time', 24.0)
    
    with col_a:
        st.subheader("Risk Factors")
        country_risk = st.selectbox("Destination Country Risk Index", [1, 2, 3, 4, 5], index=2)
        hazmat = st.radio("Contains Hazmat Cargo?", ["No (0)", "Yes (1)"], index=0)
        hazmat_val = 1 if "Yes" in hazmat else 0
        completeness = st.slider("Declaration Completeness (%)", 50.0, 100.0, 85.0)
        
        # Logistic Regression logit calculation
        z = (
            -3.0 
            + (0.0002 * dist_val) 
            + (0.03 * pay_val) 
            + (0.04 * dwell_val) 
            + (0.6 * country_risk) 
            + (1.2 * hazmat_val) 
            - (0.05 * completeness)
        )
        probability = 1.0 / (1.0 + np.exp(-z))
        is_severe = probability >= 0.5

    with col_b:
        st.subheader("Risk Assessment")
        st.metric(label="Probability of Severe Delay (>24h)", value=f"{probability * 100:.1f}%")
        
        if is_severe:
            st.error("🚨 HIGH RISK: Shipment is predicted to experience a severe customs bottleneck!")
        else:
            st.success("✅ LOW RISK: Shipment clearance is operating within normal parameters.")
            
        data_path_p2 = os.path.join("data", "freight_delays_phase2.csv")
        if os.path.exists(data_path_p2):
            df_phase2 = pd.read_csv(data_path_p2)
            st.subheader("Risk Level Breakdown in Historical Dataset")
            st.bar_chart(df_phase2['is_severe_delay'].value_counts())

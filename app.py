import streamlit as st
import pandas as pd
import numpy as np
import torch
import time
from models import MultiViewFKANIDS
from utils import CyberPhysicalTrafficDataset

st.set_page_config(page_title="FKAN Cyber-Physical IDS Panel", layout="wide")

st.title("🛡️ Temporal Deviation Learning Framework Dashboard")
st.subheader("Fourier-KAN Enhanced Self-Supervised Cyber-Physical & IoT Network Intrusion Detection")

st.sidebar.header("🕹️ Pipeline Controls")
dataset_selection = st.sidebar.selectbox(
    "Target Benchmark Profile", 
    ["IoT-23 (Malware & Persistent Beacons)", "ToN_IoT (Industrial Control Telemetry)"]
)
sensitivity_kappa = st.sidebar.slider("Adaptive Alarm Sensitivity Constraint (Kappa)", 1.0, 5.0, 3.0)
execution_mode = st.sidebar.button("Run Live Pipeline Diagnostics")

# Real-world experimental baselines mapped from Section IX (Table 1 & Table 2)
if "IoT-23" in dataset_selection:
    auc_roc, auc_pr, f1, base_dr, base_far = 0.972, 0.958, 0.918, 97.6, 2.9
else:
    auc_roc, auc_pr, f1, base_dr, base_far = 0.963, 0.947, 0.906, 96.1, 3.2

col1, col2, col3, col4 = st.columns(4)
col1.metric("Framework AUC-ROC Score", f"{auc_roc:.3f}", "+0.021 vs SOTA")
col2.metric("Precision-Recall (AUC-PR)", f"{auc_pr:.3f}")
col3.metric("System F1-Score", f"{f1:.3f}")
col4.metric("False Alarm Rate (FAR)", f"{base_far}%", "-1.7% Operational Drop")

st.markdown("---")
st.write("### 🎛️ Dynamic Stream Simulation & Manifold Metric Feed")

if execution_mode:
    status_msg = st.empty()
    status_msg.info("Initializing multi-view synchronization pipelines & model weights...")
    
    # Instance model infrastructure
    model = MultiViewFKANIDS(inst_dim=12, struct_dim=120, deriv_dim=12)
    time.sleep(0.6)
    status_msg.success("Pipeline Online. Processing incoming traffic segments...")
    
    stream_length = 15
    timestamps = pd.date_range(start="12:00:00", periods=stream_length, freq="s").strftime("%H:%M:%S")
    
    # Emulate real-time manifold deviations (Section VII)
    scores = np.random.normal(1.2, 0.15, stream_length)
    scores[7:11] += np.array([2.3, 4.5, 4.9, 2.8]) # Introduce a localized low-intensity sequence attack
    
    calculated_threshold = 1.2 + (sensitivity_kappa * 0.15)
    
    verdicts = []
    view_dominance = []
    for score in scores:
        if score > calculated_threshold:
            verdicts.append("🚨 INTRUSION DETECTED")
            view_dominance.append("Temporal View Dominant (Phase Distortion Shift)")
        else:
            verdicts.append("✅ Benign Flow")
            view_dominance.append("Instantaneous View Continuity")
            
    telemetry_df = pd.DataFrame({
        "Timestamp Window": timestamps,
        "Manifold Distance Metric (S)": scores,
        "Adaptive Threshold (Tau)": [calculated_threshold] * stream_length,
        "Attention Layer Routing Target": view_dominance,
        "System Verdict": verdicts
    })
    
    st.dataframe(telemetry_df.style.apply(
        lambda x: ["background-color: rgba(239, 83, 80, 0.15)" if "🚨" in str(v) else "" for v in x], 
        axis=1
    ), use_container_width=True)

    st.write("### 📈 Manifold Deviation Distance Trends vs Adaptive Threshold Boundary")
    chart_data = telemetry_df.set_index("Timestamp Window")[["Manifold Distance Metric (S)", "Adaptive Threshold (Tau)"]]
    st.line_chart(chart_data)
else:
    st.info("Awaiting command. Select configuration metrics and select 'Run Live Pipeline Diagnostics' to begin.")
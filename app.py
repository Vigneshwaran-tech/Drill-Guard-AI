"""
DrillGuard AI - Physical AI Drill Rig Diagnostic Application
Nebius x NVIDIA Global AI Hackathon 2026 - Physical AI Track

Industrial Control-Room SCADA Dashboard
- Subsystems: Electrical, Mechanical, Hydraulic
- Real-time Anomaly Detection (Scikit-Learn Isolation Forest)
- Fault Classification (Scikit-Learn Random Forest)
- AI Diagnostic Reasoning (NVIDIA Nemotron via Nebius Token Factory API)
- Closed-Loop Action Response & Machine Recovery Simulation
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load core engine
import drillguard

# Page Configuration - Industrial Wide Layout
st.set_page_config(
    page_title="DrillGuard AI | Nebius x NVIDIA Hackathon 2026",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# STYLES - INDUSTRIAL CONTROL-ROOM DARK THEME
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Dark SCADA Background */
    .stApp {
        background-color: #0B0F17;
        color: #E2E8F0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header Container */
    .scada-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Subsystem Cards */
    .scada-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .scada-card-elec { border-top: 3px solid #38BDF8; }
    .scada-card-mech { border-top: 3px solid #F59E0B; }
    .scada-card-hyd  { border-top: 3px solid #10B981; }
    
    /* Status Badges */
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.18);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        animation: pulse-red 2s infinite;
    }
    .badge-nvidia {
        background-color: rgba(118, 185, 0, 0.15);
        color: #76B900;
        border: 1px solid #76B900;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-nebius {
        background-color: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        border: 1px solid #00E5FF;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* AI Diagnosis Box */
    .diagnosis-container {
        background: linear-gradient(145deg, #0F172A 0%, #131F37 100%);
        border: 1px solid #2563EB;
        border-radius: 8px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    
    /* Recovery Container */
    .recovery-container {
        background: linear-gradient(145deg, #064E3B 0%, #065F46 100%);
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    /* Metric Value typography */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# STATE INITIALIZATION
# ---------------------------------------------------------
if "ml_engine" not in st.session_state:
    ml_engine = drillguard.DrillGuardML(model_dir="models")
    with st.spinner("Initializing Scikit-Learn Isolation Forest & Random Forest models..."):
        ml_engine.train_or_load()
    st.session_state["ml_engine"] = ml_engine

if "current_scenario" not in st.session_state:
    st.session_state["current_scenario"] = "NORMAL"

if "action_applied" not in st.session_state:
    st.session_state["action_applied"] = False

if "recovery_trajectory" not in st.session_state:
    st.session_state["recovery_trajectory"] = None

if "last_action_name" not in st.session_state:
    st.session_state["last_action_name"] = None

if "snapshot_key" not in st.session_state:
    st.session_state["snapshot_key"] = 0

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ DRILLGUARD AI")
    st.caption("Physical AI Diagnostic Control Room")
    st.markdown("<span class='badge-nvidia'>NVIDIA Nemotron</span> <span class='badge-nebius'>Nebius Token Factory</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("⚙️ Rig Operation Controls")
    operating_mode = st.selectbox(
        "Operating Mode",
        ["Rotary Drilling (Full Bit Contact)", "Tripping In / Out", "Circulating & Conditioning", "Casing Run"],
        index=0
    )

    st.markdown("#### 🚨 Fault Injection Selector")
    scenario_options = list(drillguard.SCENARIOS.keys())
    scenario_labels = [f"{k} - {drillguard.SCENARIOS[k]['name']}" for k in scenario_options]

    # Map label to code
    label_to_code = {f"{k} - {drillguard.SCENARIOS[k]['name']}": k for k in scenario_options}
    
    # Current index
    current_idx = scenario_options.index(st.session_state["current_scenario"])
    selected_label = st.selectbox(
        "Select Subsystem Scenario:",
        scenario_labels,
        index=current_idx,
        help="Inject synthetic fault patterns across Electrical (E1-E3), Mechanical (M1-M3), or Hydraulic (H1-H3) subsystems."
    )
    new_scenario = label_to_code[selected_label]

    if new_scenario != st.session_state["current_scenario"]:
        st.session_state["current_scenario"] = new_scenario
        st.session_state["action_applied"] = False
        st.session_state["recovery_trajectory"] = None
        st.session_state["last_action_name"] = None
        st.rerun()

    noise_mult = st.slider("Sensor Noise & Dynamic Turbulence", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 New Snapshot", use_container_width=True):
            st.session_state["snapshot_key"] += 1
            st.session_state["action_applied"] = False
            st.rerun()
    with c2:
        if st.button("♻️ Retrain Models", use_container_width=True):
            with st.spinner("Retraining Isolation Forest & Random Forest..."):
                st.session_state["ml_engine"].train_or_load(force_retrain=True)
            st.success("Models retrained!")

    st.markdown("---")
    st.subheader("🔑 Nebius Token Factory API")
    env_api_key = os.getenv("NEBIUS_API_KEY", "")
    api_key_input = st.text_input(
        "Nebius API Key (from .env or manual)",
        value=env_api_key if env_api_key and env_api_key != "your_nebius_api_key_here" else "",
        type="password",
        help="API Key for Nebius Token Factory (Studio). If unset, DrillGuard AI uses its physics-grounded local fallback."
    )
    if api_key_input:
        os.environ["NEBIUS_API_KEY"] = api_key_input

    nebius_model_choice = st.selectbox(
        "Nemotron Model:",
        [
            "nvidia/Llama-3_1-Nemotron-70B-Instruct-HF",
            "nvidia/nemotron-4-340b-instruct"
        ],
        index=0
    )
    os.environ["NEBIUS_MODEL"] = nebius_model_choice

    active_key = os.getenv("NEBIUS_API_KEY", "")
    if active_key and active_key != "your_nebius_api_key_here":
        st.markdown("<span class='badge-normal'>🟢 Nebius API Configured</span>", unsafe_allow_html=True)
        st.caption(f"Target: `{nebius_model_choice}`")
    else:
        st.markdown("<span class='badge-moderate'>🟠 Fallback Mode Active</span>", unsafe_allow_html=True)
        st.caption("No valid Nebius key detected. Local physics fallback active.")

    st.markdown("---")
    st.caption("🔒 **Physical AI Safety Notice**:")
    st.caption("Software-only simulation. Correlated synthetic data. Do not connect to or control physical machinery.")

# ---------------------------------------------------------
# CURRENT TELEMETRY SNAPSHOT & ML INFERENCE
# ---------------------------------------------------------
current_code = st.session_state["current_scenario"]
scenario_meta = drillguard.SCENARIOS[current_code]

# If action has been applied, fetch the stabilized state
if st.session_state["action_applied"] and st.session_state["recovery_trajectory"]:
    # Last point of recovery
    current_telemetry = st.session_state["recovery_trajectory"][-1]
    derived_feats = current_telemetry
else:
    # Generate live telemetry snapshot
    current_telemetry = drillguard.generate_telemetry_point(current_code, t_sec=float(st.session_state["snapshot_key"]), noise_mult=noise_mult)
    derived_feats = drillguard.compute_derived_features(current_telemetry)

# Run ML Analysis
ml_results = st.session_state["ml_engine"].analyze(derived_feats)

# ---------------------------------------------------------
# HEADER BAR - SCADA RIG IDENTITY
# ---------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

with header_col1:
    st.markdown(f"""
    <div style='padding-top: 5px;'>
        <div style='font-size: 1.6rem; font-weight: 800; letter-spacing: 0.5px;'>
            🛡️ DRILLGUARD AI <span style='font-size: 0.9rem; font-weight: 500; color: #94A3B8;'>| DEEP RIG SCADA</span>
        </div>
        <div style='font-size: 0.82rem; color: #64748B;'>
            RIG-ALPHA-07 // CYBER-PHYSICAL MONITORING & DIAGNOSTIC REASONING SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    status_class = "badge-normal" if not ml_results["is_anomaly"] else ("badge-critical" if scenario_meta["severity"] == "CRITICAL" else "badge-moderate")
    state_label = "RECOVERED / STABILIZED" if st.session_state["action_applied"] else ("HEALTHY (NOMINAL)" if not ml_results["is_anomaly"] else f"ANOMALY DETECTED ({scenario_meta['category'].upper()})")
    
    st.markdown(f"""
    <div style='text-align: right;'>
        <div style='font-size: 0.75rem; color: #94A3B8;'>RIG HEALTH STATUS</div>
        <span class='{status_class}'>{state_label}</span>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    utc_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""
    <div style='text-align: right;'>
        <div style='font-size: 0.75rem; color: #94A3B8;'>SCADA TELEMETRY SYNC</div>
        <div style='font-size: 0.85rem; font-family: monospace; color: #38BDF8;'>{utc_time}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 10px; margin-bottom: 18px; border-color: #1E293B;'/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIMULATED MACHINE RESPONSE BANNER (IF ACTION APPLIED)
# ---------------------------------------------------------
if st.session_state["action_applied"]:
    st.markdown(f"""
    <div class='recovery-container'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h4 style='margin: 0; color: #34D399;'>✅ Corrective Action Successfully Executed</h4>
                <p style='margin: 4px 0 0 0; color: #A7F3D0; font-size: 0.9rem;'>
                    Action applied: <strong>{st.session_state["last_action_name"]}</strong>. 
                    Transient recovery trajectory stabilized rig parameters back within nominal operating safety limits.
                </p>
            </div>
            <div>
                <span class='badge-normal'>MACHINE STABILIZED</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# OVERVIEW KEY METRICS (KPIs)
# ---------------------------------------------------------
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)

with kpi_c1:
    v_val = derived_feats["voltage_v"]
    v_delta = round(v_val - 480.0, 1)
    st.metric("Bus Voltage", f"{v_val:.1f} V", delta=f"{v_delta:+} V", delta_color="inverse" if abs(v_delta) > 15 else "normal")

with kpi_c2:
    i_val = derived_feats["current_a"]
    i_delta = round(i_val - 205.0, 1)
    st.metric("Motor Current", f"{i_val:.1f} A", delta=f"{i_delta:+} A", delta_color="inverse" if i_delta > 30 else "normal")

with kpi_c3:
    rpm_val = derived_feats["rpm"]
    st.metric("Top-Drive Speed", f"{rpm_val:.1f} RPM", delta=f"{round(rpm_val - 135.0, 1):+} RPM")

with kpi_c4:
    vib_val = derived_feats["vibration_rms_mms"]
    st.metric("Vibration RMS", f"{vib_val:.2f} mm/s", delta=f"{round(vib_val - 2.4, 2):+} mm/s", delta_color="inverse" if vib_val > 4.5 else "normal")

with kpi_c5:
    p_hyd_val = derived_feats["hydraulic_pressure_bar"]
    p_delta = round(p_hyd_val - 200.0, 1)
    st.metric("Hydraulic Pressure", f"{p_hyd_val:.1f} bar", delta=f"{p_delta:+} bar", delta_color="inverse" if abs(p_delta) > 30 else "normal")

with kpi_c6:
    dp_val = derived_feats["differential_pressure_bar"]
    st.metric("Filter dP", f"{dp_val:.2f} bar", delta=f"{round(dp_val - 1.4, 2):+} bar", delta_color="inverse" if dp_val > 3.0 else "normal")

# ---------------------------------------------------------
# THREE SUBSYSTEM TELEMETRY PANELS
# ---------------------------------------------------------
st.markdown("### 📡 Subsystem Telemetry Panels")

tab_elec, tab_mech, tab_hyd, tab_charts = st.tabs([
    "⚡ 1. Electrical Subsystem", 
    "⚙️ 2. Mechanical Subsystem", 
    "💧 3. Hydraulic Subsystem",
    "📈 Live Telemetry Waveforms"
])

# Tab 1: Electrical Subsystem
with tab_elec:
    st.markdown("<div class='scada-card scada-card-elec'>", unsafe_allow_html=True)
    e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)
    with e_col1:
        st.metric("Bus Frequency", f"{derived_feats['frequency_hz']:.2f} Hz", "Nominal: 60.0 Hz")
    with e_col2:
        st.metric("Power Factor (cosφ)", f"{derived_feats['power_factor']:.3f}", "Nominal: 0.91")
    with e_col3:
        st.metric("Active Power", f"{derived_feats['active_power_kw']:.1f} kW", f"{derived_feats['apparent_power_kva']:.1f} kVA Apparent")
    with e_col4:
        st.metric("VFD Temp", f"{derived_feats['vfd_temp_c']:.1f} °C", "Limit: 75.0 °C")
    with e_col5:
        st.metric("Motor Efficiency", f"{derived_feats['motor_efficiency']*100:.1f} %", "Est. Mech/Elec")
    st.markdown("</div>", unsafe_allow_html=True)

    # Subsystem Mini-Chart
    hist_df = drillguard.generate_telemetry_history(current_code, n_points=40)
    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(x=hist_df.index, y=hist_df["current_a"], name="Current (A)", line=dict(color="#38BDF8", width=2)))
    fig_e.add_trace(go.Scatter(x=hist_df.index, y=hist_df["voltage_v"], name="Voltage (V)", yaxis="y2", line=dict(color="#F43F5E", width=1.5, dash="dot")))
    fig_e.update_layout(
        title="Electrical Subsystem: Current & Voltage Correlation Window",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=260,
        margin=dict(l=40, r=40, t=40, b=30),
        yaxis=dict(title="Current (A)"),
        yaxis2=dict(title="Voltage (V)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_e, use_container_width=True)

# Tab 2: Mechanical Subsystem
with tab_mech:
    st.markdown("<div class='scada-card scada-card-mech'>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Drill Torque", f"{derived_feats['torque_knm']:.2f} kNm", "Nominal: 21.5 kNm")
    with m_col2:
        st.metric("Vibration Peak", f"{derived_feats['vibration_peak_mms']:.2f} mm/s", "Limit: 7.0 mm/s")
    with m_col3:
        st.metric("Crest Factor", f"{derived_feats['vibration_crest_factor']:.2f}", "Peak/RMS Ratio")
    with m_col4:
        st.metric("Bearing Temp", f"{derived_feats['bearing_temp_c']:.1f} °C", "Limit: 85.0 °C")
    with m_col5:
        st.metric("Gearbox Oil Temp", f"{derived_feats['gearbox_oil_temp_c']:.1f} °C", "Limit: 90.0 °C")
    st.markdown("</div>", unsafe_allow_html=True)

    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=hist_df.index, y=hist_df["vibration_rms_mms"], name="Vibration RMS (mm/s)", line=dict(color="#F59E0B", width=2)))
    fig_m.add_trace(go.Scatter(x=hist_df.index, y=hist_df["vibration_peak_mms"], name="Vibration Peak (mm/s)", line=dict(color="#EF4444", width=1.5, dash="dash")))
    fig_m.add_hline(y=7.1, line_dash="dot", line_color="#EF4444", annotation_text="ISO 10816 Class IV Danger Limit (7.1 mm/s)")
    fig_m.update_layout(
        title="Mechanical Vibration Signature & Peak Shock Envelope",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=260,
        margin=dict(l=40, r=40, t=40, b=30),
        yaxis=dict(title="Vibration (mm/s)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_m, use_container_width=True)

# Tab 3: Hydraulic Subsystem
with tab_hyd:
    st.markdown("<div class='scada-card scada-card-hyd'>", unsafe_allow_html=True)
    h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)
    with h_col1:
        st.metric("Hydraulic Flow", f"{derived_feats['flow_rate_lpm']:.1f} L/min", "Nominal: 380 L/min")
    with h_col2:
        st.metric("Fluid Temp", f"{derived_feats['fluid_temp_c']:.1f} °C", "Limit: 70.0 °C")
    with h_col3:
        st.metric("Pump Speed", f"{derived_feats['pump_rpm']:.0f} RPM", "Nominal: 1500 RPM")
    with h_col4:
        st.metric("Reservoir Level", f"{derived_feats['reservoir_level_pct']:.1f} %", "Min: 70 %")
    with h_col5:
        st.metric("Hydraulic Power", f"{derived_feats['hydraulic_power_kw']:.1f} kW", "Fluid Output Power")
    st.markdown("</div>", unsafe_allow_html=True)

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=hist_df.index, y=hist_df["hydraulic_pressure_bar"], name="Hydraulic Pressure (bar)", line=dict(color="#10B981", width=2)))
    fig_h.add_trace(go.Scatter(x=hist_df.index, y=hist_df["differential_pressure_bar"], name="Filter dP (bar)", yaxis="y2", line=dict(color="#A855F7", width=2)))
    fig_h.add_hline(y=5.0, yref="y2", line_dash="dash", line_color="#EF4444", annotation_text="Filter Bypass Trip (5.0 bar)")
    fig_h.update_layout(
        title="Hydraulic Main Loop Pressure & Micron Filter Differential Loss",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=260,
        margin=dict(l=40, r=40, t=40, b=30),
        yaxis=dict(title="Hydraulic Pressure (bar)"),
        yaxis2=dict(title="Filter dP (bar)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_h, use_container_width=True)

# Tab 4: Live Telemetry Waveforms (3-Subsystem Comparison)
with tab_charts:
    fig_all = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        subplot_titles=("⚡ Electrical (Current vs Nominal)", "⚙️ Mechanical (Vibration vs ISO Limit)", "💧 Hydraulic (Loop Pressure vs Nominal)"),
        vertical_spacing=0.08
    )
    fig_all.add_trace(go.Scatter(x=hist_df.index, y=hist_df["current_a"], name="Current (A)", line=dict(color="#38BDF8")), row=1, col=1)
    fig_all.add_hline(y=205.0, line_dash="dot", line_color="#64748B", row=1, col=1)

    fig_all.add_trace(go.Scatter(x=hist_df.index, y=hist_df["vibration_rms_mms"], name="Vibration (mm/s)", line=dict(color="#F59E0B")), row=2, col=1)
    fig_all.add_hline(y=7.1, line_dash="dash", line_color="#EF4444", row=2, col=1)

    fig_all.add_trace(go.Scatter(x=hist_df.index, y=hist_df["hydraulic_pressure_bar"], name="Pressure (bar)", line=dict(color="#10B981")), row=3, col=1)
    fig_all.add_hline(y=200.0, line_dash="dot", line_color="#64748B", row=3, col=1)

    fig_all.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=450,
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    st.plotly_chart(fig_all, use_container_width=True)

# ---------------------------------------------------------
# ML DIAGNOSTIC ENSEMBLE SECTION
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🤖 Dual Machine Learning Diagnostic Engine")

ml_col1, ml_col2 = st.columns([1, 1])

with ml_col1:
    st.markdown("#### 1. Isolation Forest (Anomaly Detection)")
    
    anom_score = ml_results["anomaly_score"]
    is_anom = ml_results["is_anomaly"]
    
    # Anomaly status indicator
    if is_anom:
        st.markdown(f"<span class='badge-critical'>🚨 ANOMALY DETECTED (Score: {anom_score:.3f})</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-normal'>✅ NOMINAL ENVELOPE (Score: {anom_score:.3f})</span>", unsafe_allow_html=True)

    # Anomaly Score Gauge
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=anom_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Anomaly Severity Score (%)", 'font': {'size': 14, 'color': '#CBD5E1'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#EF4444" if is_anom else "#10B981"},
            'bgcolor': "#1E293B",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [40, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.75,
                'value': 50.0
            }
        }
    ))
    gauge_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        height=200,
        margin=dict(l=30, r=30, t=30, b=20)
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

with ml_col2:
    st.markdown("#### 2. Random Forest (Fault Classification)")
    
    pred_fault_code = ml_results["predicted_fault_code"]
    pred_fault_name = ml_results["predicted_fault_name"]
    conf = ml_results["confidence"]
    
    st.markdown(f"**Predicted Signature:** `{pred_fault_code}` - **{pred_fault_name}** (Confidence: **{conf*100:.1f}%**)")
    
    # Class Probabilities Bar Chart
    probs = ml_results["class_probabilities"]
    prob_df = pd.DataFrame({
        "Scenario": list(probs.keys()),
        "Probability": [v * 100 for v in probs.values()]
    }).sort_values(by="Probability", ascending=True)

    bar_fig = px.bar(
        prob_df, 
        x="Probability", 
        y="Scenario", 
        orientation="h",
        text="Probability",
        color="Probability",
        color_continuous_scale=["#1E293B", "#38BDF8", "#F59E0B", "#EF4444"]
    )
    bar_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=200,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="Probability (%)", range=[0, 115]),
        yaxis=dict(title=""),
        coloraxis_showscale=False
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# Feature Attribution / Evidence Table
st.markdown("#### 🔍 Feature Drift & Diagnostic Evidence")
top_devs = ml_results.get("top_deviating_features", [])
if top_devs:
    dev_cols = st.columns(len(top_devs))
    for idx, d in enumerate(top_devs):
        with dev_cols[idx]:
            is_high_drift = d["deviation_pct"] > 25.0
            color = "#EF4444" if is_high_drift else "#F59E0B"
            st.markdown(f"""
            <div style='background-color: #1E293B; border-left: 3px solid {color}; padding: 10px; border-radius: 4px;'>
                <div style='font-size: 0.8rem; color: #94A3B8;'>{d['name']}</div>
                <div style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC;'>{d['value']} {d['unit']}</div>
                <div style='font-size: 0.75rem; color: {color};'>Baseline: {d['nominal']} (Δ {d['deviation_pct']}%)</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# NVIDIA NEMOTRON DIAGNOSIS VIA NEBIUS TOKEN FACTORY
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🧠 NVIDIA Nemotron Physical AI Diagnostic Reasoning")
st.caption("Integrated via Nebius Token Factory API (Studio) • Automated Local Fallback Protected")

# Execute Diagnosis
diag_engine = drillguard.NemotronDiagnosticEngine()
with st.spinner("Generating deep physical diagnostic reasoning..."):
    diag_response = diag_engine.diagnose(derived_feats, ml_results, scenario_meta)

# Render Diagnosis Card
source_badge = diag_response["badge"]
model_tag = diag_response["model_used"]
severity_badge = "badge-normal" if scenario_meta["severity"] == "NORMAL" else ("badge-critical" if scenario_meta["severity"] == "CRITICAL" else "badge-moderate")

st.markdown(f"""
<div class='diagnosis-container'>
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
        <div>
            <span style='font-size: 1.2rem; font-weight: 700; color: #60A5FA;'>Diagnostic Intelligence Engine</span>
            <div style='font-size: 0.8rem; color: #94A3B8;'>Model / Engine: <code>{model_tag}</code></div>
        </div>
        <div>
            {source_badge} &nbsp; <span class='{severity_badge}'>SEVERITY: {scenario_meta['severity']}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Markdown Diagnostic Text
st.markdown(diag_response["diagnostic_text"])

# ---------------------------------------------------------
# CLOSED-LOOP RECOMMENDED ACTION & SIMULATED MACHINE RESPONSE
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### ⚡ Corrective Action & Machine Response Closed-Loop")

act_col1, act_col2 = st.columns([2, 1])

with act_col1:
    st.markdown(f"**Recommended SOP Action:**")
    st.info(f"👉 **{scenario_meta['action_short']}**: {scenario_meta['recommended_action']}")

with act_col2:
    st.markdown("**Simulate Operator Intervention:**")
    apply_btn = st.button(
        f"⚡ Apply Simulated Action: {scenario_meta['action_short']}", 
        type="primary", 
        use_container_width=True,
        help="Simulates the physical transient response of the drill rig as the automated corrective action is applied."
    )

if apply_btn:
    with st.spinner("Simulating machine transient recovery curve..."):
        time.sleep(0.4)
        trajectory = drillguard.simulate_machine_response(current_code, steps=6)
        st.session_state["action_applied"] = True
        st.session_state["recovery_trajectory"] = trajectory
        st.session_state["last_action_name"] = scenario_meta["action_short"]
        st.rerun()

# If recovery trajectory exists, visualize recovery transient curves
if st.session_state["action_applied"] and st.session_state["recovery_trajectory"]:
    traj = st.session_state["recovery_trajectory"]
    traj_df = pd.DataFrame(traj)

    st.markdown("#### 🔄 Machine Response Transient Trajectory (Before vs After Action)")
    
    t_c1, t_c2, t_c3 = st.columns(3)
    
    with t_c1:
        fig_rec_i = px.line(
            traj_df, x="step_label", y="current_a", 
            title="Motor Current Stabilization (A)",
            markers=True
        )
        fig_rec_i.add_hline(y=205.0, line_dash="dot", line_color="#10B981", annotation_text="Nominal (205A)")
        fig_rec_i.update_layout(template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0B0F17", height=230, margin=dict(l=30, r=20, t=35, b=30))
        st.plotly_chart(fig_rec_i, use_container_width=True)

    with t_c2:
        fig_rec_v = px.line(
            traj_df, x="step_label", y="vibration_rms_mms", 
            title="Vibration Attenuation (mm/s)",
            markers=True, color_discrete_sequence=["#F59E0B"]
        )
        fig_rec_v.add_hline(y=2.4, line_dash="dot", line_color="#10B981", annotation_text="Nominal (2.4mm/s)")
        fig_rec_v.update_layout(template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0B0F17", height=230, margin=dict(l=30, r=20, t=35, b=30))
        st.plotly_chart(fig_rec_v, use_container_width=True)

    with t_c3:
        fig_rec_p = px.line(
            traj_df, x="step_label", y="hydraulic_pressure_bar", 
            title="Hydraulic Pressure Recovery (bar)",
            markers=True, color_discrete_sequence=["#10B981"]
        )
        fig_rec_p.add_hline(y=200.0, line_dash="dot", line_color="#10B981", annotation_text="Nominal (200 bar)")
        fig_rec_p.update_layout(template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0B0F17", height=230, margin=dict(l=30, r=20, t=35, b=30))
        st.plotly_chart(fig_rec_p, use_container_width=True)

# ---------------------------------------------------------
# FOOTER & SAFETY COMPLIANCE NOTICE
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748B; font-size: 0.8rem; padding: 15px;'>
    <strong>DrillGuard AI</strong> — Built for the <strong>Nebius x NVIDIA Global AI Hackathon 2026 (Physical AI Track)</strong>.<br/>
    ⚠️ <strong>COMPLIANCE & SAFETY NOTICE</strong>: This is a software-only cyber-physical simulator utilizing realistic correlated synthetic telemetry for demonstration and research purposes. 
    It is not a validated physical certified simulator. Do not connect to or attempt to control actual machinery.
</div>
""", unsafe_allow_html=True)

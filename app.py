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
    page_title="DrillGuard AI | SCADA Diagnostic System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# STYLES - INDUSTRIAL CONTROL-ROOM DARK THEME (HIGH CONTRAST)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Dark SCADA Background */
    .stApp {
        background-color: #0B0F17;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Container */
    .scada-header {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    
    /* Subsystem Cards */
    .scada-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25);
    }
    
    .scada-card-elec { border-top: 3px solid #38BDF8; }
    .scada-card-mech { border-top: 3px solid #F59E0B; }
    .scada-card-hyd  { border-top: 3px solid #10B981; }
    
    /* Tabs High Contrast Styling */
    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 8px 16px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }
    
    /* Status Badges */
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #059669;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid #D97706;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-fallback {
        background-color: rgba(148, 163, 184, 0.15);
        color: #CBD5E1;
        border: 1px solid #64748B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    
    /* Recovery Container */
    .recovery-container {
        background: #064E3B;
        border: 1px solid #10B981;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 10px;
        margin-bottom: 16px;
    }

    /* Metric Typography - High Contrast */
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #CBD5E1 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# STATE INITIALIZATION
# ---------------------------------------------------------
if "ml_engine" not in st.session_state:
    ml_engine = drillguard.DrillGuardML(model_dir="models")
    with st.spinner("Initializing ML models..."):
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
    st.markdown("### DRILLGUARD AI")
    st.caption("Cyber-Physical Rig SCADA System")
    st.markdown("---")

    st.subheader("Rig Operation Controls")
    operating_mode = st.selectbox(
        "Operating Mode",
        ["Rotary Drilling (Full Bit Contact)", "Tripping In / Out", "Circulating & Conditioning", "Casing Run"],
        index=0
    )

    st.markdown("#### Fault Injection Selector")
    scenario_options = list(drillguard.SCENARIOS.keys())
    scenario_labels = [f"{k} - {drillguard.SCENARIOS[k]['name']}" for k in scenario_options]

    label_to_code = {f"{k} - {drillguard.SCENARIOS[k]['name']}": k for k in scenario_options}
    current_idx = scenario_options.index(st.session_state["current_scenario"])
    
    selected_label = st.selectbox(
        "Select Scenario:",
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

    noise_mult = st.slider("Sensor Noise & Turbulence", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("New Snapshot", use_container_width=True):
            st.session_state["snapshot_key"] += 1
            st.session_state["action_applied"] = False
            st.rerun()
    with c2:
        if st.button("Retrain Models", use_container_width=True):
            with st.spinner("Retraining ML models..."):
                st.session_state["ml_engine"].train_or_load(force_retrain=True)
            st.success("Models retrained")

    # Secure Credentials & Connection Status from .env
    st.markdown("---")
    st.markdown("#### Nebius & NVIDIA Inference")
    
    env_api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    env_model = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning").strip()
    has_valid_key = bool(env_api_key and env_api_key != "your_nebius_api_key_here")

    if has_valid_key:
        st.markdown(f"""
        <div style='background-color: #111827; border: 1px solid #1F2937; border-left: 3px solid #10B981; padding: 10px 12px; border-radius: 4px; margin-bottom: 10px;'>
            <div style='font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;'>Inference Service</div>
            <div style='font-size: 0.88rem; font-weight: 600; color: #34D399;'>Nebius Token Factory (Connected)</div>
            <div style='font-size: 0.75rem; color: #CBD5E1; margin-top: 3px;'>Model: <code>{env_model}</code></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #111827; border: 1px solid #1F2937; border-left: 3px solid #64748B; padding: 10px 12px; border-radius: 4px; margin-bottom: 10px;'>
            <div style='font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;'>Inference Service</div>
            <div style='font-size: 0.88rem; font-weight: 600; color: #CBD5E1;'>Local Diagnostic Fallback</div>
            <div style='font-size: 0.75rem; color: #94A3B8; margin-top: 3px;'>No API key detected in .env</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Test Connection", use_container_width=True, help="Sends a test ping to Nebius Token Factory API."):
        with st.spinner("Testing API connection..."):
            test_res = drillguard.test_nemotron_connection()
            if test_res["success"]:
                st.success(f"Connected ({test_res['latency_sec']}s latency)")
            else:
                st.error(f"Connection test failed:\n{test_res['error']}")

    st.markdown("---")
    st.caption("Safety Notice: Software-only cyber-physical simulation using synthetic telemetry. Not connected to physical machinery.")

# ---------------------------------------------------------
# CURRENT TELEMETRY SNAPSHOT & ML INFERENCE
# ---------------------------------------------------------
current_code = st.session_state["current_scenario"]
scenario_meta = drillguard.SCENARIOS[current_code]

# If action has been applied, fetch the stabilized state
if st.session_state["action_applied"] and st.session_state["recovery_trajectory"]:
    current_telemetry = st.session_state["recovery_trajectory"][-1]
    derived_feats = current_telemetry
else:
    current_telemetry = drillguard.generate_telemetry_point(current_code, t_sec=float(st.session_state["snapshot_key"]), noise_mult=noise_mult)
    derived_feats = drillguard.compute_derived_features(current_telemetry)

# Run ML Analysis
ml_results = st.session_state["ml_engine"].analyze(derived_feats)

# ---------------------------------------------------------
# HEADER BAR - SCADA RIG IDENTITY
# ---------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

with header_col1:
    st.markdown("""
    <div style='padding-top: 4px;'>
        <div style='font-size: 1.45rem; font-weight: 800; letter-spacing: 0.5px;'>
            DRILLGUARD AI <span style='font-size: 0.85rem; font-weight: 500; color: #94A3B8;'>| RIG-ALPHA-07 SCADA</span>
        </div>
        <div style='font-size: 0.8rem; color: #94A3B8;'>
            Cyber-Physical Subsystem Telemetry & AI Diagnostic Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    status_class = "badge-normal" if not ml_results["is_anomaly"] else ("badge-critical" if scenario_meta["severity"] == "CRITICAL" else "badge-moderate")
    state_label = "RECOVERED / STABILIZED" if st.session_state["action_applied"] else ("HEALTHY (NOMINAL)" if not ml_results["is_anomaly"] else f"ANOMALY ({scenario_meta['category'].upper()})")
    
    st.markdown(f"""
    <div style='text-align: right;'>
        <div style='font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;'>Rig Health Status</div>
        <span class='{status_class}'>{state_label}</span>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    utc_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""
    <div style='text-align: right;'>
        <div style='font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;'>Telemetry Clock</div>
        <div style='font-size: 0.85rem; font-family: monospace; color: #38BDF8;'>{utc_time}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 8px; margin-bottom: 14px; border-color: #1E293B;'/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIMULATED MACHINE RESPONSE BANNER (IF ACTION APPLIED)
# ---------------------------------------------------------
if st.session_state["action_applied"]:
    st.markdown(f"""
    <div class='recovery-container'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <div style='font-size: 1.05rem; font-weight: 700; color: #34D399;'>Corrective Action Executed</div>
                <div style='margin-top: 3px; color: #D1FAE5; font-size: 0.88rem;'>
                    Action applied: <strong>{st.session_state["last_action_name"]}</strong>. 
                    Transient recovery trajectory stabilized rig parameters back within nominal operating safety limits.
                </div>
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
st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
st.markdown("### Subsystem Telemetry Panels")

tab_elec, tab_mech, tab_hyd, tab_charts = st.tabs([
    "1. Electrical Subsystem", 
    "2. Mechanical Subsystem", 
    "3. Hydraulic Subsystem",
    "Live Waveforms"
])

hist_df = drillguard.generate_telemetry_history(current_code, n_points=40)

# Tab 1: Electrical Subsystem
with tab_elec:
    st.markdown("<div class='scada-card scada-card-elec'>", unsafe_allow_html=True)
    e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)
    with e_col1:
        st.metric("Bus Frequency", f"{derived_feats['frequency_hz']:.2f} Hz", "Nominal: 60.0 Hz")
    with e_col2:
        st.metric("Power Factor", f"{derived_feats['power_factor']:.3f}", "Nominal: 0.91")
    with e_col3:
        st.metric("Active Power", f"{derived_feats['active_power_kw']:.1f} kW", f"{derived_feats['apparent_power_kva']:.1f} kVA Apparent")
    with e_col4:
        st.metric("VFD Temp", f"{derived_feats['vfd_temp_c']:.1f} °C", "Limit: 75.0 °C")
    with e_col5:
        st.metric("Motor Efficiency", f"{derived_feats['motor_efficiency']*100:.1f} %", "Est. Mech/Elec")
    st.markdown("</div>", unsafe_allow_html=True)

    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(x=hist_df.index, y=hist_df["current_a"], name="Current (A)", line=dict(color="#38BDF8", width=2)))
    fig_e.add_trace(go.Scatter(x=hist_df.index, y=hist_df["voltage_v"], name="Voltage (V)", yaxis="y2", line=dict(color="#F43F5E", width=1.5, dash="dot")))
    fig_e.update_layout(
        title="Electrical Subsystem: Current and Voltage Correlation",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=260,
        margin=dict(l=45, r=45, t=40, b=30),
        xaxis=dict(title="Sample Window", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0"), gridcolor="#1E293B"),
        yaxis=dict(title="Current (A)", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0"), gridcolor="#1E293B"),
        yaxis2=dict(title="Voltage (V)", overlaying="y", side="right", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1"))
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
    fig_m.add_hline(y=7.1, line_dash="dot", line_color="#EF4444", annotation_text="ISO 10816 Limit (7.1 mm/s)", annotation_font_color="#EF4444")
    fig_m.update_layout(
        title="Mechanical Vibration Signature and Shock Envelope",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=260,
        margin=dict(l=45, r=45, t=40, b=30),
        xaxis=dict(title="Sample Window", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0"), gridcolor="#1E293B"),
        yaxis=dict(title="Vibration (mm/s)", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0"), gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1"))
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
    fig_h.add_hline(y=5.0, yref="y2", line_dash="dash", line_color="#EF4444", annotation_text="Filter Bypass Trip (5.0 bar)", annotation_font_color="#EF4444")
    fig_h.update_layout(
        title="Hydraulic Loop Pressure and Micron Filter Differential Loss",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=260,
        margin=dict(l=45, r=45, t=40, b=30),
        xaxis=dict(title="Sample Window", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0"), gridcolor="#1E293B"),
        yaxis=dict(title="Pressure (bar)", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0"), gridcolor="#1E293B"),
        yaxis2=dict(title="Filter dP (bar)", overlaying="y", side="right", tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1"))
    )
    st.plotly_chart(fig_h, use_container_width=True)

# Tab 4: Live Telemetry Waveforms (3-Subsystem Comparison)
with tab_charts:
    fig_all = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        subplot_titles=("Electrical: Current vs Nominal", "Mechanical: Vibration vs ISO Limit", "Hydraulic: Loop Pressure vs Nominal"),
        vertical_spacing=0.08
    )
    fig_all.add_trace(go.Scatter(x=hist_df.index, y=hist_df["current_a"], name="Current (A)", line=dict(color="#38BDF8")), row=1, col=1)
    fig_all.add_hline(y=205.0, line_dash="dot", line_color="#94A3B8", row=1, col=1)

    fig_all.add_trace(go.Scatter(x=hist_df.index, y=hist_df["vibration_rms_mms"], name="Vibration (mm/s)", line=dict(color="#F59E0B")), row=2, col=1)
    fig_all.add_hline(y=7.1, line_dash="dash", line_color="#EF4444", row=2, col=1)

    fig_all.add_trace(go.Scatter(x=hist_df.index, y=hist_df["hydraulic_pressure_bar"], name="Pressure (bar)", line=dict(color="#10B981")), row=3, col=1)
    fig_all.add_hline(y=200.0, line_dash="dot", line_color="#94A3B8", row=3, col=1)

    fig_all.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=450,
        showlegend=False,
        margin=dict(l=45, r=45, t=40, b=30)
    )
    fig_all.update_xaxes(tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B")
    fig_all.update_yaxes(tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B")
    st.plotly_chart(fig_all, use_container_width=True)

# ---------------------------------------------------------
# ML DIAGNOSTIC ENSEMBLE SECTION
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Dual Machine Learning Diagnostic Engine")

ml_col1, ml_col2 = st.columns([1, 1])

with ml_col1:
    st.markdown("#### 1. Isolation Forest (Anomaly Detection)")
    
    anom_score = ml_results["anomaly_score"]
    is_anom = ml_results["is_anomaly"]
    
    if is_anom:
        st.markdown(f"<span class='badge-critical'>ANOMALY DETECTED (Score: {anom_score:.3f})</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-normal'>NOMINAL ENVELOPE (Score: {anom_score:.3f})</span>", unsafe_allow_html=True)

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
        height=320,
        margin=dict(l=30, r=30, t=30, b=20)
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

with ml_col2:
    st.markdown("#### 2. Random Forest (Fault Classification)")
    
    pred_fault_code = ml_results["predicted_fault_code"]
    pred_fault_name = ml_results["predicted_fault_name"]
    conf = ml_results["confidence"]
    
    st.markdown(f"**Predicted Signature:** `{pred_fault_code}` — **{pred_fault_name}** (Confidence: **{conf*100:.1f}%**)")
    
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
    bar_fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont=dict(color="#CBD5E1", size=11))
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F17",
        height=320,
        margin=dict(l=65, r=50, t=10, b=25),
        xaxis=dict(title="Probability (%)", range=[0, 115], tickfont=dict(color="#CBD5E1"), title_font=dict(color="#E2E8F0"), gridcolor="#1E293B"),
        yaxis=dict(title="", tickfont=dict(color="#CBD5E1", size=11)),
        coloraxis_showscale=False
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# Feature Attribution / Evidence Table
st.markdown("#### Feature Drift & Diagnostic Evidence")
top_devs = ml_results.get("top_deviating_features", [])
if top_devs:
    dev_cols = st.columns(len(top_devs))
    for idx, d in enumerate(top_devs):
        with dev_cols[idx]:
            is_high_drift = d["deviation_pct"] > 25.0
            border_color = "#EF4444" if is_high_drift else "#F59E0B"
            st.markdown(f"""
            <div style='background-color: #111827; border: 1px solid #1F2937; border-left: 3px solid {border_color}; padding: 10px 12px; border-radius: 4px;'>
                <div style='font-size: 0.78rem; color: #94A3B8;'>{d['name']}</div>
                <div style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC;'>{d['value']} {d['unit']}</div>
                <div style='font-size: 0.75rem; color: #CBD5E1;'>Baseline: {d['nominal']} (Δ {d['deviation_pct']}%)</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# NVIDIA NEMOTRON DIAGNOSIS VIA NEBIUS TOKEN FACTORY
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Nemotron Diagnostic Reasoning")
st.caption("Physical AI root-cause analysis powered by NVIDIA Nemotron via Nebius Token Factory")

# Execute Diagnosis
diag_engine = drillguard.NemotronDiagnosticEngine()
with st.spinner("Analyzing telemetry with NVIDIA Nemotron..."):
    diag_response = diag_engine.diagnose(derived_feats, ml_results, scenario_meta)

structured = diag_response.get("structured", {})
severity_val = structured.get("severity", scenario_meta["severity"]).upper()
fault_val = structured.get("fault", scenario_meta["name"])
comp_val = structured.get("component", scenario_meta["category"])
explanation_val = structured.get("explanation", "")
rec_action_val = structured.get("recommended_action", scenario_meta["action_short"])
maint_action_val = structured.get("maintenance_action", scenario_meta["recommended_action"])
checks_val = structured.get("additional_checks", [])

is_fallback = diag_response.get("is_fallback", False)
model_tag = diag_response.get("model_used", env_model)

if not is_fallback:
    inference_badge = "<span class='badge-live'>● Live Nemotron 3.5 Lightning (Nebius)</span>"
else:
    inference_badge = "<span class='badge-fallback'>○ Local Diagnostic Fallback</span>"

sev_css = "badge-normal" if severity_val == "NORMAL" else ("badge-critical" if severity_val == "CRITICAL" else "badge-moderate")

# Render Top Diagnostic Header
st.markdown(f"""
<div style='background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 12px 18px; margin-bottom: 14px;'>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>
        <div>
            <div style='font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;'>Inference Engine</div>
            <div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC;'>{model_tag}</div>
        </div>
        <div style='display: flex; align-items: center; gap: 10px;'>
            {inference_badge}
            <span class='{sev_css}'>SEVERITY: {severity_val}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Structured JSON Cards
d_col1, d_col2 = st.columns(2)
with d_col1:
    st.markdown(f"""
    <div style='background-color: #111827; border: 1px solid #1F2937; border-left: 3px solid #38BDF8; padding: 12px 14px; border-radius: 6px; margin-bottom: 12px;'>
        <div style='font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;'>Diagnosed Fault</div>
        <div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC;'>{fault_val}</div>
    </div>
    """, unsafe_allow_html=True)

with d_col2:
    st.markdown(f"""
    <div style='background-color: #111827; border: 1px solid #1F2937; border-left: 3px solid #F59E0B; padding: 12px 14px; border-radius: 6px; margin-bottom: 12px;'>
        <div style='font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;'>Degraded Component</div>
        <div style='font-size: 1.05rem; font-weight: 700; color: #F8FAFC;'>{comp_val}</div>
    </div>
    """, unsafe_allow_html=True)

# Physical Explanation Card
st.markdown("#### Root Cause Analysis")
st.markdown(f"""
<div style='background-color: #0F172A; border: 1px solid #1E293B; padding: 14px 18px; border-radius: 6px; font-size: 0.92rem; line-height: 1.6; color: #E2E8F0; margin-bottom: 14px;'>
    {explanation_val}
</div>
""", unsafe_allow_html=True)

# Two-column Action & Maintenance Cards
card_c1, card_c2 = st.columns(2)
with card_c1:
    st.markdown(f"""
    <div style='background-color: #111827; border: 1px solid #1F2937; border-left: 3px solid #10B981; padding: 12px 14px; border-radius: 6px; height: 100%;'>
        <div style='font-size: 0.75rem; font-weight: 700; color: #34D399; text-transform: uppercase; margin-bottom: 4px;'>Immediate Operational Action</div>
        <div style='font-size: 0.92rem; color: #F1F5F9; line-height: 1.5;'>{rec_action_val}</div>
    </div>
    """, unsafe_allow_html=True)

with card_c2:
    st.markdown(f"""
    <div style='background-color: #111827; border: 1px solid #1F2937; border-left: 3px solid #38BDF8; padding: 12px 14px; border-radius: 6px; height: 100%;'>
        <div style='font-size: 0.75rem; font-weight: 700; color: #60A5FA; text-transform: uppercase; margin-bottom: 4px;'>Maintenance SOP</div>
        <div style='font-size: 0.92rem; color: #F1F5F9; line-height: 1.5;'>{maint_action_val}</div>
    </div>
    """, unsafe_allow_html=True)

# Additional Checks Checklist
if checks_val:
    with st.expander("Diagnostic Verification Checklist", expanded=False):
        for chk in checks_val:
            st.markdown(f"- {chk}")

# ---------------------------------------------------------
# CLOSED-LOOP RECOMMENDED ACTION & SIMULATED MACHINE RESPONSE
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Corrective Action & Machine Recovery")

act_col1, act_col2 = st.columns([2, 1])

with act_col1:
    st.markdown(f"**Recommended Action:** {rec_action_val}")
    st.caption(f"Follow-up Maintenance: {maint_action_val}")

with act_col2:
    apply_btn = st.button(
        "Execute Corrective Action", 
        type="primary", 
        use_container_width=True,
        help="Simulates the drill rig physical recovery transient based on the recommended action."
    )

if apply_btn:
    with st.spinner("Simulating machine recovery transient..."):
        time.sleep(0.3)
        trajectory = drillguard.simulate_machine_response(current_code, steps=6)
        st.session_state["action_applied"] = True
        st.session_state["recovery_trajectory"] = trajectory
        st.session_state["last_action_name"] = rec_action_val
        st.rerun()

# If recovery trajectory exists, visualize recovery transient curves
if st.session_state["action_applied"] and st.session_state["recovery_trajectory"]:
    traj = st.session_state["recovery_trajectory"]
    traj_df = pd.DataFrame(traj)

    st.markdown("#### Machine Response Transient Trajectory")
    
    t_c1, t_c2, t_c3 = st.columns(3)
    
    with t_c1:
        fig_rec_i = px.line(
            traj_df, x="step_label", y="current_a", 
            title="Motor Current Stabilization (A)",
            markers=True
        )
        fig_rec_i.add_hline(y=205.0, line_dash="dot", line_color="#10B981", annotation_text="Nominal (205A)", annotation_font_color="#10B981")
        fig_rec_i.update_layout(
            template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0B0F17", height=230, margin=dict(l=35, r=20, t=35, b=30),
            xaxis=dict(title="", tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B"),
            yaxis=dict(title="Current (A)", tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B")
        )
        st.plotly_chart(fig_rec_i, use_container_width=True)

    with t_c2:
        fig_rec_v = px.line(
            traj_df, x="step_label", y="vibration_rms_mms", 
            title="Vibration Attenuation (mm/s)",
            markers=True, color_discrete_sequence=["#F59E0B"]
        )
        fig_rec_v.add_hline(y=2.4, line_dash="dot", line_color="#10B981", annotation_text="Nominal (2.4 mm/s)", annotation_font_color="#10B981")
        fig_rec_v.update_layout(
            template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0B0F17", height=230, margin=dict(l=35, r=20, t=35, b=30),
            xaxis=dict(title="", tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B"),
            yaxis=dict(title="Vibration (mm/s)", tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B")
        )
        st.plotly_chart(fig_rec_v, use_container_width=True)

    with t_c3:
        fig_rec_p = px.line(
            traj_df, x="step_label", y="hydraulic_pressure_bar", 
            title="Hydraulic Pressure Recovery (bar)",
            markers=True, color_discrete_sequence=["#10B981"]
        )
        fig_rec_p.add_hline(y=200.0, line_dash="dot", line_color="#10B981", annotation_text="Nominal (200 bar)", annotation_font_color="#10B981")
        fig_rec_p.update_layout(
            template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#0B0F17", height=230, margin=dict(l=35, r=20, t=35, b=30),
            xaxis=dict(title="", tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B"),
            yaxis=dict(title="Pressure (bar)", tickfont=dict(color="#CBD5E1"), gridcolor="#1E293B")
        )
        st.plotly_chart(fig_rec_p, use_container_width=True)

# ---------------------------------------------------------
# FOOTER & SAFETY COMPLIANCE NOTICE
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; padding: 14px;'>
    <strong>DrillGuard AI</strong> — Nebius x NVIDIA Global AI Hackathon 2026 (Physical AI Track).<br/>
    <strong>Safety Notice</strong>: Software-only cyber-physical simulator utilizing correlated synthetic telemetry for demonstration and research. Not connected to physical machinery.
</div>
""", unsafe_allow_html=True)

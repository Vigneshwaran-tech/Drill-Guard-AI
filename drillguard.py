"""
DrillGuard AI - Core Physical AI Diagnostic Engine
Nebius x NVIDIA Global AI Hackathon 2026 - Physical AI Track

Simulates a 3-subsystem industrial drill rig:
1. Electrical Subsystem (VFD, Current, Voltage, Power Factor, Bus Frequency)
2. Mechanical Subsystem (Top-drive RPM, Torque, Multi-axis Vibration, Bearings, Gearbox)
3. Hydraulic Subsystem (Main Loop Pressure, Flow, Filter dP, Fluid Temp, Reservoir)

Features:
- Telemetry generation with realistic correlated sensor physics (clearly labeled SIMULATED)
- Physical feature engineering
- Scikit-learn Isolation Forest (Anomaly Detection)
- Scikit-learn Random Forest (Multi-class Fault Classification)
- NVIDIA Nemotron Diagnostic Engine via Nebius Token Factory API
- Physics-grounded local fallback diagnosis when API key is unset/offline
- Closed-loop machine response simulation for corrective actions
"""

import os
import time
import math
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check OpenAI SDK
try:
    from openai import (
        OpenAI,
        APIError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        RateLimitError,
        BadRequestError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ---------------------------------------------------------
# CONSTANTS & SCENARIO DEFINITIONS
# ---------------------------------------------------------

SCENARIOS = {
    "NORMAL": {
        "code": "NORMAL",
        "name": "Normal Operation",
        "category": "Nominal",
        "severity": "NORMAL",
        "description": "All subsystems operating within calibrated baseline industrial safety envelopes.",
        "recommended_action": "Maintain standard automated drilling schedule; continue routine telemetry logging.",
        "action_short": "Continue Normal Operation",
    },
    "E1": {
        "code": "E1",
        "name": "E1: Electrical Overcurrent",
        "category": "Electrical",
        "severity": "CRITICAL",
        "description": "Massive stator phase overcurrent detected with rapid VFD thermal escalation.",
        "recommended_action": "Clamp VFD inverter current limit to 210A, throttle drill rotary torque, and engage auxiliary cabinet blowers.",
        "action_short": "Clamp VFD Inverter Current Limit",
    },
    "E2": {
        "code": "E2",
        "name": "E2: Voltage Instability",
        "category": "Electrical",
        "severity": "MODERATE",
        "description": "Severe 3-phase bus voltage sags/swells and frequency jitter due to generator bus hunting.",
        "recommended_action": "Switch rig bus transformer tap changer to automated regulation mode and engage dynamic line reactors.",
        "action_short": "Engage Active Voltage Conditioning",
    },
    "E3": {
        "code": "E3",
        "name": "E3: Motor Abnormality",
        "category": "Electrical",
        "severity": "CRITICAL",
        "description": "Top-drive motor stator winding asymmetry, excessive slip, and severe thermal buildup without commensurate mechanical output.",
        "recommended_action": "Derate motor duty cycle by 40%, re-zero vector encoder offset, and rebalance PWM carrier frequency.",
        "action_short": "Derate Duty Cycle & Re-balance PWM",
    },
    "M1": {
        "code": "M1",
        "name": "M1: Excessive Vibration",
        "category": "Mechanical",
        "severity": "CRITICAL",
        "description": "Harmonic drill string lateral whirl and torsional stick-slip causing dangerous vibration RMS & peak spikes.",
        "recommended_action": "Activate Soft-Torque rotary feedback control, reduce Weight-On-Bit (WOB) by 25%, and elevate rotary speed by 15 RPM to break resonance.",
        "action_short": "Engage Soft-Torque Dampener & Tune WOB",
    },
    "M2": {
        "code": "M2",
        "name": "M2: Bearing Degradation",
        "category": "Mechanical",
        "severity": "MODERATE",
        "description": "Inner race fatigue spalling in top-drive main thrust bearing; elevated high-frequency crest factor and thermal rise.",
        "recommended_action": "Trigger automated high-pressure synthetic grease purging cycle and cap maximum top-drive RPM to 110.",
        "action_short": "Purge Bearing Grease & Cap RPM",
    },
    "M3": {
        "code": "M3",
        "name": "M3: Gearbox Abnormality",
        "category": "Mechanical",
        "severity": "MODERATE",
        "description": "Planetary stage tooth mesh distress; accelerated lubricant breakdown and gearbox oil overheating.",
        "recommended_action": "Engage auxiliary lube cooling radiator, circulate secondary filtration loop, and limit drill torque slew rate.",
        "action_short": "Engage Secondary Lube Cooler & Limit Torque Slew",
    },
    "H1": {
        "code": "H1",
        "name": "H1: Pump Degradation",
        "category": "Hydraulic",
        "severity": "MODERATE",
        "description": "Piston slipper wear and internal swashplate bypass in main hydraulic pump; pressure decay and flow collapse under load.",
        "recommended_action": "Transfer hydraulic power unit circuit to redundant standby Pump-B and schedule swashplate valve overhaul.",
        "action_short": "Switch to Redundant Standby Pump-B",
    },
    "H2": {
        "code": "H2",
        "name": "H2: Filter Blockage",
        "category": "Hydraulic",
        "severity": "MODERATE",
        "description": "High particulate contamination causing differential pressure surge across return-line micron filter; cavitation risk.",
        "recommended_action": "Rotate duplex filter selector to clean standby cartridge and initiate automatic backflush sequence.",
        "action_short": "Switch Duplex Filter Cartridge & Backflush",
    },
    "H3": {
        "code": "H3",
        "name": "H3: Hydraulic Pressure Loss",
        "category": "Hydraulic",
        "severity": "CRITICAL",
        "description": "Severe hydraulic fluid pressure loss due to high-pressure manifold seal breach or stuck open relief valve.",
        "recommended_action": "Isolate high-pressure supply manifold, engage safety accumulator emergency hold, and trip rig hydraulic safety stop.",
        "action_short": "Isolate Manifold & Deploy Safety Accumulator",
    },
}

# Sensor Nominal Specs
SENSOR_SPECS = {
    # Electrical Subsystem
    "voltage_v": {"name": "Bus Voltage", "unit": "V", "nom": 480.0, "min": 450.0, "max": 510.0, "subsystem": "Electrical"},
    "current_a": {"name": "Motor Current", "unit": "A", "nom": 205.0, "min": 150.0, "max": 250.0, "subsystem": "Electrical"},
    "frequency_hz": {"name": "Bus Frequency", "unit": "Hz", "nom": 60.0, "min": 59.2, "max": 60.8, "subsystem": "Electrical"},
    "power_factor": {"name": "Power Factor", "unit": "cosφ", "nom": 0.91, "min": 0.85, "max": 0.96, "subsystem": "Electrical"},
    "vfd_temp_c": {"name": "VFD Inverter Temp", "unit": "°C", "nom": 52.0, "min": 35.0, "max": 70.0, "subsystem": "Electrical"},
    "active_power_kw": {"name": "Active Power", "unit": "kW", "nom": 155.0, "min": 110.0, "max": 195.0, "subsystem": "Electrical"},

    # Mechanical Subsystem
    "rpm": {"name": "Top-Drive RPM", "unit": "RPM", "nom": 135.0, "min": 110.0, "max": 160.0, "subsystem": "Mechanical"},
    "torque_knm": {"name": "Drill String Torque", "unit": "kNm", "nom": 21.5, "min": 15.0, "max": 28.0, "subsystem": "Mechanical"},
    "vibration_rms_mms": {"name": "Vibration RMS", "unit": "mm/s", "nom": 2.4, "min": 0.5, "max": 4.5, "subsystem": "Mechanical"},
    "vibration_peak_mms": {"name": "Vibration Peak", "unit": "mm/s", "nom": 3.8, "min": 1.0, "max": 7.0, "subsystem": "Mechanical"},
    "bearing_temp_c": {"name": "Main Bearing Temp", "unit": "°C", "nom": 64.0, "min": 45.0, "max": 80.0, "subsystem": "Mechanical"},
    "gearbox_oil_temp_c": {"name": "Gearbox Oil Temp", "unit": "°C", "nom": 68.0, "min": 45.0, "max": 85.0, "subsystem": "Mechanical"},

    # Hydraulic Subsystem
    "hydraulic_pressure_bar": {"name": "Hydraulic Pressure", "unit": "bar", "nom": 200.0, "min": 175.0, "max": 225.0, "subsystem": "Hydraulic"},
    "flow_rate_lpm": {"name": "Hydraulic Flow", "unit": "L/min", "nom": 380.0, "min": 330.0, "max": 430.0, "subsystem": "Hydraulic"},
    "differential_pressure_bar": {"name": "Filter dP", "unit": "bar", "nom": 1.4, "min": 0.5, "max": 2.5, "subsystem": "Hydraulic"},
    "fluid_temp_c": {"name": "Hydraulic Fluid Temp", "unit": "°C", "nom": 54.0, "min": 35.0, "max": 70.0, "subsystem": "Hydraulic"},
    "pump_rpm": {"name": "Pump Speed", "unit": "RPM", "nom": 1500.0, "min": 1420.0, "max": 1560.0, "subsystem": "Hydraulic"},
    "reservoir_level_pct": {"name": "Reservoir Level", "unit": "%", "nom": 88.0, "min": 75.0, "max": 98.0, "subsystem": "Hydraulic"},
}

RAW_SENSOR_NAMES = list(SENSOR_SPECS.keys())

FEATURE_NAMES = RAW_SENSOR_NAMES + [
    "apparent_power_kva",
    "mechanical_power_kw",
    "motor_efficiency",
    "vibration_crest_factor",
    "hydraulic_power_kw",
    "filter_clog_ratio",
    "volumetric_efficiency_ratio",
    "thermal_headroom_c",
    "voltage_deviation_pct",
    "current_overload_pct",
]


# ---------------------------------------------------------
# SYNTHETIC TELEMETRY GENERATOR
# ---------------------------------------------------------

def generate_telemetry_point(scenario_code: str, t_sec: float = 0.0, noise_mult: float = 1.0) -> Dict[str, float]:
    """
    Generates a single synthetic sensor telemetry sample with realistic correlated physical responses.
    Clearly labeled as simulated drill-rig telemetry.
    """
    rng = np.random.default_rng(int((time.time() * 1000 + t_sec * 100) % 2**31))

    # Base nominal values with stochastic operational drift
    v = 480.0 + rng.normal(0, 2.0 * noise_mult)
    i = 205.0 + rng.normal(0, 3.5 * noise_mult)
    f = 60.0 + rng.normal(0, 0.05 * noise_mult)
    pf = 0.91 + rng.normal(0, 0.008 * noise_mult)
    vfd_t = 52.0 + rng.normal(0, 0.8 * noise_mult)

    rpm = 135.0 + rng.normal(0, 1.8 * noise_mult)
    torq = 21.5 + rng.normal(0, 0.6 * noise_mult)
    vib_rms = 2.4 + rng.normal(0, 0.15 * noise_mult)
    vib_peak = 3.8 + rng.normal(0, 0.25 * noise_mult)
    bear_t = 64.0 + rng.normal(0, 0.7 * noise_mult)
    gear_t = 68.0 + rng.normal(0, 0.8 * noise_mult)

    hyd_p = 200.0 + rng.normal(0, 2.5 * noise_mult)
    flow = 380.0 + rng.normal(0, 4.0 * noise_mult)
    dp = 1.4 + rng.normal(0, 0.08 * noise_mult)
    fluid_t = 54.0 + rng.normal(0, 0.6 * noise_mult)
    p_rpm = 1500.0 + rng.normal(0, 5.0 * noise_mult)
    res_lvl = 88.0 + rng.normal(0, 0.3 * noise_mult)

    # Injected Fault Physics
    if scenario_code == "E1":  # Electrical Overcurrent
        i = 385.0 + rng.normal(0, 15.0 * noise_mult)
        v = 462.0 + rng.normal(0, 4.0 * noise_mult)  # sag under heavy current draw
        pf = 0.82 + rng.normal(0, 0.02 * noise_mult)
        vfd_t = 88.5 + rng.normal(0, 2.5 * noise_mult)
        vib_rms += 0.8

    elif scenario_code == "E2":  # Voltage Instability
        # Erratic bus swing and hunting
        swing = 65.0 * math.sin(t_sec * 1.5)
        v = 480.0 + swing + rng.normal(0, 18.0 * noise_mult)
        f = 60.0 + 1.2 * math.sin(t_sec * 0.8) + rng.normal(0, 0.45 * noise_mult)
        pf = 0.74 + rng.normal(0, 0.05 * noise_mult)
        i = 238.0 + rng.normal(0, 22.0 * noise_mult)

    elif scenario_code == "E3":  # Motor Abnormality
        i = 295.0 + rng.normal(0, 8.0 * noise_mult)  # high draw without torque
        rpm = 105.0 + rng.normal(0, 4.5 * noise_mult)  # severe rotor slip
        vfd_t = 94.0 + rng.normal(0, 2.2 * noise_mult)
        torq = 17.0 + rng.normal(0, 1.2 * noise_mult)
        vib_rms = 4.8 + rng.normal(0, 0.3 * noise_mult)

    elif scenario_code == "M1":  # Excessive Vibration
        vib_rms = 18.4 + 4.0 * math.sin(t_sec * 3.2) + rng.normal(0, 2.0 * noise_mult)
        vib_peak = 34.2 + 8.0 * math.sin(t_sec * 3.2) + rng.normal(0, 3.5 * noise_mult)
        torq = 21.5 + 4.5 * math.sin(t_sec * 4.0) + rng.normal(0, 1.5 * noise_mult)  # stick-slip torque chatter
        rpm = 135.0 + rng.normal(0, 5.0 * noise_mult)

    elif scenario_code == "M2":  # Bearing Degradation
        bear_t = 104.5 + rng.normal(0, 2.8 * noise_mult)
        vib_rms = 7.6 + rng.normal(0, 0.6 * noise_mult)
        vib_peak = 18.2 + rng.normal(0, 1.5 * noise_mult)  # elevated crest factor
        torq += 2.0  # slight friction drag

    elif scenario_code == "M3":  # Gearbox Abnormality
        gear_t = 109.0 + rng.normal(0, 3.0 * noise_mult)
        vib_rms = 6.4 + rng.normal(0, 0.5 * noise_mult)
        vib_peak = 14.5 + rng.normal(0, 1.2 * noise_mult)
        torq = 23.8 + rng.normal(0, 1.8 * noise_mult)

    elif scenario_code == "H1":  # Pump Degradation
        hyd_p = 126.0 + rng.normal(0, 5.0 * noise_mult)  # pressure decay
        flow = 235.0 + rng.normal(0, 9.0 * noise_mult)   # severe flow collapse
        fluid_t = 82.0 + rng.normal(0, 2.0 * noise_mult) # friction heating from internal bypass
        p_rpm = 1515.0 + rng.normal(0, 6.0 * noise_mult)

    elif scenario_code == "H2":  # Filter Blockage
        dp = 7.8 + rng.normal(0, 0.45 * noise_mult)     # massive dP surge (nominal 1.4)
        hyd_p = 242.0 + rng.normal(0, 4.0 * noise_mult)  # upstream backpressure
        flow = 310.0 + rng.normal(0, 6.0 * noise_mult)   # restricted flow

    elif scenario_code == "H3":  # Hydraulic Pressure Loss
        hyd_p = 56.0 + rng.normal(0, 4.5 * noise_mult)  # catastrophic pressure collapse
        flow = 165.0 + rng.normal(0, 14.0 * noise_mult)
        res_lvl = 62.0 + rng.normal(0, 1.2 * noise_mult) # leaking fluid reservoir drop
        dp = 0.4 + rng.normal(0, 0.05 * noise_mult)

    # Correlated active power: P = sqrt(3) * V * I * pf / 1000
    active_pwr = (math.sqrt(3) * v * i * pf) / 1000.0

    return {
        "voltage_v": round(float(v), 2),
        "current_a": round(float(i), 2),
        "frequency_hz": round(float(f), 2),
        "power_factor": round(float(np.clip(pf, 0.5, 0.99)), 3),
        "vfd_temp_c": round(float(vfd_t), 1),
        "active_power_kw": round(float(active_pwr), 2),
        "rpm": round(float(rpm), 1),
        "torque_knm": round(float(torq), 2),
        "vibration_rms_mms": round(float(max(0.1, vib_rms)), 2),
        "vibration_peak_mms": round(float(max(0.2, vib_peak)), 2),
        "bearing_temp_c": round(float(bear_t), 1),
        "gearbox_oil_temp_c": round(float(gear_t), 1),
        "hydraulic_pressure_bar": round(float(max(5.0, hyd_p)), 1),
        "flow_rate_lpm": round(float(max(10.0, flow)), 1),
        "differential_pressure_bar": round(float(max(0.1, dp)), 2),
        "fluid_temp_c": round(float(fluid_t), 1),
        "pump_rpm": round(float(p_rpm), 1),
        "reservoir_level_pct": round(float(np.clip(res_lvl, 10.0, 100.0)), 1),
    }


def compute_derived_features(data: Dict[str, float]) -> Dict[str, float]:
    """
    Computes physics-derived domain features from raw drill rig telemetry.
    """
    v = data["voltage_v"]
    i = data["current_a"]
    pf = data["power_factor"]
    p_active = data["active_power_kw"]
    rpm = data["rpm"]
    torq = data["torque_knm"]
    vib_rms = data["vibration_rms_mms"]
    vib_peak = data["vibration_peak_mms"]
    p_hyd = data["hydraulic_pressure_bar"]
    flow = data["flow_rate_lpm"]
    dp = data["differential_pressure_bar"]
    pump_rpm = data["pump_rpm"]

    # 1. Apparent Power (kVA)
    apparent_pwr = (math.sqrt(3) * v * i) / 1000.0

    # 2. Mechanical Power delivered to drill string (kW): P_mech = Torque(Nm) * omega(rad/s)
    mech_pwr = (torq * 1000.0) * (rpm * 2.0 * math.pi / 60.0) / 1000.0

    # 3. Motor Efficiency (Mechanical Power / Active Electrical Power)
    motor_eff = float(np.clip(mech_pwr / (p_active + 1e-4), 0.0, 1.0))

    # 4. Vibration Crest Factor (Peak / RMS) - sharp indicator of bearing / impact degradation
    crest_factor = vib_peak / (vib_rms + 1e-4)

    # 5. Hydraulic Output Power (kW) = (Pressure in Pa * Flow in m3/s) / 1000
    hyd_power = ((p_hyd * 1e5) * (flow / 60000.0)) / 1000.0

    # 6. Hydraulic Filter Clog Ratio (Normalized relative to clean filter nominal 1.4 bar)
    filter_ratio = dp / 1.4

    # 7. Volumetric Efficiency Ratio (Flow per pump rev relative to nominal baseline)
    vol_ratio = (flow / (pump_rpm + 1e-4)) / (380.0 / 1500.0)

    # 8. Thermal Headroom (°C) below safety shutdown limit of 100°C
    max_temp = max(data["vfd_temp_c"], data["bearing_temp_c"], data["gearbox_oil_temp_c"], data["fluid_temp_c"])
    thermal_headroom = 100.0 - max_temp

    # 9. Voltage Deviation Percentage
    v_dev = abs(v - 480.0) / 480.0 * 100.0

    # 10. Current Overload Percentage above nominal 205A
    i_overload = max(0.0, (i - 205.0) / 205.0 * 100.0)

    features = dict(data)
    features.update({
        "apparent_power_kva": round(apparent_pwr, 2),
        "mechanical_power_kw": round(mech_pwr, 2),
        "motor_efficiency": round(motor_eff, 3),
        "vibration_crest_factor": round(crest_factor, 2),
        "hydraulic_power_kw": round(hyd_power, 2),
        "filter_clog_ratio": round(filter_ratio, 2),
        "volumetric_efficiency_ratio": round(vol_ratio, 3),
        "thermal_headroom_c": round(thermal_headroom, 1),
        "voltage_deviation_pct": round(v_dev, 2),
        "current_overload_pct": round(i_overload, 2),
    })
    return features


def generate_telemetry_history(scenario_code: str, n_points: int = 50, step_sec: float = 1.0) -> pd.DataFrame:
    """
    Generates a continuous time series history dataframe of telemetry for UI trend charts.
    """
    records = []
    now = time.time()
    for idx in range(n_points):
        t_offset = (idx - n_points) * step_sec
        pt = generate_telemetry_point(scenario_code, t_sec=idx * step_sec, noise_mult=1.0)
        pt_feat = compute_derived_features(pt)
        pt_feat["timestamp"] = pd.to_datetime(now + t_offset, unit="s")
        pt_feat["scenario"] = scenario_code
        records.append(pt_feat)
    return pd.DataFrame(records)


# ---------------------------------------------------------
# MACHINE LEARNING MODULE (Anomaly & Classification)
# ---------------------------------------------------------

class DrillGuardML:
    """
    Embedded machine learning diagnostics:
    - Scikit-learn Isolation Forest for Anomaly Detection (Trained on NORMAL baseline)
    - Scikit-learn Random Forest for 10-Class Fault Classification
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.iso_forest: IsolationForest = None
        self.rf_classifier: RandomForestClassifier = None
        self.classes: List[str] = list(SCENARIOS.keys())
        self.is_trained = False

    def train_or_load(self, force_retrain: bool = False):
        os.makedirs(self.model_dir, exist_ok=True)
        iso_path = os.path.join(self.model_dir, "isolation_forest.joblib")
        rf_path = os.path.join(self.model_dir, "random_forest.joblib")

        if not force_retrain and os.path.exists(iso_path) and os.path.exists(rf_path):
            try:
                self.iso_forest = joblib.load(iso_path)
                self.rf_classifier = joblib.load(rf_path)
                self.is_trained = True
                return
            except Exception:
                pass  # Retrain if corrupted

        self._train_models()
        try:
            joblib.dump(self.iso_forest, iso_path)
            joblib.dump(self.rf_classifier, rf_path)
        except Exception:
            pass

    def _train_models(self):
        """Generates synthetic training dataset and trains both Isolation Forest and Random Forest."""
        samples_per_class = 250
        x_data = []
        y_labels = []
        normal_samples = []

        for sc_code in self.classes:
            for s_idx in range(samples_per_class):
                pt = generate_telemetry_point(sc_code, t_sec=s_idx * 0.2, noise_mult=1.2)
                feats = compute_derived_features(pt)
                vec = [feats[fn] for fn in FEATURE_NAMES]
                x_data.append(vec)
                y_labels.append(sc_code)
                if sc_code == "NORMAL":
                    normal_samples.append(vec)

        X = np.array(x_data)
        y = np.array(y_labels)
        X_normal = np.array(normal_samples)

        # 1. Isolation Forest: Calibrated strictly on healthy baseline physics
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.03,
            random_state=42,
            n_jobs=-1
        )
        self.iso_forest.fit(X_normal)

        # 2. Random Forest Classifier: Trained for multi-class fault classification
        self.rf_classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            random_state=42,
            n_jobs=-1
        )
        self.rf_classifier.fit(X, y)
        self.is_trained = True

    def analyze(self, telemetry_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs dual inference: Isolation Forest anomaly scoring + Random Forest fault classification.
        """
        if not self.is_trained:
            self.train_or_load()

        vec = np.array([[telemetry_features[fn] for fn in FEATURE_NAMES]])

        # Anomaly Detection
        # score_samples returns opposite of anomaly score (lower is more abnormal)
        raw_score = float(self.iso_forest.score_samples(vec)[0])
        pred_iso = int(self.iso_forest.predict(vec)[0])  # +1 for inlier, -1 for outlier
        is_anomaly = bool(pred_iso == -1)
        # Normalize anomaly score to 0.0 (normal) - 1.0 (highly anomalous)
        normalized_anomaly_score = float(np.clip(1.0 - (raw_score + 0.6) / 0.45, 0.0, 1.0))

        # Fault Classification
        probs = self.rf_classifier.predict_proba(vec)[0]
        class_probs = {cls_name: float(p) for cls_name, p in zip(self.rf_classifier.classes_, probs)}
        predicted_code = str(self.rf_classifier.predict(vec)[0])
        confidence = float(class_probs.get(predicted_code, 0.0))

        # Feature Importance / Attribution for Explainability
        # Compare current features against nominal sensor baselines
        feature_deviations = []
        for fn in FEATURE_NAMES:
            cur_val = telemetry_features[fn]
            if fn in SENSOR_SPECS:
                nom = SENSOR_SPECS[fn]["nom"]
                unit = SENSOR_SPECS[fn]["unit"]
                diff_pct = abs(cur_val - nom) / (abs(nom) + 1e-4) * 100.0
                feature_deviations.append({
                    "feature": fn,
                    "name": SENSOR_SPECS[fn]["name"],
                    "value": cur_val,
                    "nominal": nom,
                    "unit": unit,
                    "deviation_pct": round(diff_pct, 1),
                })
        # Sort by largest deviation from nominal
        feature_deviations.sort(key=lambda x: x["deviation_pct"], reverse=True)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(normalized_anomaly_score, 3),
            "raw_iso_score": round(raw_score, 4),
            "predicted_fault_code": predicted_code,
            "predicted_fault_name": SCENARIOS.get(predicted_code, {}).get("name", predicted_code),
            "confidence": round(confidence, 3),
            "class_probabilities": {k: round(v, 4) for k, v in class_probs.items()},
            "top_deviating_features": feature_deviations[:5],
            "severity": SCENARIOS.get(predicted_code, {}).get("severity", "NORMAL"),
        }


# ---------------------------------------------------------
# NEBIUS TOKEN FACTORY & NVIDIA NEMOTRON DIAGNOSTIC ENGINE
# ---------------------------------------------------------

def format_nebius_error(e: Exception, model: str, base_url: str) -> str:
    """Format API errors safely without exposing secret credentials."""
    if not HAS_OPENAI:
        return f"OpenAI SDK error: {str(e)}"

    if isinstance(e, AuthenticationError):
        return (
            "Authentication Failed (HTTP 401): The Nebius API key is invalid, revoked, or unauthorized. "
            "Please check NEBIUS_API_KEY in .env or generate a new key in Nebius Token Factory console."
        )
    elif isinstance(e, PermissionDeniedError):
        return (
            "Permission Denied (HTTP 403): Account lacks permission to access this Nebius Token Factory resource."
        )
    elif isinstance(e, RateLimitError):
        return (
            "Quota / Billing / Rate Limit (HTTP 429): Nebius Token Factory token credit limit reached or rate limit exceeded. "
            "Please check your Nebius project billing balance."
        )
    elif isinstance(e, NotFoundError):
        return (
            f"Model Not Found (HTTP 404): The model ID '{model}' was not found on regional endpoint '{base_url}'. "
            "Please verify the model identifier."
        )
    elif isinstance(e, BadRequestError):
        return f"Bad Request (HTTP 400): Incompatible request parameters for model '{model}': {str(e)}"
    elif isinstance(e, (APITimeoutError, APIConnectionError)):
        return (
            f"Network Connection Failed / Timeout: Unable to reach Nebius regional endpoint at '{base_url}'. "
            "Please verify internet connection and regional URL availability."
        )
    elif isinstance(e, InternalServerError):
        return "Nebius Server Error (HTTP 500/503): Nebius Token Factory is experiencing internal server issues."
    elif isinstance(e, APIError):
        return f"Nebius API Error: {getattr(e, 'message', str(e))}"
    else:
        return f"Unexpected Error: {str(e)}"


def extract_and_validate_json(text: str, fallback_entry: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Extracts structured JSON from model output and validates required schema:
    severity, explanation, fault, component, recommended_action, maintenance_action, additional_checks.
    Handles reasoning/thinking traces and markdown fences robustly.
    """
    extracted_obj = None

    # 1. Try markdown fenced code block: ```json ... ``` or ``` ... ```
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    for block in reversed(fence_matches):
        block = block.strip()
        if block.startswith("{") and block.endswith("}"):
            try:
                extracted_obj = json.loads(block)
                break
            except Exception:
                pass

    # 2. Search for valid JSON object by finding matching braces backwards from the end of the string
    if not extracted_obj:
        last_brace = text.rfind("}")
        while last_brace != -1:
            depth = 0
            for i in range(last_brace, -1, -1):
                if text[i] == "}":
                    depth += 1
                elif text[i] == "{":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:last_brace + 1].strip()
                        try:
                            extracted_obj = json.loads(candidate)
                            break
                        except Exception:
                            pass
            if extracted_obj:
                break
            last_brace = text.rfind("}", 0, last_brace)

    # 3. Direct parse attempt
    if not extracted_obj:
        try:
            extracted_obj = json.loads(text.strip())
        except Exception:
            pass

    # If extraction completely failed, return fallback entry with is_valid=False
    if not isinstance(extracted_obj, dict):
        return dict(fallback_entry), False

    # Validate and normalize required fields
    validated = {}

    # 1. severity
    raw_sev = str(extracted_obj.get("severity", fallback_entry.get("severity", "NORMAL"))).upper()
    if raw_sev not in ["NORMAL", "MINOR", "MODERATE", "CRITICAL"]:
        raw_sev = fallback_entry.get("severity", "NORMAL")
    validated["severity"] = raw_sev

    # 2. explanation
    validated["explanation"] = str(extracted_obj.get("explanation") or fallback_entry.get("explanation", ""))

    # 3. fault
    validated["fault"] = str(extracted_obj.get("fault") or fallback_entry.get("fault", ""))

    # 4. component
    validated["component"] = str(extracted_obj.get("component") or fallback_entry.get("component", ""))

    # 5. recommended_action
    validated["recommended_action"] = str(extracted_obj.get("recommended_action") or fallback_entry.get("recommended_action", ""))

    # 6. maintenance_action
    validated["maintenance_action"] = str(extracted_obj.get("maintenance_action") or fallback_entry.get("maintenance_action", ""))

    # 7. additional_checks
    raw_checks = extracted_obj.get("additional_checks")
    if isinstance(raw_checks, list):
        validated["additional_checks"] = [str(c).strip() for c in raw_checks if str(c).strip()]
    elif isinstance(raw_checks, str):
        validated["additional_checks"] = [c.strip() for c in raw_checks.split(",") if c.strip()]
    else:
        validated["additional_checks"] = list(fallback_entry.get("additional_checks", []))

    if not validated["additional_checks"]:
        validated["additional_checks"] = list(fallback_entry.get("additional_checks", []))

    return validated, True


def format_diagnostic_markdown(structured: Dict[str, Any], is_fallback: bool = False, error_msg: str = None) -> str:
    """Formats structured diagnostic output into clean industrial SCADA markdown report."""
    header = ""
    if is_fallback:
        header = f"> ⚠️ **[LOCAL RULE-BASED FALLBACK DIAGNOSIS]** {error_msg or 'Nebius API unavailable or invalid output'}\n\n"

    checks_md = "\n".join([f"- {c}" for c in structured.get("additional_checks", [])])

    return f"""{header}### 1. Executive Anomaly Assessment & Severity
**Severity Level:** **{structured['severity']}** | **Diagnosed Fault:** {structured['fault']}  
**Degraded Component:** `{structured['component']}`

### 2. Multi-Subsystem Physical Breakdown & Root Cause
{structured['explanation']}

### 3. Immediate Recommended Operational SOP
👉 **{structured['recommended_action']}**

### 4. Technician Maintenance SOP (Follow-Up)
🛠️ **{structured['maintenance_action']}**

### 5. Additional Diagnostic Verifications
{checks_md}"""


class NemotronDiagnosticEngine:
    """
    Communicates with NVIDIA Nemotron hosted on Nebius Token Factory API.
    Provides verifiable diagnostic reasoning with industrial engineering precision.
    Outputs structured JSON schema:
    severity, explanation, fault, component, recommended_action, maintenance_action, additional_checks.
    Includes robust fallback mechanism if API key is unset or service is unreachable.
    """
    def __init__(self):
        self.api_key = os.getenv("NEBIUS_API_KEY", "").strip()
        self.base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.eu-north1.nebius.com/v1/").strip()
        self.model = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning").strip()

    def diagnose(self, telemetry: Dict[str, float], ml_analysis: Dict[str, Any], scenario_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes AI diagnostic reasoning with end-to-end structured JSON parsing.
        Returns:
            - structured: Dict[str, Any] (validated fields)
            - diagnostic_text: str (Markdown report for UI)
            - source: str ("NEBIUS_NEMOTRON" or "LOCAL_FALLBACK")
            - status: str ("SUCCESS" or "FALLBACK")
            - model_used: str
            - is_fallback: bool
            - error_message: optional str
        """
        code = scenario_meta.get("code", "NORMAL")
        pred_code = ml_analysis.get("predicted_fault_code", code)
        fallback_entry = self._get_fallback_entry(pred_code, telemetry, ml_analysis, scenario_meta)

        # Check if API key is provided and valid format
        if not self.api_key or self.api_key == "your_nebius_api_key_here":
            return self._generate_local_fallback(
                fallback_entry,
                reason="Nebius Token Factory API key not configured in .env (NEBIUS_API_KEY is empty/placeholder)."
            )

        if not HAS_OPENAI:
            return self._generate_local_fallback(
                fallback_entry,
                reason="OpenAI Python SDK not installed in environment."
            )

        # Format diagnostic prompt context
        prompt = self._build_prompt(telemetry, ml_analysis, scenario_meta)

        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=35.0
            )

            system_instruction = (
                "You are the Senior Drill Rig Diagnostic Engineer & Physical AI Specialist operating DrillGuard AI. "
                "Analyze multi-subsystem industrial drill rig telemetry across Electrical, Mechanical, and Hydraulic domains. "
                "Return ONLY a single valid JSON object. Do not output any thinking process, preamble, or conversational text outside the JSON. "
                "The JSON must contain exactly these 7 keys:\n"
                "- severity: \"CRITICAL\" | \"MODERATE\" | \"MINOR\" | \"NORMAL\"\n"
                "- explanation: detailed physics explanation of the root cause based on sensor telemetry\n"
                "- fault: clear name of the diagnosed fault\n"
                "- component: degraded physical component\n"
                "- recommended_action: immediate operational action to mitigate or recover\n"
                "- maintenance_action: technician maintenance SOP\n"
                "- additional_checks: array of 3 specific physical checks\n"
                "Start directly with '{' and end with '}'."
            )

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2500,
            )

            raw_response = completion.choices[0].message.content.strip()

            # Parse and validate structured JSON
            structured_data, is_valid_json = extract_and_validate_json(raw_response, fallback_entry)

            if not is_valid_json:
                return self._generate_local_fallback(
                    fallback_entry,
                    reason="Nemotron returned non-JSON format; applied validated structured recovery fallback."
                )

            markdown_report = format_diagnostic_markdown(structured_data, is_fallback=False)

            return {
                "source": "NEBIUS_NEMOTRON",
                "status": "SUCCESS",
                "badge": f"🟢 NVIDIA Nemotron ({self.model}) via Nebius",
                "model_used": self.model,
                "structured": structured_data,
                "diagnostic_text": markdown_report,
                "raw_response": raw_response,
                "is_fallback": False,
                "error_message": None,
            }

        except Exception as e:
            # Under hackathon rules: NEVER claim an LLM call succeeded if it failed.
            formatted_err = format_nebius_error(e, self.model, self.base_url)
            return self._generate_local_fallback(
                fallback_entry,
                reason=formatted_err
            )

    def _build_prompt(self, telemetry: Dict[str, float], ml_analysis: Dict[str, Any], scenario_meta: Dict[str, Any]) -> str:
        """
        Builds the prompt passing actual sensor readings, detected subsystem, predicted fault,
        classifier probability, and evidence.
        """
        evidence_lines = [
            f"  * {d['name']} ({d['feature']}): {d['value']} {d['unit']} (Nominal: {d['nominal']} {d['unit']}, Deviation: Delta {d['deviation_pct']}%)"
            for d in ml_analysis.get("top_deviating_features", [])
        ]
        evidence_str = "\n".join(evidence_lines) if evidence_lines else "  * All parameters operating within normal variance"

        subsystem_category = scenario_meta.get("category", "General Rig Operations")
        predicted_fault = ml_analysis.get("predicted_fault_name", scenario_meta.get("name", "Unknown"))
        confidence_pct = ml_analysis.get("confidence", 1.0) * 100.0
        is_anom = ml_analysis.get("is_anomaly", False)
        anom_score = ml_analysis.get("anomaly_score", 0.0)

        prompt = f"""[DRILL RIG TELEMETRY INGESTION - SIMULATED TEST BENCH]

1. DETECTED SUBSYSTEM & FAULT CLASSIFICATION:
- Subsystem Domain: {subsystem_category}
- Isolation Forest Anomaly Detection: {'ANOMALY DETECTED' if is_anom else 'NOMINAL ENVELOPE'} (Severity Score: {anom_score:.3f})
- Random Forest Predicted Fault: {predicted_fault}
- Classifier Confidence: {confidence_pct:.1f}%

2. CLASSIFICATION EVIDENCE (TOP DRIFTING FEATURES):
{evidence_str}

3. ACTUAL TELEMETRY READINGS:
ELECTRICAL SUBSYSTEM:
- Bus Voltage: {telemetry['voltage_v']} V (Nominal: 480 V)
- Motor Current: {telemetry['current_a']} A (Nominal: 205 A)
- Bus Frequency: {telemetry['frequency_hz']} Hz (Nominal: 60.0 Hz)
- Power Factor: {telemetry['power_factor']} (Nominal: 0.91)
- VFD Inverter Temp: {telemetry['vfd_temp_c']} °C (Nominal: 52 °C)
- Active Power: {telemetry['active_power_kw']} kW

MECHANICAL SUBSYSTEM:
- Top-Drive Speed: {telemetry['rpm']} RPM (Nominal: 135 RPM)
- Drill String Torque: {telemetry['torque_knm']} kNm (Nominal: 21.5 kNm)
- Vibration RMS: {telemetry['vibration_rms_mms']} mm/s (Nominal: 2.4 mm/s, ISO Danger > 7.1 mm/s)
- Vibration Peak: {telemetry['vibration_peak_mms']} mm/s (Nominal: 3.8 mm/s)
- Main Thrust Bearing Temp: {telemetry['bearing_temp_c']} °C (Nominal: 64 °C)
- Gearbox Oil Temp: {telemetry['gearbox_oil_temp_c']} °C (Nominal: 68 °C)

HYDRAULIC SUBSYSTEM:
- Main Hydraulic Pressure: {telemetry['hydraulic_pressure_bar']} bar (Nominal: 200 bar)
- Flow Rate: {telemetry['flow_rate_lpm']} L/min (Nominal: 380 L/min)
- Return Filter dP: {telemetry['differential_pressure_bar']} bar (Nominal: 1.4 bar, Trip Limit: 5.0 bar)
- Hydraulic Fluid Temp: {telemetry['fluid_temp_c']} °C (Nominal: 54 °C)
- Pump Speed: {telemetry['pump_rpm']} RPM (Nominal: 1500 RPM)
- Reservoir Level: {telemetry['reservoir_level_pct']} % (Nominal: 88 %)

Analyze this rig telemetry and provide your diagnosis in strict JSON format:
{{
  "severity": "CRITICAL" | "MODERATE" | "MINOR" | "NORMAL",
  "explanation": "Deep physical explanation of root cause based on sensor readings",
  "fault": "{predicted_fault}",
  "component": "Specific physical component degraded",
  "recommended_action": "Immediate operational action to mitigate or recover",
  "maintenance_action": "Follow-up technician maintenance SOP",
  "additional_checks": ["Check 1", "Check 2", "Check 3"]
}}"""
        return prompt

    def _get_fallback_entry(self, pred_code: str, telemetry: Dict[str, float], ml_analysis: Dict[str, Any], scenario_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Provides structured knowledge base entries with all 7 fields for each scenario."""
        kb = {
            "NORMAL": {
                "severity": "NORMAL",
                "fault": "Nominal Healthy Drilling Operation",
                "component": "Integrated Subsystems (Electrical VFD, Top-Drive Swivel, Hydraulic Main Loop)",
                "explanation": (
                    "All three subsystems (Electrical, Mechanical, Hydraulic) are operating synchronously within nominal operating tolerances. "
                    f"Bus voltage is steady at {telemetry['voltage_v']}V, top-drive vibration RMS is {telemetry['vibration_rms_mms']} mm/s "
                    f"(well below the 7.1 mm/s ISO danger threshold), and hydraulic loop maintains healthy {telemetry['hydraulic_pressure_bar']} bar pressure."
                ),
                "recommended_action": "Maintain automated drilling program parameters and continue continuous background SCADA telemetry logging.",
                "maintenance_action": "Perform standard 250-hour lubrication rounds and oil sample collection at the next planned bit change.",
                "additional_checks": [
                    "Verify MWD mud pulse telemetry carrier frequency",
                    "Check top-drive counterbalance cylinder pressure",
                    "Log generator bus voltage harmonic distortion"
                ]
            },
            "E1": {
                "severity": "CRITICAL",
                "fault": "E1: Electrical Overcurrent",
                "component": "VFD Inverter Bridge & Stator Windings",
                "explanation": (
                    f"Severe stator phase overcurrent surge ({telemetry['current_a']} A vs nominal 205 A, +{(telemetry['current_a']-205)/205*100:.1f}% overload) "
                    f"causing rapid thermal escalation across VFD IGBT junction modules ({telemetry['vfd_temp_c']} °C). Immediate risk of IGBT thermal breakdown."
                ),
                "recommended_action": "Clamp VFD inverter current ceiling to 210A, throttle drill rotary torque by 30%, and cycle auxiliary cabinet forced-air blowers to maximum.",
                "maintenance_action": "Conduct megohmmeter insulation resistance test on top-drive motor phases and inspect VFD IGBT cooling fin airflow.",
                "additional_checks": [
                    "Measure phase current balance across U, V, W terminals",
                    "Check VFD DC bus ripple voltage",
                    "Inspect cabinet ventilation filters for blockage"
                ]
            },
            "E2": {
                "severity": "MODERATE",
                "fault": "E2: Voltage Instability & Bus Hunting",
                "component": "Rig Generator AVR & Transformer Tap Changer",
                "explanation": (
                    f"Generator 3-phase bus voltage fluctuating erratically ({telemetry['voltage_v']} V vs nominal 480 V) "
                    f"with frequency hunting ({telemetry['frequency_hz']} Hz) and degraded power factor ({telemetry['power_factor']}), induced by AVR governor hunting."
                ),
                "recommended_action": "Switch rig bus transformer tap changer to automated regulation mode and engage active dynamic line reactors.",
                "maintenance_action": "Calibrate generator Automatic Voltage Regulator (AVR) governor damping gains and service bus synchronizer contacts.",
                "additional_checks": [
                    "Inspect generator exciter diode bridge",
                    "Check SCR drive harmonic filters",
                    "Verify load sharing across online diesel generator sets"
                ]
            },
            "E3": {
                "severity": "CRITICAL",
                "fault": "E3: Motor Abnormality / Slip Surge",
                "component": "Induction Motor Stator Windings & Vector Encoder",
                "explanation": (
                    f"Top-drive motor stator winding asymmetry and excessive rotor slip (RPM dropped to {telemetry['rpm']} RPM with severe {telemetry['vfd_temp_c']} °C thermal rise) "
                    "indicating inter-turn insulation degradation or encoder desynchronization producing negative-sequence braking torque."
                ),
                "recommended_action": "Derate motor duty cycle by 40%, re-zero vector encoder pulse offset, and rebalance PWM carrier frequency.",
                "maintenance_action": "Perform motor surge testing, thermographic stator scan, and replace rotary pulse encoder coupling.",
                "additional_checks": [
                    "Check optical encoder pulse alignment",
                    "Perform motor winding resistance bridge test",
                    "Verify resolver cable shielding against EMI noise"
                ]
            },
            "M1": {
                "severity": "CRITICAL",
                "fault": "M1: Severe Drill-String Whirl & Stick-Slip Resonance",
                "component": "Bottom-Hole Assembly (BHA) & Top-Drive Quill",
                "explanation": (
                    f"Extreme drill string lateral whirl and torsional stick-slip oscillation. Vibration RMS spiked to {telemetry['vibration_rms_mms']} mm/s "
                    f"(ISO Class IV Danger > 7.1 mm/s) with peak shock impacts of {telemetry['vibration_peak_mms']} mm/s, threatening downhole tool joint twist-off."
                ),
                "recommended_action": "Activate Soft-Torque rotary feedback control, reduce Weight-On-Bit (WOB) by 25%, and elevate rotary speed by 15 RPM to break resonance.",
                "maintenance_action": "Inspect drill collar tool joints and shock subs for micro-fractures, and gauge stabilizer blade wear.",
                "additional_checks": [
                    "Download MWD downhole accelerometer memory logs",
                    "Perform ultrasonic NDT on top-drive quill stem",
                    "Verify rotary table torque sensor calibration"
                ]
            },
            "M2": {
                "severity": "MODERATE",
                "fault": "M2: Bearing Degradation",
                "component": "Top-Drive Main Thrust Spherical Roller Bearing",
                "explanation": (
                    f"Thrust bearing temperature elevated to {telemetry['bearing_temp_c']} °C with sharp crest factor shock impacts "
                    f"({telemetry['vibration_peak_mms'] / (telemetry['vibration_rms_mms'] + 1e-4):.2f}) indicating raceway micro-spalling and rolling element fatigue."
                ),
                "recommended_action": "Trigger automated high-pressure synthetic grease purging cycle and cap maximum top-drive RPM to 110.",
                "maintenance_action": "Draw bearing lubricant grease sample for ferrographic iron particulate analysis and schedule bearing changeout.",
                "additional_checks": [
                    "Perform ultrasonic high-frequency shock pulse measurement (SPM)",
                    "Inspect bearing seal lip for leakage",
                    "Check axial play with dial indicator"
                ]
            },
            "M3": {
                "severity": "MODERATE",
                "fault": "M3: Planetary Gearbox Overheating",
                "component": "Top-Drive Planetary Gear Reducer",
                "explanation": (
                    f"Gearbox oil temperature elevated to {telemetry['gearbox_oil_temp_c']} °C with gear mesh chatter "
                    "indicating accelerated boundary friction, localized tooth flank pitting, and lubricant viscosity breakdown."
                ),
                "recommended_action": "Engage auxiliary lube cooling radiator, circulate secondary offline micro-filtration loop, and limit drill torque slew rate to 2.0 kNm/sec.",
                "maintenance_action": "Drain and replace ISO VG 320 synthetic gear lubricant and inspect magnetic drain plugs for metallic debris.",
                "additional_checks": [
                    "Borescope inspection of sun and planet gear tooth flanks",
                    "Test oil viscosity and acid number (TAN)",
                    "Verify lube circulating pump pressure"
                ]
            },
            "H1": {
                "severity": "MODERATE",
                "fault": "H1: Hydraulic Pump Volumetric Degradation",
                "component": "Primary Axial Piston Swashplate Pump",
                "explanation": (
                    f"Main hydraulic pressure decayed to {telemetry['hydraulic_pressure_bar']} bar and flow collapsed to {telemetry['flow_rate_lpm']} L/min "
                    f"despite nominal pump speed ({telemetry['pump_rpm']} RPM) caused by internal swashplate leakage discharging fluid back to case drain."
                ),
                "recommended_action": "Transfer hydraulic circuit load to redundant standby Pump-B and unload primary pump swashplate angle.",
                "maintenance_action": "Remove primary pump for bench tear-down, replace worn bronze slipper pads and lapped valve plate.",
                "additional_checks": [
                    "Measure pump case drain leakage flow rate",
                    "Check suction strainer for particulate ingress",
                    "Inspect proportional displacement control valve"
                ]
            },
            "H2": {
                "severity": "MODERATE",
                "fault": "H2: Return Line Filter Blockage",
                "component": "Dual Duplex Return Micron Filter Element",
                "explanation": (
                    f"Severe differential pressure surge across return-line filter ({telemetry['differential_pressure_bar']} bar vs nominal 1.4 bar, "
                    f"approaching the 5.0 bar bypass limit) with elevated upstream backpressure ({telemetry['hydraulic_pressure_bar']} bar) creating cavitation risk."
                ),
                "recommended_action": "Rotate duplex filter selector valve to online clean standby cartridge and initiate automatic backflush sequence.",
                "maintenance_action": "Replace blinded 10-micron glass fiber filter element and perform patch test on hydraulic fluid contamination (ISO 4406).",
                "additional_checks": [
                    "Verify differential pressure visual indicator pop-up",
                    "Inspect bypass valve spring tension",
                    "Test return manifold check valves"
                ]
            },
            "H3": {
                "severity": "CRITICAL",
                "fault": "H3: Severe Hydraulic Pressure Collapse",
                "component": "High-Pressure Distribution Manifold & Bladder Accumulator",
                "explanation": (
                    f"Catastrophic hydraulic pressure collapse down to {telemetry['hydraulic_pressure_bar']} bar with reservoir fluid loss "
                    f"down to {telemetry['reservoir_level_pct']} % indicating high-pressure manifold breach or blown relief seal."
                ),
                "recommended_action": "Isolate high-pressure supply manifold, engage safety accumulator emergency hold, and trip rig hydraulic safety stop.",
                "maintenance_action": "Inspect high-pressure flexible hoses for burst, replace manifold O-rings, and pressure-test circuit with inert nitrogen.",
                "additional_checks": [
                    "Check fluid containment sump for environmental spill",
                    "Verify accumulator pre-charge nitrogen pressure",
                    "Inspect counterbalance cylinder check valves"
                ]
            },
        }
        return kb.get(pred_code, kb["NORMAL"])

    def _generate_local_fallback(self, fallback_entry: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Physics-grounded deterministic rule-based diagnostic fallback."""
        markdown_report = format_diagnostic_markdown(fallback_entry, is_fallback=True, error_msg=reason)

        return {
            "source": "LOCAL_FALLBACK",
            "status": "FALLBACK",
            "badge": "🟠 Local Rule-Based Fallback Engine",
            "model_used": "Embedded Physical Knowledge Base (Local Rule Engine)",
            "structured": fallback_entry,
            "diagnostic_text": markdown_report,
            "raw_response": None,
            "is_fallback": True,
            "error_message": reason,
        }


def test_nemotron_connection(diagnostic_question: str = None) -> Dict[str, Any]:
    """
    Sends a test physical diagnostic query to NVIDIA Nemotron on Nebius Token Factory.
    Safely reads credentials from .env, validates the connection, and measures latency.
    Never exposes secret API keys.
    """
    if not HAS_OPENAI:
        return {
            "success": False,
            "error": "OpenAI Python SDK is not installed in the current environment.",
            "source": "SDK_ERROR",
            "model": None,
            "base_url": None,
            "question": diagnostic_question,
            "response": None,
            "latency_sec": 0.0,
        }

    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.eu-north1.nebius.com/v1/").strip()
    model = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning").strip()

    if not api_key or api_key == "your_nebius_api_key_here":
        return {
            "success": False,
            "error": "NEBIUS_API_KEY is missing or still set to placeholder in .env.",
            "source": "CONFIG_ERROR",
            "model": model,
            "base_url": base_url,
            "question": diagnostic_question,
            "response": None,
            "latency_sec": 0.0,
        }

    question = diagnostic_question or (
        "Rig telemetry test: Top-drive vibration RMS reached 19.2 mm/s with stick-slip chatter. "
        "Provide immediate physical diagnosis and primary mitigation in 2 concise sentences."
    )

    t0 = time.time()
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=25.0)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Senior Drill Rig Diagnostic Engineer & Physical AI Specialist."},
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        latency = round(time.time() - t0, 2)
        response_text = completion.choices[0].message.content.strip()

        return {
            "success": True,
            "error": None,
            "source": "NEBIUS_NEMOTRON",
            "model": model,
            "base_url": base_url,
            "question": question,
            "response": response_text,
            "latency_sec": latency,
        }
    except Exception as e:
        latency = round(time.time() - t0, 2)
        err_msg = format_nebius_error(e, model, base_url)
        return {
            "success": False,
            "error": err_msg,
            "source": "API_ERROR",
            "model": model,
            "base_url": base_url,
            "question": question,
            "response": None,
            "latency_sec": latency,
        }


# ---------------------------------------------------------
# SIMULATED MACHINE RESPONSE TO CORRECTIVE ACTION
# ---------------------------------------------------------

def simulate_machine_response(scenario_code: str, steps: int = 6) -> List[Dict[str, Any]]:
    """
    Simulates the closed-loop recovery trajectory of the drill rig after the user
    clicks 'Apply Simulated Corrective Action'. Demonstrates how physical telemetry
    re-stabilizes from the fault state back toward nominal baseline.
    """
    if scenario_code == "NORMAL":
        # Normal machine remains stable
        t_pts = []
        for i in range(steps):
            pt = generate_telemetry_point("NORMAL", t_sec=i * 2.0, noise_mult=0.5)
            pt_feat = compute_derived_features(pt)
            pt_feat["step"] = i
            pt_feat["step_label"] = f"t+{i*2}s (Steady)"
            t_pts.append(pt_feat)
        return t_pts

    # Generate initial fault baseline
    fault_pt = generate_telemetry_point(scenario_code, t_sec=0.0, noise_mult=0.3)
    norm_pt = generate_telemetry_point("NORMAL", t_sec=100.0, noise_mult=0.3)

    trajectory = []
    for step in range(steps):
        # Nonlinear exponential recovery curve: alpha goes from 0.0 (fault) to 1.0 (recovered)
        alpha = 1.0 - math.exp(-step / 1.5)
        interpolated = {}
        for key in RAW_SENSOR_NAMES:
            val_start = fault_pt[key]
            val_target = norm_pt[key]
            # Add decreasing noise as system stabilizes
            residual_noise = np.random.normal(0, 0.02 * (1.0 - alpha) * val_target)
            interpolated[key] = round(float(val_start + alpha * (val_target - val_start) + residual_noise), 2)

        feat = compute_derived_features(interpolated)
        feat["step"] = step
        feat["step_label"] = f"t+{step*3}s" if step > 0 else "t=0s (Action Triggered)"
        feat["recovery_pct"] = round(alpha * 100.0, 1)
        trajectory.append(feat)

    return trajectory

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
    from openai import OpenAI
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

class NemotronDiagnosticEngine:
    """
    Communicates with NVIDIA Nemotron hosted on Nebius Token Factory API.
    Provides verifiable diagnostic reasoning with industrial engineering precision.
    Includes robust fallback mechanism if API key is unset or service is unreachable.
    """
    def __init__(self):
        self.api_key = os.getenv("NEBIUS_API_KEY", "").strip()
        self.base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/").strip()
        self.model = os.getenv("NEBIUS_MODEL", "nvidia/Llama-3_1-Nemotron-70B-Instruct-HF").strip()

    def diagnose(self, telemetry: Dict[str, float], ml_analysis: Dict[str, Any], scenario_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes AI diagnostic reasoning.
        Returns:
            - diagnostic_text: str (Structured analysis)
            - source: str ("NEBIUS_NEMOTRON" or "LOCAL_FALLBACK")
            - status: str ("SUCCESS" or "FALLBACK")
            - model_used: str
            - raw_response: optional str
        """
        # Format diagnostic prompt context
        prompt = self._build_prompt(telemetry, ml_analysis, scenario_meta)

        # Check if API key is provided and valid format
        if not self.api_key or self.api_key == "your_nebius_api_key_here":
            return self._generate_local_fallback(
                telemetry, ml_analysis, scenario_meta,
                reason="Nebius Token Factory API key not configured in .env (NEBIUS_API_KEY is empty/placeholder)."
            )

        if not HAS_OPENAI:
            return self._generate_local_fallback(
                telemetry, ml_analysis, scenario_meta,
                reason="OpenAI Python SDK not installed in environment."
            )

        try:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=25.0
            )

            system_instruction = (
                "You are the Senior Drill Rig Diagnostic Engineer & Physical AI Specialist operating DrillGuard AI. "
                "Analyze multi-subsystem industrial drill rig telemetry across Electrical, Mechanical, and Hydraulic domains. "
                "Provide rigorous physics-grounded root cause diagnosis, failure progression horizons, and decisive corrective SOPs. "
                "Clearly address the physical phenomena observed in the sensor readings. "
                "Format your answer with clear markdown headings:\n"
                "### 1. Executive Anomaly Assessment & Severity\n"
                "### 2. Multi-Subsystem Physical Telemetry Breakdown\n"
                "### 3. Root Cause Physical Hypothesis\n"
                "### 4. Component Degradation Horizon & Operational Risk\n"
                "### 5. Recommended Corrective Action SOP\n"
                "### 6. Predicted Machine Response Trajectory"
            )

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=900,
            )

            response_text = completion.choices[0].message.content.strip()

            return {
                "source": "NEBIUS_NEMOTRON",
                "status": "SUCCESS",
                "badge": "🟢 NVIDIA Nemotron via Nebius Token Factory",
                "model_used": self.model,
                "diagnostic_text": response_text,
                "is_fallback": False,
                "error_message": None,
            }

        except Exception as e:
            # Under hackathon rules: NEVER claim an LLM call succeeded if it failed.
            error_str = str(e)
            return self._generate_local_fallback(
                telemetry, ml_analysis, scenario_meta,
                reason=f"Nebius Token Factory API request failed: {error_str}"
            )

    def _build_prompt(self, telemetry: Dict[str, float], ml_analysis: Dict[str, Any], scenario_meta: Dict[str, Any]) -> str:
        dev_str = "\n".join([
            f"- {d['name']} ({d['feature']}): {d['value']} {d['unit']} (Nominal: {d['nominal']} {d['unit']}, Δ {d['deviation_pct']}%)"
            for d in ml_analysis.get("top_deviating_features", [])
        ])

        prompt = f"""[DRILL RIG TELEMETRY INGESTION - SIMULATED TEST BENCH]
Subsystem Telemetry Snapshot:
ELECTRICAL SUBSYSTEM:
- Bus Voltage: {telemetry['voltage_v']} V (Nominal: 480 V)
- Motor Current: {telemetry['current_a']} A (Nominal: 205 A)
- Bus Frequency: {telemetry['frequency_hz']} Hz (Nominal: 60.0 Hz)
- Power Factor: {telemetry['power_factor']} (Nominal: 0.91)
- VFD Inverter Temp: {telemetry['vfd_temp_c']} °C (Nominal: 52 °C)
- Active Power: {telemetry['active_power_kw']} kW

MECHANICAL SUBSYSTEM:
- Top-Drive Speed: {telemetry['rpm']} RPM (Nominal: 135 RPM)
- Drill Torque: {telemetry['torque_knm']} kNm (Nominal: 21.5 kNm)
- Vibration RMS: {telemetry['vibration_rms_mms']} mm/s (Nominal: 2.4 mm/s, ISO Critical > 7.1)
- Vibration Peak: {telemetry['vibration_peak_mms']} mm/s (Nominal: 3.8 mm/s)
- Thrust Bearing Temp: {telemetry['bearing_temp_c']} °C (Nominal: 64 °C)
- Gearbox Lube Temp: {telemetry['gearbox_oil_temp_c']} °C (Nominal: 68 °C)

HYDRAULIC SUBSYSTEM:
- Main Hydraulic Pressure: {telemetry['hydraulic_pressure_bar']} bar (Nominal: 200 bar)
- Flow Rate: {telemetry['flow_rate_lpm']} L/min (Nominal: 380 L/min)
- Return Filter dP: {telemetry['differential_pressure_bar']} bar (Nominal: 1.4 bar, Trip Limit: 5.0 bar)
- Hydraulic Fluid Temp: {telemetry['fluid_temp_c']} °C (Nominal: 54 °C)
- Pump Speed: {telemetry['pump_rpm']} RPM (Nominal: 1500 RPM)
- Reservoir Level: {telemetry['reservoir_level_pct']} % (Nominal: 88 %)

ML ENSEMBLE DIAGNOSTIC METRICS:
- Isolation Forest Anomaly Flag: {'ANOMALOUS' if ml_analysis['is_anomaly'] else 'NOMINAL'} (Score: {ml_analysis['anomaly_score']})
- Random Forest Fault Classification: {ml_analysis['predicted_fault_name']} (Confidence: {ml_analysis['confidence']*100:.1f}%)
- Pre-configured Test Scenario: {scenario_meta['code']} - {scenario_meta['name']}
- Top Physics Deviations from Nominal:
{dev_str}

Please generate an exhaustive physical AI diagnostic evaluation including root cause, subsystem interaction physics, immediate operational risk, and the exact corrective action SOP."""
        return prompt

    def _generate_local_fallback(self, telemetry: Dict[str, float], ml_analysis: Dict[str, Any], scenario_meta: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Physics-grounded deterministic rule-based diagnostic engine.
        Clearly labelled as local fallback to guarantee transparency and compliance.
        """
        code = scenario_meta.get("code", "NORMAL")
        pred_code = ml_analysis.get("predicted_fault_code", code)
        severity = scenario_meta.get("severity", "NORMAL")

        # Specific tailored physical diagnostic reasoning per fault code
        kb = {
            "NORMAL": {
                "summary": "All three subsystems (Electrical, Mechanical, Hydraulic) are operating synchronously within nominal operating tolerances. No thermal, harmonic, or pressure deviations detected.",
                "root_cause": "Steady-state drilling operation with balanced rock-bit interaction and nominal lubrication dynamics.",
                "subsystems": "Electrical bus voltage & power factor are stable. Mechanical vibration spectra conform to ISO 10816 Class-I permissible levels (< 2.8 mm/s). Hydraulic loop maintains healthy 200 bar pressure and differential filter loss is clean at 1.4 bar.",
                "horizon": "Indefinite nominal runtime. Next scheduled preventative maintenance window in 250 operating hours.",
                "action": "Maintain automated drilling program parameters. Continue background telemetry acquisition and ML envelope monitoring.",
                "response": "Steady baseline telemetry. Zero corrective dampening required."
            },
            "E1": {
                "summary": f"CRITICAL ELECTRICAL ANOMALY: Severe overcurrent event ({telemetry['current_a']} A vs nominal 205 A, +{(telemetry['current_a']-205)/205*100:.1f}% overload). Rapid thermal buildup in VFD inverter bridge ({telemetry['vfd_temp_c']} °C).",
                "root_cause": "Stator field saturation or transient rotary lockup causing VFD inverter bridge to deliver near short-circuit commutating current, driving thermal stress across IGBT modules.",
                "subsystems": "Electrical: Severe current surge and bus sag. Mechanical: High magnetic counter-torque on top-drive. Hydraulic: Passive baseline unaffected.",
                "horizon": "Immediate risk of catastrophic IGBT thermal breakdown within 45 to 90 seconds if current clamping is not deployed.",
                "action": "Clamp VFD inverter current ceiling to 210 A, throttle rotary torque command by 30%, and cycle auxiliary cabinet forced-air blowers to maximum duty.",
                "response": "Phase current decays to < 215 A within 3.5 seconds; VFD heat sink temperature stabilizes and decays below 65 °C."
            },
            "E2": {
                "summary": f"MODERATE ELECTRICAL INSTABILITY: Generator bus voltage fluctuating violently ({telemetry['voltage_v']} V vs nominal 480 V) with frequency hunting ({telemetry['frequency_hz']} Hz) and degraded power factor ({telemetry['power_factor']}).",
                "root_cause": "Rig auxiliary generator automatic voltage regulator (AVR) governor oscillation and reactive power load imbalance across SCR drives.",
                "subsystems": "Electrical: Large voltage ripple and low displacement factor. Mechanical: Minor torsional speed ripple induced by fluctuating electromagnetic torque.",
                "horizon": "Electronic control card trips or motor insulation dielectric fatigue within 2 to 4 hours of unmitigated harmonic cycling.",
                "action": "Switch rig bus transformer tap changer to automated regulation mode, engage dynamic line reactors, and trim AVR governor damping gain.",
                "response": "3-phase bus voltage stabilizes within ±1.5% of 480 V; line frequency stabilizes at 60.0 ± 0.08 Hz."
            },
            "E3": {
                "summary": f"CRITICAL ELECTRICAL / MOTOR FAULT: Top-drive motor stator winding asymmetry and excessive rotor slip (RPM dropped to {telemetry['rpm']} RPM with severe {telemetry['vfd_temp_c']} °C thermal rise).",
                "root_cause": "Inter-turn insulation breakdown in Stator Phase-B winding or VFD inverter PWM gate driver desynchronization, generating intense negative-sequence braking torque.",
                "subsystems": "Electrical: Inefficient electrical-to-mechanical conversion. Mechanical: Substantial rotary speed loss despite high electrical input power. Hydraulic: Normal.",
                "horizon": "Complete stator phase burnout within 15 to 30 minutes of continuous high-load drilling.",
                "action": "Derate motor duty cycle by 40%, re-zero vector encoder offset, rebalance PWM carrier frequency, and plan immediate thermographic motor inspection.",
                "response": "Rotor slip returns to < 2.5%; winding phase currents balance within 3% tolerance and stator temperatures normalize."
            },
            "M1": {
                "summary": f"CRITICAL MECHANICAL VIBRATION: Severe drill string whirl and stick-slip resonance. Vibration RMS at {telemetry['vibration_rms_mms']} mm/s (ISO Class IV Danger > 7.1 mm/s) with peak spikes to {telemetry['vibration_peak_mms']} mm/s.",
                "root_cause": "BHA (Bottom Hole Assembly) lateral resonance coupled with dynamic bit-rock friction stick-slip, inducing high torsional deceleration shockwaves up the drill stem.",
                "subsystems": "Mechanical: Catastrophic cyclic fatigue loading on top-drive quill and swivel stem. Electrical: Torque chatter reflected into motor current ripple.",
                "horizon": "Downhole tool joint twist-off or top-drive swivel bearing destruction within 20 to 60 minutes.",
                "action": "Activate automated Soft-Torque rotary feedback control, reduce Weight-On-Bit (WOB) by 25%, and elevate rotary speed by 15 RPM to break lateral resonance.",
                "response": "Vibration RMS attenuates from >18 mm/s down to < 3.2 mm/s in under 6 seconds; smooth bit rotation restored."
            },
            "M2": {
                "summary": f"MODERATE MECHANICAL FAULT: Top-drive main thrust bearing degradation. Bearing temperature elevated to {telemetry['bearing_temp_c']} °C with sharp crest factor shock impacts ({telemetry['vibration_peak_mms'] / (telemetry['vibration_rms_mms'] + 1e-4):.2f}).",
                "root_cause": "Inner raceway micro-spalling and rolling element surface fatigue caused by axial shock loads during high-impact reaming operations.",
                "subsystems": "Mechanical: Localized heat accumulation in thrust bearing housing with acoustic ultrasonic shock pulses. Electrical: Modest frictional drag increase.",
                "horizon": "Accelerated spall propagation leading to complete bearing seizure in approximately 12 to 24 operating hours.",
                "action": "Trigger automated high-pressure synthetic grease purging cycle, cap maximum top-drive RPM to 110, and monitor high-frequency vibration envelope.",
                "response": "Lubrication hydrodynamics re-established; bearing temperature plateaus and stabilizes below 72 °C with dampened shock peaks."
            },
            "M3": {
                "summary": f"MODERATE MECHANICAL FAULT: Top-drive planetary gearbox overheating ({telemetry['gearbox_oil_temp_c']} °C) with gear mesh acoustic chatter and mechanical efficiency reduction.",
                "root_cause": "Viscosity breakdown of synthetic gear lubricant combined with localized tooth flank pitting on sun/planet gear contact faces.",
                "subsystems": "Mechanical: Excessive boundary friction in gearbox casing generating accelerated thermal load. Hydraulic/Electrical: Secondary lube cooling circuit strained.",
                "horizon": "Gear tooth spalling or planetary carrier seizure within 8 to 16 hours of continuous full-torque drilling.",
                "action": "Engage auxiliary lube cooling radiator, circulate secondary offline micro-filtration loop, and limit drill torque slew rate to 2.0 kNm/sec.",
                "response": "Lube oil temperature stabilizes and gradually declines back to 68 °C; gear mesh vibration amplitudes diminish by 45%."
            },
            "H1": {
                "summary": f"MODERATE HYDRAULIC FAULT: Main pump volumetric degradation. Pressure decayed to {telemetry['hydraulic_pressure_bar']} bar and flow collapsed to {telemetry['flow_rate_lpm']} L/min despite nominal pump speed ({telemetry['pump_rpm']} RPM).",
                "root_cause": "Axial piston slipper wear and excessive internal clearance leakage across the swashplate valve plate, discharging pressurized fluid directly back to case drain.",
                "subsystems": "Hydraulic: Severe loss of volumetric efficiency with high fluid friction heating ({telemetry['fluid_temp_c']} °C). Mechanical/Electrical: Unaffected.",
                "horizon": "Total hydraulic actuator stall or complete top-drive pipe handler clamp failure within 2 to 6 hours.",
                "action": "Transfer hydraulic circuit load to redundant standby Pump-B, unload primary pump swashplate angle, and tag primary pump for bench overhaul.",
                "response": "Circuit pressure recovers immediately to 200 bar and flow rate restores to 380 L/min via Pump-B."
            },
            "H2": {
                "summary": f"MODERATE HYDRAULIC ANOMALY: Severe differential pressure surge across return-line micron filter ({telemetry['differential_pressure_bar']} bar vs nominal 1.4 bar, +{(telemetry['differential_pressure_bar']-1.4)/1.4*100:.0f}% increase).",
                "root_cause": "Heavy particulate contamination and filter media blinding, restricting fluid return circulation and approaching internal bypass valve opening limit (5.0 bar).",
                "subsystems": "Hydraulic: Dangerous backpressure on return manifold ({telemetry['hydraulic_pressure_bar']} bar) creating risk of pump suction cavitation and seal extrusion.",
                "horizon": "Filter bypass valve will burst open within 15 to 45 minutes, circulating unfiltered abrasive particulates into sensitive proportional servo valves.",
                "action": "Rotate manual duplex filter selector valve to online clean standby cartridge and initiate automatic ultrasonic backflush cycle.",
                "response": "Differential pressure across filter element plummets from >7.5 bar to 1.3 bar in under 2 seconds; return line backpressure normalizes."
            },
            "H3": {
                "summary": f"CRITICAL HYDRAULIC COLLAPSE: Catastrophic hydraulic pressure loss ({telemetry['hydraulic_pressure_bar']} bar vs nominal 200 bar) accompanied by reservoir depletion ({telemetry['reservoir_level_pct']} %).",
                "root_cause": "Major high-pressure manifold flange seal failure or stuck-open main relief valve dumping fluid directly into tank or external containment.",
                "subsystems": "Hydraulic: Complete loss of hydraulic actuation power for top-drive counter-balance, drill elevators, and iron roughneck clamping jaws.",
                "horizon": "Immediate safety hazard. Equipment slippage or dropped drill-string risk within seconds if counter-balance check valves fail.",
                "action": "Isolate high-pressure supply manifold, engage safety accumulator emergency hold, and trip rig hydraulic safety stop.",
                "response": "Accumulator locks load safety circuits at 185 bar holding pressure; fluid loss halts and emergency containment secured."
            },
        }

        entry = kb.get(pred_code, kb["NORMAL"])

        fallback_text = f"""[LOCAL RULE-BASED FALLBACK DIAGNOSIS - Nebius Token Factory API unavailable or no API key configured]

### 1. Executive Anomaly Assessment & Severity
**Classification:** {SCENARIOS.get(pred_code, {}).get('name', pred_code)}
**Severity Level:** **{severity}** | **Isolation Forest Anomaly Score:** {ml_analysis.get('anomaly_score', 0.0)}
**Summary:** {entry['summary']}

### 2. Multi-Subsystem Physical Telemetry Breakdown
{entry['subsystems']}

### 3. Root Cause Physical Hypothesis
{entry['root_cause']}

### 4. Component Degradation Horizon & Operational Risk
**Estimated Failure Window:** {entry['horizon']}

### 5. Recommended Corrective Action SOP
**Action:** {entry['action']}

### 6. Predicted Machine Response Trajectory
{entry['response']}

---
*Notice: This diagnostic report was generated by DrillGuard AI's embedded physical physics engine as a reliable local fallback because {reason}.*"""

        return {
            "source": "LOCAL_FALLBACK",
            "status": "FALLBACK",
            "badge": "🟠 Local Rule-Based Fallback Engine",
            "model_used": "Embedded Physical Knowledge Base (Local Rule Engine)",
            "diagnostic_text": fallback_text,
            "is_fallback": True,
            "error_message": reason,
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

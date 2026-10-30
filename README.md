# 🛡️ DrillGuard AI: Cyber-Physical Drill Rig Diagnostic SCADA

> **Built for the Nebius x NVIDIA Global AI Hackathon 2026 — Physical AI Track**  
> *End-to-End Software-Only Drill Rig Subsystem Telemetry, Dual-Tier ML Anomaly Detection, NVIDIA Nemotron Diagnostic Reasoning via Nebius Token Factory, and Closed-Loop Recovery Simulation.*

---

## 🎯 Project Objective

DrillGuard AI is a software-only physical AI diagnostic and control-room application simulating an industrial rotary drilling rig (`RIG-ALPHA-07`). It monitors, detects, classifies, diagnoses, and simulates corrective actions across **three coupled industrial subsystems**:

1. **⚡ Electrical Subsystem**: Variable Frequency Drives (VFD), 3-phase bus voltage, phase currents, grid frequency, power factor, and inverter junction temperatures.
2. **⚙️ Mechanical Subsystem**: Top-drive rotary speed (RPM), drill-string torque, multi-axis vibration RMS and peak shock acceleration, main thrust bearing temperatures, and planetary gearbox oil thermals.
3. **💧 Hydraulic Subsystem**: Main power loop pressure, flow rates, return-line micron filter differential pressure ($dP$), fluid temperature, pump RPM, and reservoir volume.

---

## 🔄 Core Workflow Pipeline

```
Synthetic Correlated Telemetry (Electrical + Mechanical + Hydraulic)
                        │
                        ▼
       Domain-Specific Physical Feature Engineering
         (Apparent Power, Crest Factor, Motor Efficiency, 
          Hydraulic Power, Filter Clog Ratio, Thermal Headroom)
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
Scikit-Learn Isolation Forest  Scikit-Learn Random Forest
   (Unsupervised Anomaly          (Supervised 10-Class
     Detection & Scoring)         Fault Classification)
            │                       │
            └───────────┬───────────┘
                        ▼
      Nebius Token Factory API (Studio)
                        │
                        ▼
   NVIDIA Nemotron Diagnostic Reasoning
     (Subsystem Physics, Root Cause, Failure Horizon,
      Severity Assessment & SOP Recommendation)
                        │
                        ▼
       Interactive Corrective Action Execution
                        │
                        ▼
    Simulated Closed-Loop Machine Response Trajectory
   (Transient recovery curves returning to nominal envelope)
```

---

## 🚨 10 Rig Operating Scenarios

| Code | Subsystem | Fault Scenario | Primary Physics Symptoms | Corrective Action SOP |
|:---:|:---:|:---|:---|:---|
| **NORMAL** | Base | Nominal Healthy Drilling | All parameters within green safety envelope ($\pm 3\%$) | Maintain automated drilling schedule & logging |
| **E1** | Electrical | Electrical Overcurrent | Current surges to $>380$A, VFD temp reaches $>88^\circ$C | Clamp VFD inverter limit to 210A & throttle torque |
| **E2** | Electrical | Voltage Instability | Bus swings $\pm 65$V, frequency hunting $57.8-62.2$Hz | Engage active line reactors & switch transformer taps |
| **E3** | Electrical | Motor Abnormality | Rotor slip, high stator current with torque collapse | Derate duty cycle by $40\%$ & rebalance PWM carrier |
| **M1** | Mechanical | Excessive Vibration | Vibration RMS surges to $>18$ mm/s (ISO Danger $>7.1$) | Activate Soft-Torque feedback & reduce WOB by $25\%$ |
| **M2** | Mechanical | Bearing Degradation | Bearing temp rises to $>104^\circ$C, crest factor spikes | Trigger high-pressure grease purge & cap RPM to 110 |
| **M3** | Mechanical | Gearbox Abnormality | Oil temp surges to $>108^\circ$C, gear mesh acoustic ripple | Engage auxiliary lube radiator & limit torque slew |
| **H1** | Hydraulic | Pump Degradation | Loop pressure decays to $125$ bar, flow rate collapses | Switch to redundant standby Pump-B & unload swashplate |
| **H2** | Hydraulic | Filter Blockage | Filter $dP$ surges to $7.8$ bar (limit: $5.0$ bar), backpressure | Rotate duplex filter selector & trigger backflush |
| **H3** | Hydraulic | Pressure Loss | Catastrophic drop to $<60$ bar, reservoir loss | Isolate high-pressure manifold & trip safety accumulator |

---

## 🛠️ Technology Stack

- **Application Framework**: Python 3.10+ / 3.13, Streamlit (Wide SCADA control-room layout)
- **Mathematical & Telemetry Engine**: NumPy, Pandas, SciPy
- **Machine Learning**:
  - `sklearn.ensemble.IsolationForest` (Calibrated on normal operating baseline for anomaly scoring)
  - `sklearn.ensemble.RandomForestClassifier` (100-tree ensemble for 10-class fault classification)
- **Visual Analytics**: Plotly Dark Interactive SCADA telemetry waveforms, gauges, and recovery trends
- **GenAI Reasoning**: OpenAI Python SDK connecting to **Nebius Token Factory API**
- **LLM Diagnostic Model**: **NVIDIA Nemotron 3.5 Lightning** (`nvidia/Nemotron-3_5-Lightning`) via Nebius Token Factory OpenAI-compatible API
- **Alternative Supported Models**: `nvidia/nemotron-3-super-120b-a12b`, `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/Llama-3_1-Nemotron-70B-Instruct-HF`
- **Robustness**: Automated local physical rule-based fallback when API key is missing or offline (always transparently labelled)

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Vigneshwaran-tech/Drill-Guard-AI.git
cd Drill-Guard-AI

# Create virtual environment (optional)
python -m venv .venv

# Windows (Command Prompt / PowerShell):
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
# Windows (cmd):
copy .env.example .env

# Windows (PowerShell) / Linux / macOS:
cp .env.example .env
```

Open `.env` and configure your Nebius Token Factory credentials:

```ini
NEBIUS_API_KEY=your_nebius_api_key_here
NEBIUS_BASE_URL=https://api.tokenfactory.eu-north1.nebius.com/v1/
NEBIUS_MODEL=nvidia/Nemotron-3_5-Lightning
```

> **Note**: If `NEBIUS_API_KEY` is not provided, DrillGuard AI operates in **Local Physical Fallback Mode**, providing physics-grounded diagnostic reasoning while clearly indicating the fallback status in the UI.

### 3. Run Automated Tests

Verify all 10 fault scenarios and model classifications:

```bash
python test_drillguard.py
```

Expected output:
```
[PASS] Scenario: NORMAL   -> Predicted: NORMAL   (Confidence:  99.9%, Anomaly Flag: False)
[PASS] Scenario: E1       -> Predicted: E1       (Confidence: 100.0%, Anomaly Flag: True)
...
[PASS] Scenario: H3       -> Predicted: H3       (Confidence: 100.0%, Anomaly Flag: True)
Classification Accuracy: 10/10 (100.0%)
```

### 4. Launch the Streamlit Control Room

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🖥️ Dashboard Features

1. **SCADA Header Bar**: Machine tag (`RIG-ALPHA-07`), live telemetry clock, subsystem status indicators, and hackathon track badges.
2. **Subsystem KPI Cards**: Instant visibility into Bus Voltage, Motor Current, Top-Drive Speed, Vibration RMS, Hydraulic Pressure, and Filter $dP$.
3. **Subsystem Telemetry Tabs**:
   - `⚡ Electrical`: Frequency, Power Factor, Active & Apparent Power, VFD Inverter Temperature, Motor Efficiency.
   - `⚙️ Mechanical`: Drill Torque, Vibration Peak, Crest Factor, Main Thrust Bearing Temperature, Gearbox Oil Temperature.
   - `💧 Hydraulic`: Hydraulic Flow, Fluid Temperature, Main Pump RPM, Reservoir Level, Hydraulic Fluid Output Power.
   - `📈 Live Telemetry Waveforms`: Real-time correlated time-series waveforms across all three subsystems.
4. **Fault Injection Panel**: Interactive sidebar selector to test all 10 operational and failure modes with custom noise and dynamic turbulence sliders.
5. **Dual ML Diagnostics**:
   - Isolation Forest anomaly severity gauge ($0-100\%$)
   - Random Forest probability distribution across all 10 fault signatures
   - Top anomalous telemetry features ranked by deviation from nominal baselines.
6. **NVIDIA Nemotron Diagnostic Intelligence Panel**:
   - Live Nemotron reasoning streamed via Nebius Token Factory API
   - Executive Anomaly Assessment & Severity Badge
   - Subsystem Physical Telemetry Breakdown
   - Root Cause Physical Hypothesis
   - Component Degradation Horizon
   - Recommended Corrective Action SOP
7. **Simulated Machine Response Closed-Loop**:
   - Interactive **"⚡ Apply Simulated Action"** button
   - Simulates physical transient recovery curves returning the rig to nominal operation
   - Before vs. After stabilization charts for current, vibration, and hydraulic pressure.

---

## 🔒 Physical AI Safety & Ethics Disclaimer

- **Demonstration & Research Only**: DrillGuard AI is a software-only simulation built exclusively for the Nebius x NVIDIA Global AI Hackathon 2026 (Physical AI Track).
- **Simulated Data**: All sensor telemetry, noise distributions, and dynamic transient waveforms are synthetically generated.
- **No Physical Machine Control**: This software does not connect to, actuate, or control real-world industrial drilling machinery.
- **Not a Certified Physical Simulator**: This application is not a certified physical twin or safety-critical SIL/ATEX certified controller.

---

## 📄 License

MIT License. Developed for the Nebius x NVIDIA Global AI Hackathon 2026.

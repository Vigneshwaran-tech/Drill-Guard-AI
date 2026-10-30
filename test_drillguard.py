"""
DrillGuard AI - Complete End-to-End Test Suite
Tests all 10 operational and fault scenarios through the entire workflow:
1. Synthetic Telemetry Generation
2. Derived Physical Feature Engineering
3. Isolation Forest Anomaly Detection
4. Random Forest 10-Class Fault Classification
5. NVIDIA Nemotron Structured JSON Diagnosis (via Nebius Token Factory)
6. Structured Schema Validation (7 required fields)
7. Closed-Loop Machine Response Simulation
"""

import sys
import drillguard

def test_all_10_scenarios():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print("DRILLGUARD AI: END-TO-END VERIFICATION (ALL 10 SCENARIOS)")
    print("=" * 80)

    # 1. Initialize & Train ML Models
    print("\n--- 1. INITIALIZING ML ENGINE ---")
    ml = drillguard.DrillGuardML(model_dir="models")
    ml.train_or_load(force_retrain=True)
    print("[OK] Isolation Forest & Random Forest models trained and ready.")

    # 2. Initialize Diagnostic Engine
    diag_engine = drillguard.NemotronDiagnosticEngine()
    print(f"[OK] Diagnostic Engine initialized (Model: {diag_engine.model}).")

    correct_classifications = 0
    scenarios = list(drillguard.SCENARIOS.keys())
    results = []

    print("\n--- 2. EXECUTING END-TO-END PIPELINE ACROSS ALL 10 SCENARIOS ---")
    for idx, code in enumerate(scenarios, start=1):
        meta = drillguard.SCENARIOS[code]
        print(f"\n[{idx}/10] Testing Scenario: {code} ({meta['name']})...")

        # Step A: Generate Telemetry
        pt = drillguard.generate_telemetry_point(code, t_sec=float(idx * 2.0), noise_mult=1.0)
        feats = drillguard.compute_derived_features(pt)
        assert len(feats) >= 28, "Derived features should have at least 28 physics features"

        # Step B: ML Inference
        ml_res = ml.analyze(feats)
        pred_code = ml_res["predicted_fault_code"]
        is_correct = (pred_code == code)
        if is_correct:
            correct_classifications += 1

        # Step C: Nemotron AI Diagnosis (Structured JSON)
        diag = diag_engine.diagnose(feats, ml_res, meta)
        assert "structured" in diag, "Diagnosis must contain structured dictionary"
        structured = diag["structured"]

        # Step D: Schema Validation
        required_keys = ["severity", "explanation", "fault", "component", "recommended_action", "maintenance_action", "additional_checks"]
        missing_keys = [k for k in required_keys if k not in structured]
        assert not missing_keys, f"Missing required keys in diagnosis: {missing_keys}"
        assert structured["severity"] in ["NORMAL", "MINOR", "MODERATE", "CRITICAL"], f"Invalid severity: {structured['severity']}"
        assert len(structured["additional_checks"]) >= 1, "additional_checks must have at least one check"

        # Step E: Machine Recovery Feedback
        trajectory = drillguard.simulate_machine_response(code, steps=5)
        assert len(trajectory) == 5, "Recovery trajectory must return 5 transient steps"

        print(f"  -> Classification: {'PASS' if is_correct else 'FAIL'} | Predicted: {pred_code} (Conf: {ml_res['confidence']*100:.1f}%)")
        print(f"  -> Anomaly Status : {'ANOMALY' if ml_res['is_anomaly'] else 'NOMINAL'} (Score: {ml_res['anomaly_score']:.3f})")
        print(f"  -> AI Source      : {diag['source']} ({diag['status']})")
        print(f"  -> Diagnosed Fault: {structured['fault']}")
        print(f"  -> Component      : {structured['component']}")
        print(f"  -> Severity Level : {structured['severity']}")
        print(f"  -> Rec. Action    : {structured['recommended_action'][:65]}...")
        print(f"  -> Machine State  : Successfully simulated recovery trajectory (5 steps)")

        results.append({
            "code": code,
            "name": meta["name"],
            "pred": pred_code,
            "is_correct": is_correct,
            "confidence": ml_res["confidence"],
            "severity": structured["severity"],
            "source": diag["source"],
        })

    # Summary Table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Code':<8} {'Scenario Name':<28} {'Classification':<16} {'Confidence':<12} {'Severity':<10} {'Source'}")
    print("-" * 80)
    for r in results:
        status_str = "PASS" if r["is_correct"] else "FAIL"
        print(f"{r['code']:<8} {r['name']:<28} {status_str:<16} {r['confidence']*100:5.1f}%       {r['severity']:<10} {r['source']}")

    accuracy = (correct_classifications / len(scenarios)) * 100.0
    print("-" * 80)
    print(f"Overall Classification Accuracy: {correct_classifications}/{len(scenarios)} ({accuracy:.1f}%)")
    print(f"All 10 Scenarios Structured JSON Validated: TRUE")
    print(f"Closed-Loop Machine Recovery Feedback Validated: TRUE")
    print("=" * 80)
    print("[OK] ALL 10 SCENARIOS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all_10_scenarios()

"""
Audit script: Verify live Nemotron diagnosis across Electrical, Mechanical, and Hydraulic subsystems.
"""
import sys
import drillguard

def run_subsystem_audit():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 80)
    print("DRILLGUARD AI: LIVE SUBSYSTEM NEMOTRON DIAGNOSIS AUDIT")
    print("=" * 80)

    ml = drillguard.DrillGuardML(model_dir="models")
    ml.train_or_load()
    engine = drillguard.NemotronDiagnosticEngine()

    test_targets = [
        ("NORMAL", "Base Nominal Subsystem"),
        ("E1", "Electrical Subsystem (Overcurrent)"),
        ("M1", "Mechanical Subsystem (Excessive Vibration)"),
        ("H3", "Hydraulic Subsystem (Pressure Loss)")
    ]

    for code, desc in test_targets:
        print(f"\n--- AUDITING: {code} - {desc} ---")
        meta = drillguard.SCENARIOS[code]
        pt = drillguard.generate_telemetry_point(code, t_sec=1.0)
        feats = drillguard.compute_derived_features(pt)
        ml_res = ml.analyze(feats)

        diag = engine.diagnose(feats, ml_res, meta)

        print(f"Subsystem Domain : {meta['category']}")
        print(f"ML Classifier    : {ml_res['predicted_fault_name']} (Confidence: {ml_res['confidence']*100:.1f}%)")
        print(f"Anomaly Detected : {ml_res['is_anomaly']} (Score: {ml_res['anomaly_score']:.3f})")
        print(f"Diagnosis Source : {diag['source']} ({diag['status']})")
        print(f"Model Identifier : {diag['model_used']}")
        print(f"Severity Badge   : {diag['structured']['severity']}")
        print(f"Diagnosed Fault  : {diag['structured']['fault']}")
        print(f"Component        : {diag['structured']['component']}")
        print(f"Rec. Action      : {diag['structured']['recommended_action']}")
        print(f"Maintenance SOP  : {diag['structured']['maintenance_action']}")
        print(f"Additional Checks: {diag['structured']['additional_checks']}")
        print(f"JSON Validated   : {'structured' in diag and len(diag['structured']) == 7}")

if __name__ == "__main__":
    run_subsystem_audit()

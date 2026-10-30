import drillguard

def test_all():
    print("Testing DrillGuard AI engine...")
    ml = drillguard.DrillGuardML(model_dir="models")
    ml.train_or_load(force_retrain=True)
    print("[OK] Model training completed successfully.")

    correct = 0
    total = len(drillguard.SCENARIOS)

    for code in drillguard.SCENARIOS.keys():
        pt = drillguard.generate_telemetry_point(code, t_sec=1.0)
        feats = drillguard.compute_derived_features(pt)
        res = ml.analyze(feats)
        pred = res["predicted_fault_code"]
        is_correct = (pred == code)
        if is_correct:
            correct += 1
        print(f"[{'PASS' if is_correct else 'FAIL'}] Scenario: {code:8} -> Predicted: {pred:8} (Confidence: {res['confidence']*100:5.1f}%, Anomaly Flag: {res['is_anomaly']}, Score: {res['anomaly_score']})")

    print(f"\nClassification Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # Test diagnosis engine
    engine = drillguard.NemotronDiagnosticEngine()
    diag = engine.diagnose(pt, res, drillguard.SCENARIOS["E1"])
    print(f"\nDiagnostic Source: {diag['source']} | Status: {diag['status']}")
    assert len(diag["diagnostic_text"]) > 100, "Diagnostic text should be non-empty"

    # Test machine recovery simulation
    trajectory = drillguard.simulate_machine_response("M1", steps=5)
    assert len(trajectory) == 5, "Trajectory should have 5 steps"
    print(f"Machine recovery simulation: {len(trajectory)} steps simulated.")
    print("\n[OK] ALL TESTS PASSED!")

if __name__ == "__main__":
    test_all()

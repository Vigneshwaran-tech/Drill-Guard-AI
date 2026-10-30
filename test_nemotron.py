"""
DrillGuard AI - NVIDIA Nemotron 3.5 Lightning Connection Verification
Nebius x NVIDIA Global AI Hackathon 2026 - Physical AI Track

Tests the OpenAI-compatible API connection to Nebius Token Factory:
- Reads all credentials from .env
- Uses the region-specific base URL
- Connects to nvidia/Nemotron-3_5-Lightning
- Sends a short rig diagnostic question
- Displays the model response and latency
- Masks the API key to protect credentials
- Handles authentication, billing, and model errors clearly
"""

import os
import sys
from dotenv import load_dotenv
import drillguard

def run_test():
    # Ensure stdout handles UTF-8 characters cleanly on Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("DRILLGUARD AI: NVIDIA NEMOTRON 3.5 LIGHTNING CONNECTION TEST")
    print("=" * 70)

    # 1. Load credentials from .env
    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.eu-north1.nebius.com/v1/").strip()
    model = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning").strip()

    # Mask API key for security
    if api_key:
        masked_key = f"{api_key[:8]}...{api_key[-6:]} (Length: {len(api_key)} chars)"
    else:
        masked_key = "[MISSING OR EMPTY]"

    print(f"Region Base URL : {base_url}")
    print(f"Target Model    : {model}")
    print(f"API Key Status  : Configured ({masked_key})")
    print("-" * 70)

    # 2. Define short diagnostic question
    diagnostic_question = (
        "Rig telemetry test: Top-drive vibration RMS reached 19.2 mm/s with stick-slip chatter. "
        "Provide immediate physical diagnosis and primary mitigation in 2 concise sentences."
    )
    print(f"Diagnostic Query: {diagnostic_question}\n")
    print("Connecting to Nebius Token Factory and awaiting Nemotron response...")

    # 3. Execute connection test
    result = drillguard.test_nemotron_connection(diagnostic_question=diagnostic_question)

    print("-" * 70)
    if result["success"]:
        print(f"STATUS  : [SUCCESS] Connected to NVIDIA Nemotron!")
        print(f"LATENCY : {result['latency_sec']} seconds")
        print(f"MODEL   : {result['model']}")
        print(f"ENDPOINT: {result['base_url']}")
        print("\n--- MODEL DIAGNOSTIC RESPONSE ---")
        print(result["response"])
        print("-" * 70)
        print("[OK] Test passed! DrillGuard AI is connected to NVIDIA Nemotron 3.5 Lightning.")
        return 0
    else:
        print(f"STATUS  : [FAILED] Connection or Model Error")
        print(f"LATENCY : {result['latency_sec']} seconds")
        print(f"ERROR   : {result['error']}")
        print("-" * 70)
        print("[FAIL] Please check credentials or billing quotas above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_test())

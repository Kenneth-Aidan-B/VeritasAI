"""Quick test script for the VeritasAI analysis endpoint."""
import requests
import json
import sys

PORT = 8000  # Change to match your running server

try:
    r = requests.post(
        f"http://localhost:{PORT}/api/analyze",
        json={
            "text": "The Great Wall of China is visible from space with the naked eye.",
            "enable_debate": False,
        },
        timeout=180,
    )
    d = r.json()
    print(f"Status: {d.get('status')}")
    print(f"Error: {d.get('error')}")
    print(f"Claims found: {len(d.get('claims', []))}")
    for c in d.get("claims", []):
        claim = c.get("claim", {})
        verdict = c.get("verdict", {})
        print(f"  Claim {claim.get('id')}: {claim.get('atomic_claim', 'N/A')[:80]}")
        print(f"    Verdict: {verdict.get('verdict', 'N/A')} ({verdict.get('confidence', 0):.0f}%)")
    print("\nFull response:")
    print(json.dumps(d, indent=2)[:5000])
except requests.exceptions.ConnectionError:
    print(f"ERROR: Cannot connect to backend at http://localhost:{PORT}")
    print("Make sure the backend is running: uvicorn backend.main:app --port 8001")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

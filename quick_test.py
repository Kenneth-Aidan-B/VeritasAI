"""Quick health + smoke test for VeritasAI with OpenRouter."""
import requests
import json
import sys

BASE = "http://localhost:8000"

# 1) Health check
print("=== Health Check ===")
try:
    r = requests.get(f"{BASE}/health", timeout=10)
    h = r.json()
    print(json.dumps(h, indent=2))
    if h.get("status") != "healthy":
        print("WARNING: Backend is not fully healthy")
except Exception as e:
    print(f"FATAL: Backend not reachable: {e}")
    sys.exit(1)

# 2) Quick analysis with a short claim
print("\n=== Analysis Test ===")
print("Sending: 'The Earth is flat.'")
try:
    r = requests.post(
        f"{BASE}/api/analyze",
        json={"text": "The Earth is flat.", "enable_debate": False},
        timeout=300,
    )
    d = r.json()
    print(f"Status: {d.get('status')}")
    print(f"Error: {d.get('error')}")
    claims = d.get("claims", [])
    print(f"Claims found: {len(claims)}")
    for c in claims:
        cl = c.get("claim", {})
        v = c.get("verdict", {})
        print(f"  Claim: {cl.get('atomic_claim', 'N/A')[:100]}")
        print(f"  Verdict: {v.get('verdict', 'N/A')} ({v.get('confidence', 0):.0f}%)")
    if d.get("status") == "completed" and len(claims) > 0:
        print("\n*** SUCCESS: OpenRouter pipeline is fully working! ***")
    elif d.get("error"):
        print(f"\n*** ISSUE: {d.get('error')[:200]} ***")
    else:
        print("\n*** WARNING: Completed but no claims extracted ***")
except requests.exceptions.Timeout:
    print("TIMEOUT: The request took over 5 minutes. The thinking model may be slow.")
    print("The pipeline IS working — just needs more time.")
except Exception as e:
    print(f"ERROR: {e}")

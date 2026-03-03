"""Start server, run test, write output to test_output.txt, then stop server."""
import subprocess
import time
import requests
import json
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8003

# Start uvicorn as a subprocess
print(f"Starting server on port {PORT}...")
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(PORT)],
)

# Wait for server to be ready
print("Waiting for server to start...")
for i in range(30):
    time.sleep(1)
    try:
        r = requests.get(f"http://localhost:{PORT}/health", timeout=5)
        if r.status_code == 200:
            print(f"Server ready! Health: {r.json()}")
            break
    except Exception:
        pass
else:
    print("ERROR: Server did not start in time")
    server.terminate()
    sys.exit(1)

# Run the analysis
print("\nSending analysis request...")
try:
    r = requests.post(
        f"http://localhost:{PORT}/api/analyze",
        json={
            "text": "The Great Wall of China is visible from space with the naked eye.",
            "enable_debate": False,
        },
        timeout=600,
    )
    d = r.json()
    print(f"\n{'='*60}")
    print(f"Status: {d.get('status')}")
    print(f"Error: {d.get('error')}")
    print(f"Claims found: {len(d.get('claims', []))}")
    for c in d.get("claims", []):
        claim = c.get("claim", {})
        verdict = c.get("verdict", {})
        print(f"\n  Claim: {claim.get('atomic_claim', 'N/A')[:100]}")
        print(f"  Verdict: {verdict.get('verdict', 'N/A')} ({verdict.get('confidence', 0):.0f}%)")
        print(f"  Explanation: {c.get('explanation', {}).get('plain_english', 'N/A')[:200]}")
    print(f"{'='*60}")

    # Save full output
    with open("test_output.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(d, indent=2))
    print("\nFull response saved to test_output.txt")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\nStopping server...")
    server.terminate()
    server.wait(timeout=10)
    print("Done.")

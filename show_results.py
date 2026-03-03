"""Quick script to show analysis results."""
import requests
import json

r = requests.post(
    "http://localhost:8000/api/analyze",
    json={"text": "The Earth is flat.", "session_id": "results_test"},
    timeout=600
)

data = r.json()
print(f"Status: {r.status_code}")
print(f"\n{'='*60}")
print(f"OVERALL VERDICT: {data.get('overall_assessment', {}).get('verdict', 'N/A')}")
print(f"CONFIDENCE: {data.get('overall_assessment', {}).get('confidence', 'N/A')}")
print(f"{'='*60}")

claims = data.get("claims", [])
print(f"\nClaims analyzed: {len(claims)}")

for i, claim in enumerate(claims):
    print(f"\n--- Claim {i+1} ---")
    print(f"  Text: {claim.get('claim_text', 'N/A')}")
    print(f"  Verdict: {claim.get('verdict', 'N/A')}")
    conf = claim.get('confidence', 0)
    print(f"  Confidence: {conf*100:.1f}%")
    print(f"  Explanation: {claim.get('explanation', 'N/A')[:200]}...")
    
    corrections = claim.get("corrections", [])
    if corrections:
        print(f"  Corrections: {len(corrections)}")
        for j, c in enumerate(corrections):
            print(f"    {j+1}. {c[:150]}...")
    
    evidence = claim.get("evidence", [])
    print(f"  Evidence items: {len(evidence)}")

print(f"\n{'='*60}")
print("Pipeline completed successfully! ✅")

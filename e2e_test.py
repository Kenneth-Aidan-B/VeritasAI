"""Quick end-to-end test for Meta Llama 3.3 70B Instruct model."""
import requests, json, time

print("=" * 60)
print("  VeritasAI End-to-End Test (Meta Llama 3.3 70B Instruct)")
print("=" * 60)

# Health check
try:
    h = requests.get("http://localhost:8000/health", timeout=5).json()
    print(f"\nBackend: {h['status']} | Model: {h['llm_model']}")
except Exception as e:
    print(f"\n❌ Backend not reachable: {e}")
    exit(1)

# Run analysis
text = "The Great Wall of China is visible from space with the naked eye."
print(f"\nQuery: {text}")
print("Analyzing...")
start = time.time()
try:
    r = requests.post(
        "http://localhost:8000/api/analyze",
        json={"text": text, "enable_debate": False},
        timeout=300,
    )
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s | Status: {r.status_code}")
except Exception as e:
    print(f"❌ Request failed: {e}")
    exit(1)

if r.status_code == 200:
    data = r.json()
    claims = data.get("claims", [])
    print(f"\nClaims found: {len(claims)}")
    for i, c in enumerate(claims):
        ci = c.get("claim") or {}
        vi = c.get("verdict") or {}
        ei = c.get("evidence") or {}
        ex = c.get("explanation") or {}
        co = c.get("correction") or {}
        print(f"\n  Claim {i+1}: {(ci.get('atomic_claim') or '?')[:80]}")
        print(f"    Verdict:     {vi.get('verdict', '?')} ({vi.get('confidence', 0):.0f}%)")
        print(f"    Evidence:    {len(ei.get('evidence_items', []))} items")
        print(f"    Explanation: {'yes' if ex else 'no'}")
        print(f"    Correction:  {'yes' if co else 'no'}")

    # Dump first claim keys for debugging
    if claims:
        print(f"\n  [Debug] First claim keys: {list(claims[0].keys())}")

    rai = data.get("responsible_ai_card", {})
    audit = data.get("audit_log", [])
    print(f"\nRAI card: {'yes' if rai else 'no'} | Audit log: {len(audit)} entries")
    print(f"\n{'=' * 60}")
    print("  ✅ TEST PASSED")
    print(f"{'=' * 60}")
else:
    print(f"\n❌ ERROR {r.status_code}: {r.text[:500]}")

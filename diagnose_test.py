"""Diagnostic test to check evidence retrieval and verdict quality."""
import requests
import json
import time

def test_claim(text, test_name="Test"):
    print(f"\n{'=' * 80}")
    print(f"{test_name}: {text}")
    print("=" * 80)

    start = time.time()
    r = requests.post(
        "http://localhost:8000/api/analyze",
        json={"text": text, "enable_debate": False},
        timeout=120,
    )
    elapsed = time.time() - start

    d = r.json()
    print(f"STATUS: {r.status_code} (took {elapsed:.1f}s)")

    if r.status_code != 200:
        print(f"ERROR: {json.dumps(d, indent=2)[:500]}")
        return

    claims = d.get("claims") or []
    print(f"Number of claims: {len(claims)}")

    for ci, c in enumerate(claims):
        ev = c.get("evidence") or {}
        items = ev.get("evidence_items") or []
        print(f"\nTotal evidence items: {len(items)}")

        sources = {}
        for i, item in enumerate(items):
            src = item.get("source_type", "?")
            stance = item.get("stance", "?")
            cred = item.get("credibility_score", 0)
            title = item.get("title", "")[:70]
            snippet = item.get("snippet", "")[:100]
            print(f"  [{i}] {src:18s} | stance={stance:8s} | cred={cred:.2f} | {title}")
            sources[src] = sources.get(src, 0) + 1

        print(f"\nSources used: {sources}")
        matrix = ev.get("agreement_matrix") or {}
        print(f"Agreement: +{matrix.get('supporting',0)} / -{matrix.get('refuting',0)} / ~{matrix.get('neutral',0)} | consensus={matrix.get('consensus')}")

        v = c.get("verdict") or {}
        print(f"\n>>> VERDICT: {v.get('verdict')} | CONFIDENCE: {v.get('confidence')}")
        print(f"    Band: {v.get('confidence_band')}")
        print(f"    Reasoning: {v.get('reasoning_chain', '')[:200]}")

# Test 1: Obviously false claim
test_claim("The Earth is flat.", "Test 1 (false)")

# Test 2: Obviously true claim
test_claim("Water boils at 100 degrees Celsius at standard atmospheric pressure.", "Test 2 (true)")

# Test 3: Another false claim
test_claim("COVID-19 vaccines contain microchips.", "Test 3 (false)")

# Test 4: Another true claim
test_claim("The speed of light is approximately 300,000 kilometers per second.", "Test 4 (true)")

print("\n\nDone!")

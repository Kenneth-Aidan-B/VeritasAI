"""Quick end-to-end verification test."""
import requests
import json

tests = [
    ("Modi killed by Iran", "Modi was killed by Iran in a military operation.", "REFUTED"),
    ("Water boils at 100C", "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "SUPPORTED"),
]

print("=" * 60)
print("VeritasAI End-to-End Test")
print("=" * 60)

for name, text, expected in tests:
    print(f"\n--- Testing: {name} ---")
    try:
        r = requests.post(
            "http://localhost:8000/api/analyze",
            json={"text": text, "enable_debate": True},
            timeout=120,
        )
        data = r.json()
        claims = data.get("claims", [])
        if claims:
            c = claims[0]
            v = c.get("verdict", {}).get("verdict", "N/A")
            conf = c.get("verdict", {}).get("confidence", 0)
            items = (c.get("evidence", {}) or {}).get("evidence_items", [])
            sources = set(i.get("source_type", "?") for i in items)
            
            # Check for news brands in URLs
            brands_found = []
            for item in items:
                url = item.get("url", "")
                if url:
                    for domain in ["reuters", "bbc", "cnn", "ndtv", "wikipedia", "espncricinfo"]:
                        if domain in url.lower():
                            brands_found.append(domain)
            
            result = "PASS" if v == expected else "FAIL"
            icon = "+" if result == "PASS" else "x"
            print(f"  [{icon}] Verdict: {v} ({conf:.0f}%) | Expected: {expected}")
            print(f"      Sources: {sources}")
            if brands_found:
                print(f"      News brands found: {brands_found}")
            print(f"      Evidence items: {len(items)}")
        else:
            print("  [x] No claims found")
    except Exception as e:
        print(f"  [x] Error: {e}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)

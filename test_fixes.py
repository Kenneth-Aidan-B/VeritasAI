"""Quick test to verify accuracy + regional news fixes."""
import asyncio
import json
import httpx

CLAIMS = [
    "Modi killed by Iran",
    "Water boils at 100 degrees Celsius",
    "MK Stalin resigned as Tamil Nadu CM",
    "India won the Cricket World Cup 2025",
]

async def test():
    for claim in CLAIMS:
        print("\n" + "=" * 60)
        print("CLAIM: " + claim)
        print("=" * 60)
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(
                    "http://localhost:8000/api/analyze",
                    json={"text": claim, "mode": "quick"},
                )
                data = resp.json()
                claims_list = data.get("claims", [])
                if claims_list:
                    c = claims_list[0]
                    verdict_obj = c.get("verdict", {})
                    print("VERDICT:    " + str(verdict_obj.get("verdict", "?")))
                    print("CONFIDENCE: " + str(verdict_obj.get("confidence", "?")) + "%")
                    evidence_obj = c.get("evidence", {})
                    items = evidence_obj.get("evidence_items", [])
                    print("EVIDENCE:   " + str(len(items)) + " items")
                    print("SUPPORTING: " + str(evidence_obj.get("total_supporting", 0)))
                    print("REFUTING:   " + str(evidence_obj.get("total_refuting", 0)))
                    print("NEUTRAL:    " + str(evidence_obj.get("total_neutral", 0)))
                    for e in items[:8]:
                        stance = e.get("stance", "?")
                        src = e.get("source_type", "?")
                        title = (e.get("title", "?") or "?")[:70]
                        print("  [" + stance.rjust(10) + "] (" + src + ") " + title)
                else:
                    print("No claims found")
                    print(json.dumps(data, indent=2)[:500])
        except Exception as ex:
            print("FAILED: " + str(ex))

if __name__ == "__main__":
    asyncio.run(test())

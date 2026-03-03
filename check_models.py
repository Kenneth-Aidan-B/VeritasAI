"""List available free models on OpenRouter."""
import requests

r = requests.get("https://openrouter.ai/api/v1/models")
models = r.json()["data"]

free = []
for m in models:
    p = m.get("pricing", {})
    if p.get("prompt") == "0" and p.get("completion") == "0":
        free.append(m)

free.sort(key=lambda m: int(m.get("context_length", 0)), reverse=True)

print(f"Total free models: {len(free)}\n")
for m in free:
    mid = m["id"]
    ctx = m.get("context_length", "?")
    print(f"  {mid:60s} ctx={ctx:>8}")

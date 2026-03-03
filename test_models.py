"""Quick model test — checks which free model works."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("backend/.env")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

models_to_test = [
    "qwen/qwen3-235b-a22b-thinking-2507",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

for model_id in models_to_test:
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=20,
        )
        print(f"✅ {model_id:55s} => {r.choices[0].message.content.strip()[:60]}")
    except Exception as e:
        print(f"❌ {model_id:55s} => {e}")

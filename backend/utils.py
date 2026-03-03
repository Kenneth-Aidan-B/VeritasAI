"""
VeritasAI — Utility functions.
Shared helpers for the backend agents and services.
"""

import re


def strip_thinking_tags(text: str) -> str:
    """
    Strip <think>...</think> blocks from thinking/reasoning model responses.
    Many reasoning models (e.g., Qwen3 Thinking) wrap their internal chain-of-thought
    inside <think> tags before producing the final answer.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text

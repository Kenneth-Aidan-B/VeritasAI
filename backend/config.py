"""
VeritasAI Configuration Module
Centralizes all settings and environment variable management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables — check multiple possible locations
_root_env = Path(__file__).parent.parent / ".env"
_backend_env = Path(__file__).parent / ".env"

# Prefer backend/.env if it exists, otherwise fall back to project root
env_path = _backend_env if _backend_env.exists() else _root_env
load_dotenv(dotenv_path=env_path, override=True)


class Settings:
    """Application settings loaded from environment variables."""

    # API Keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    GOOGLE_FACTCHECK_API_KEY: str = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    GOOGLE_AI_STUDIO_API_KEY: str = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")

    # Gemini
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # OpenRouter
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Backend
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "900"))

    # Performance / Fast mode
    FAST_MODE: bool = os.getenv("FAST_MODE", "true").lower() in ("1", "true", "yes", "on")
    MAX_CLAIMS: int = int(os.getenv("MAX_CLAIMS", "1"))
    MAX_EVIDENCE_PER_SOURCE: int = int(os.getenv("MAX_EVIDENCE_PER_SOURCE", "6"))
    EXTERNAL_API_TIMEOUT_SEC: float = float(os.getenv("EXTERNAL_API_TIMEOUT_SEC", "15"))

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # Veracity Labels
    VERACITY_LABELS = {
        "SUPPORTED": {"emoji": "✅", "color": "#22c55e", "description": "Strong evidence confirms the claim"},
        "LIKELY_SUPPORTED": {"emoji": "🟢", "color": "#86efac", "description": "Moderate evidence leans toward confirmation"},
        "INSUFFICIENT_EVIDENCE": {"emoji": "⚪", "color": "#9ca3af", "description": "Not enough data to determine"},
        "LIKELY_REFUTED": {"emoji": "🟠", "color": "#fb923c", "description": "Moderate evidence leans toward refutation"},
        "REFUTED": {"emoji": "❌", "color": "#ef4444", "description": "Strong evidence contradicts the claim"},
    }

    @classmethod
    def validate(cls) -> list[str]:
        """Validate that required API keys are set. Returns list of missing keys."""
        missing = []
        if not cls.OPENROUTER_API_KEY or cls.OPENROUTER_API_KEY == "your_openrouter_api_key_here":
            missing.append("OPENROUTER_API_KEY")
        if not cls.TAVILY_API_KEY or cls.TAVILY_API_KEY == "your_tavily_api_key_here":
            missing.append("TAVILY_API_KEY")
        return missing


settings = Settings()

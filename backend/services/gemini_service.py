"""
VeritasAI — Google Gemini AI Verification Service
Uses Google's Gemini model for intelligent claim cross-verification.
Provides high-accuracy factual verification as a 5th evidence source.
"""

import asyncio
import logging
import re
from datetime import datetime
from backend.config import settings
from backend.models.schemas import EvidenceItem, SourceType

logger = logging.getLogger(__name__)


async def verify_with_gemini(claim_text: str, max_results: int = 1) -> list[EvidenceItem]:
    """
    Verify a claim using Google Gemini AI.
    
    Gemini serves as a high-authority cross-verification source that understands
    context, nuance, and has extensive world knowledge.
    
    Args:
        claim_text: The claim to verify
        max_results: Not used (Gemini returns one comprehensive response)
        
    Returns:
        List with one EvidenceItem containing Gemini's verification result
    """
    api_key = settings.GOOGLE_AI_STUDIO_API_KEY
    if not api_key or api_key in ("", "your_google_ai_studio_api_key_here"):
        logger.warning("Google AI Studio API key not configured, skipping Gemini verification")
        return []

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            settings.GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
            ),
        )

        prompt = f"""You are a rigorous fact-checker. Analyze this claim and determine its factual accuracy.

CLAIM: "{claim_text}"

Instructions:
- Determine if the claim is factually TRUE, FALSE, or UNVERIFIABLE
- Provide a brief, evidence-based explanation (2-3 sentences)
- State the key factual point that supports your verdict
- Be definitive — avoid hedging when evidence is clear

Respond in EXACTLY this format (no markdown, no extra text):
VERDICT: TRUE or FALSE or UNVERIFIABLE
CONFIDENCE: HIGH or MEDIUM or LOW
EXPLANATION: [Your brief factual explanation]
KEY_FACT: [The single most important fact supporting your verdict]"""

        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=settings.EXTERNAL_API_TIMEOUT_SEC + 5,  # Slightly longer timeout for AI
        )

        if not response or not response.text:
            logger.warning("Gemini returned empty response")
            return []

        response_text = response.text.strip()

        # Parse structured response
        verdict, confidence_level, explanation, key_fact = _parse_gemini_response(response_text)

        # Map verdict to stance
        stance = "neutral"
        if verdict == "TRUE":
            stance = "supports"
        elif verdict == "FALSE":
            stance = "refutes"

        # Map confidence level to score
        confidence_score = {"HIGH": 0.92, "MEDIUM": 0.75, "LOW": 0.55}.get(confidence_level, 0.70)

        # Build snippet from parsed components
        snippet = explanation
        if key_fact and key_fact not in explanation:
            snippet += f" Key fact: {key_fact}"

        item = EvidenceItem(
            source_type=SourceType.GEMINI,
            title=f"Gemini AI Verification — {verdict}",
            url="",
            snippet=snippet[:500],
            full_text=response_text,
            credibility_score=confidence_score,
            relevance_score=0.90,
            stance=stance,
            retrieved_at=datetime.utcnow().isoformat(),
        )

        logger.info(
            f"Gemini verified claim: verdict={verdict}, stance={stance}, "
            f"confidence={confidence_level} for: {claim_text[:50]}..."
        )
        return [item]

    except asyncio.TimeoutError:
        logger.warning(f"Gemini verification timed out for: {claim_text[:50]}...")
        return []
    except Exception as e:
        logger.error(f"Gemini verification failed: {e}")
        return []


def _parse_gemini_response(text: str) -> tuple[str, str, str, str]:
    """
    Parse Gemini's structured response into components.
    
    Returns:
        (verdict, confidence_level, explanation, key_fact)
    """
    verdict = "UNVERIFIABLE"
    confidence = "MEDIUM"
    explanation = text  # Fallback to full text
    key_fact = ""

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        upper_line = line.upper()

        if upper_line.startswith("VERDICT:"):
            raw_verdict = line.split(":", 1)[1].strip().upper()
            if "TRUE" in raw_verdict and "FALSE" not in raw_verdict and "UNVERIFIABLE" not in raw_verdict:
                verdict = "TRUE"
            elif "FALSE" in raw_verdict:
                verdict = "FALSE"
            elif "UNVERIFIABLE" in raw_verdict or "UNVERIFIED" in raw_verdict:
                verdict = "UNVERIFIABLE"

        elif upper_line.startswith("CONFIDENCE:"):
            raw_conf = line.split(":", 1)[1].strip().upper()
            if "HIGH" in raw_conf:
                confidence = "HIGH"
            elif "LOW" in raw_conf:
                confidence = "LOW"
            else:
                confidence = "MEDIUM"

        elif upper_line.startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

        elif upper_line.startswith("KEY_FACT:") or upper_line.startswith("KEY FACT:"):
            key_fact = line.split(":", 1)[1].strip()

    # If explanation is still the full text (parsing failed), try to extract meaningful part
    if explanation == text and len(text) > 100:
        # Try to get just the explanation line
        for line in lines:
            if "explanation" not in line.lower() and "verdict" not in line.lower() \
               and "confidence" not in line.lower() and "key" not in line.lower() \
               and len(line.strip()) > 20:
                explanation = line.strip()
                break

    return verdict, confidence, explanation, key_fact

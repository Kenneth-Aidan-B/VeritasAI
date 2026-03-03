"""
VeritasAI — Google Fact Check Tools API Service
Searches existing fact-checks using the Google Fact Check Tools API.
"""

import logging
from datetime import datetime
import httpx
from backend.config import settings
from backend.models.schemas import EvidenceItem, SourceType

logger = logging.getLogger(__name__)

FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


async def search_factcheck(query: str, max_results: int = 5) -> list[EvidenceItem]:
    """
    Search Google Fact Check Tools API for existing fact-checks.
    
    Args:
        query: The claim text to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of EvidenceItem objects from ClaimReview database
    """
    api_key = settings.GOOGLE_FACTCHECK_API_KEY
    if not api_key or api_key == "your_google_factcheck_api_key_here":
        logger.warning("Google Fact Check API key not configured, skipping")
        return []

    try:
        params = {
            "key": api_key,
            "query": query,
            "pageSize": max_results,
            "languageCode": "en",
        }

        async with httpx.AsyncClient(timeout=settings.EXTERNAL_API_TIMEOUT_SEC) as client:
            response = await client.get(FACTCHECK_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        evidence_items = []

        for claim_data in data.get("claims", []):
            claim_text = claim_data.get("text", "")
            claimant = claim_data.get("claimant", "Unknown")

            for review in claim_data.get("claimReview", []):
                publisher = review.get("publisher", {})
                publisher_name = publisher.get("name", "Unknown")

                # Map fact-check rating to stance
                rating = review.get("textualRating", "").lower()
                stance = _map_rating_to_stance(rating)

                # If rating text doesn't clearly map, also check the full
                # snippet and claim text for obvious refuting / supporting
                # language (Google Fact Check often has descriptive ratings)
                if stance == "neutral":
                    full_context = f"{rating} {claim_text}".lower()
                    stance = _infer_stance_from_context(full_context)

                item = EvidenceItem(
                    source_type=SourceType.GOOGLE_FACTCHECK,
                    title=f"[{publisher_name}] Fact-check: {review.get('title', claim_text)[:100]}",
                    url=review.get("url", ""),
                    snippet=f"Claim by {claimant}: \"{claim_text}\". "
                            f"Rating: {review.get('textualRating', 'Unknown')} "
                            f"(by {publisher_name})",
                    full_text=f"Claimant: {claimant}\n"
                              f"Claim: {claim_text}\n"
                              f"Rating: {review.get('textualRating', 'Unknown')}\n"
                              f"Publisher: {publisher_name}\n"
                              f"Review URL: {review.get('url', 'N/A')}\n"
                              f"Review Date: {review.get('reviewDate', 'N/A')}",
                    credibility_score=0.95,  # Fact-check orgs are highly credible
                    relevance_score=0.85,
                    stance=stance,
                    retrieved_at=datetime.utcnow().isoformat(),
                )
                evidence_items.append(item)

        logger.info(f"Google Fact Check returned {len(evidence_items)} results for: {query[:50]}...")
        return evidence_items

    except httpx.HTTPStatusError as e:
        logger.error(f"Google Fact Check API HTTP error: {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Google Fact Check search failed: {e}")
        return []


def _map_rating_to_stance(rating: str) -> str:
    """Map a fact-check textual rating to a stance."""
    refuting_keywords = [
        "false", "pants on fire", "incorrect", "wrong", "misleading",
        "mostly false", "inaccurate", "debunked", "no", "lie",
        "not true", "unproven", "unsupported", "unfounded", "fake",
        "fabricated", "distort", "hoax", "myth", "disproven",
        "baseless", "without evidence", "no evidence", "flawed",
    ]
    supporting_keywords = [
        "true", "correct", "accurate", "mostly true", "yes",
        "confirmed", "verified", "supported", "factual",
    ]
    mixed_keywords = [
        "half true", "mixture", "partly", "partial", "mixed",
        "needs context", "unproven", "altered", "exaggerated",
        "out of context",
    ]

    rating_lower = rating.lower()
    for kw in refuting_keywords:
        if kw in rating_lower:
            return "refutes"
    for kw in supporting_keywords:
        if kw in rating_lower:
            return "supports"
    for kw in mixed_keywords:
        if kw in rating_lower:
            return "neutral"
    return "neutral"


def _infer_stance_from_context(text: str) -> str:
    """Infer stance from the broader context text when rating keywords fail."""
    text = text.lower()
    refute_signals = [
        "not true", "not flat", "is false", "is wrong", "disproven",
        "scientifically disproven", "no evidence", "no scientific",
        "abundant evidence", "debunk", "myth", "conspiracy",
        "not supported by", "contradicted by", "refuted by",
        "there is no", "no basis", "no credible", "has been shown",
        "spherical", "globe", "round earth",
    ]
    support_signals = [
        "is true", "is correct", "evidence shows", "confirmed",
        "verified", "supported by", "backed by", "well-established",
    ]

    refute_count = sum(1 for s in refute_signals if s in text)
    support_count = sum(1 for s in support_signals if s in text)

    if refute_count > support_count and refute_count >= 1:
        return "refutes"
    if support_count > refute_count and support_count >= 1:
        return "supports"
    return "neutral"

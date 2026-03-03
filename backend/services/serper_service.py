"""
VeritasAI — Serper Web Search Service
Uses the Serper.dev Google Search API as a backup/additional web evidence source.
"""

import logging
from datetime import datetime
import httpx
from backend.config import settings
from backend.models.schemas import EvidenceItem, SourceType

logger = logging.getLogger(__name__)

SERPER_API_URL = "https://google.serper.dev/search"


async def search_serper(query: str, max_results: int = 5) -> list[EvidenceItem]:
    """
    Search Serper.dev (Google Search API) for web evidence related to a claim.

    Args:
        query: The claim text to search for
        max_results: Maximum number of results to return

    Returns:
        List of EvidenceItem objects
    """
    api_key = settings.SERPER_API_KEY
    if not api_key or api_key in ("", "your_serper_api_key_here"):
        logger.warning("Serper API key not configured, skipping Serper search")
        return []

    try:
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": f"fact check {query}",
            "num": max_results,
        }

        async with httpx.AsyncClient(timeout=settings.EXTERNAL_API_TIMEOUT_SEC) as client:
            response = await client.post(SERPER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        evidence_items = []

        # Process organic results
        for result in data.get("organic", [])[:max_results]:
            url = result.get("link", "")
            credibility = _estimate_domain_credibility(url)
            snippet = result.get("snippet", "")

            item = EvidenceItem(
                source_type=SourceType.SERPER,
                title=result.get("title", ""),
                url=url,
                snippet=snippet[:500],
                full_text=snippet,
                credibility_score=credibility,
                relevance_score=0.65,
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            )
            evidence_items.append(item)

        # Also process knowledge graph if available
        kg = data.get("knowledgeGraph")
        if kg and kg.get("description"):
            item = EvidenceItem(
                source_type=SourceType.SERPER,
                title=kg.get("title", "Knowledge Graph"),
                url=kg.get("descriptionLink", ""),
                snippet=kg.get("description", "")[:500],
                full_text=kg.get("description", ""),
                credibility_score=0.75,
                relevance_score=0.7,
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            )
            evidence_items.append(item)

        # Also check answer box
        answer_box = data.get("answerBox")
        if answer_box:
            answer_text = answer_box.get("answer", "") or answer_box.get("snippet", "")
            if answer_text:
                item = EvidenceItem(
                    source_type=SourceType.SERPER,
                    title=answer_box.get("title", "Answer Box"),
                    url=answer_box.get("link", ""),
                    snippet=answer_text[:500],
                    full_text=answer_text,
                    credibility_score=0.7,
                    relevance_score=0.8,
                    stance="neutral",
                    retrieved_at=datetime.utcnow().isoformat(),
                )
                evidence_items.append(item)

        logger.info(f"Serper returned {len(evidence_items)} results for: {query[:50]}...")
        return evidence_items

    except httpx.HTTPStatusError as e:
        logger.error(f"Serper API HTTP error: {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Serper search failed: {e}")
        return []


def _estimate_domain_credibility(url: str) -> float:
    """Estimate credibility score based on domain reputation."""
    high_credibility_domains = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "nytimes.com", "washingtonpost.com", "nature.com",
        "science.org", "who.int", "cdc.gov", "nih.gov",
        "gov.uk", ".gov", ".edu", "snopes.com", "factcheck.org",
        "politifact.com", "fullfact.org",
    ]
    medium_credibility_domains = [
        "wikipedia.org", "britannica.com", "cnn.com", "theguardian.com",
        "nbcnews.com", "abcnews.go.com", "cbsnews.com", "pbs.org",
        "npr.org", "aljazeera.com",
    ]

    url_lower = url.lower()
    for domain in high_credibility_domains:
        if domain in url_lower:
            return 0.9
    for domain in medium_credibility_domains:
        if domain in url_lower:
            return 0.7
    return 0.5

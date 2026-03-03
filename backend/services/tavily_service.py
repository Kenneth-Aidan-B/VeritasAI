"""
VeritasAI — Tavily Search Service
Retrieves real-time web evidence using the Tavily Search API.
"""

import logging
import asyncio
from datetime import datetime
from backend.config import settings
from backend.models.schemas import EvidenceItem, SourceType

logger = logging.getLogger(__name__)


async def search_tavily(query: str, max_results: int = 5) -> list[EvidenceItem]:
    """
    Search Tavily for real-time web evidence related to a claim.
    
    Args:
        query: The claim text to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of EvidenceItem objects
    """
    if not settings.TAVILY_API_KEY or settings.TAVILY_API_KEY == "your_tavily_api_key_here":
        logger.warning("Tavily API key not configured, skipping Tavily search")
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.search,
                query=f"fact check: {query}",
                search_depth="basic" if settings.FAST_MODE else "advanced",
                max_results=max_results,
                include_raw_content=False,
                include_answer=True,
            ),
            timeout=settings.EXTERNAL_API_TIMEOUT_SEC,
        )

        evidence_items = []

        # Process the Tavily answer (direct factual summary) if available
        answer = response.get("answer")
        if answer and len(answer) > 20:
            answer_item = EvidenceItem(
                source_type=SourceType.TAVILY,
                title="Tavily AI Answer (aggregated)",
                url="",
                snippet=answer[:500],
                full_text=answer,
                credibility_score=0.75,
                relevance_score=0.85,
                stance="neutral",  # Stance will be classified by the retriever
                retrieved_at=datetime.utcnow().isoformat(),
            )
            evidence_items.append(answer_item)

        # Process search results
        for result in response.get("results", []):
            # Estimate credibility based on domain
            url = result.get("url", "")
            credibility = _estimate_domain_credibility(url)

            item = EvidenceItem(
                source_type=SourceType.TAVILY,
                title=result.get("title", ""),
                url=url,
                snippet=result.get("content", "")[:500],
                full_text=result.get("content", ""),
                credibility_score=credibility,
                relevance_score=result.get("score", 0.5),
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            )
            evidence_items.append(item)

        logger.info(f"Tavily returned {len(evidence_items)} results for: {query[:50]}...")
        return evidence_items

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
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

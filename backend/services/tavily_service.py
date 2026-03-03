"""
VeritasAI — Tavily Search Service
Retrieves real-time web evidence using the Tavily Search API.
Uses multiple query strategies for accurate, unbiased evidence retrieval.
"""

import logging
import asyncio
from datetime import datetime
from backend.config import settings
from backend.models.schemas import EvidenceItem, SourceType

logger = logging.getLogger(__name__)


async def search_tavily(query: str, max_results: int = 5) -> list[EvidenceItem]:
    """
    Search Tavily with multiple query strategies for comprehensive coverage.
    Strategy 1: Direct claim text (unbiased, finds general coverage)
    Strategy 2: "is it true" prefix (finds verification/debunking articles)
    """
    if not settings.TAVILY_API_KEY or settings.TAVILY_API_KEY == "your_tavily_api_key_here":
        logger.warning("Tavily API key not configured, skipping Tavily search")
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)

        # Detect India/TN claims and add regional query
        q = query.lower()
        india_kw = ["india", "modi", "bjp", "cricket", "rupee", "sensex", "isro",
                    "tamil", "chennai", "delhi", "mumbai", "kolkata", "ipl",
                    "bollywood", "dmk", "aiadmk", "congress"]
        is_india = any(kw in q for kw in india_kw)

        # Run search strategies in parallel
        tasks = [
            _tavily_search(client, query, max_results, include_answer=True),
            _tavily_search(client, f"is it true that {query}", max(2, max_results // 2), include_answer=False),
        ]

        # Add India-specific / regional search if claim is India-related
        if is_india:
            tasks.append(_tavily_search(client, f"{query} India latest news", max(2, max_results // 2), include_answer=False))
        else:
            tasks.append(_tavily_search(client, f"{query} latest news", max(2, max_results // 2), include_answer=False))

        # Add India-specific / regional search if claim is India-related
        if is_india:
            tasks.append(_tavily_search(client, f"{query} India latest news", max(2, max_results // 2), include_answer=False))
        else:
            tasks.append(_tavily_search(client, f"{query} latest news", max(2, max_results // 2), include_answer=False))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items = []
        for r in results:
            if isinstance(r, list):
                all_items.extend(r)

        # Deduplicate by URL
        seen = set()
        unique = []
        for item in all_items:
            key = item.url.lower().rstrip("/") if item.url else item.title
            if key and key not in seen:
                seen.add(key)
                unique.append(item)

        # Sort best evidence first
        unique.sort(key=lambda e: (float(e.credibility_score) + float(e.relevance_score)), reverse=True)

        logger.info(f"Tavily returned {len(unique)} unique results for: {query[:50]}...")
        return unique[:max_results + 2]

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


async def _tavily_search(client, query: str, max_results: int, include_answer: bool) -> list[EvidenceItem]:
    """Execute a single Tavily search."""
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.search,
                query=query,
                search_depth="basic" if settings.FAST_MODE else "advanced",
                max_results=max_results,
                include_raw_content=False,
                include_answer=include_answer,
            ),
            timeout=settings.EXTERNAL_API_TIMEOUT_SEC,
        )

        evidence_items = []

        # Process the Tavily AI answer if available
        answer = response.get("answer")
        if answer and len(answer) > 20:
            evidence_items.append(EvidenceItem(
                source_type=SourceType.TAVILY,
                title="Tavily AI Answer (aggregated)",
                url="",
                snippet=answer[:500],
                full_text=answer,
                credibility_score=0.80,
                relevance_score=0.90,
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            ))

        for result in response.get("results", []):
            url = result.get("url", "")
            evidence_items.append(EvidenceItem(
                source_type=SourceType.TAVILY,
                title=result.get("title", ""),
                url=url,
                snippet=result.get("content", "")[:500],
                full_text=result.get("content", ""),
                credibility_score=_estimate_domain_credibility(url),
                relevance_score=result.get("score", 0.5),
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            ))

        return evidence_items
    except Exception as e:
        logger.warning(f"Tavily search variant failed: {e}")
        return []


def _estimate_domain_credibility(url: str) -> float:
    """Estimate credibility score based on domain reputation."""
    high_credibility_domains = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "nytimes.com", "washingtonpost.com", "nature.com",
        "science.org", "who.int", "cdc.gov", "nih.gov",
        "gov.uk", ".gov", ".edu", "snopes.com", "factcheck.org",
        "politifact.com", "fullfact.org",
        # Indian high-credibility
        "thehindu.com", "indianexpress.com", "ndtv.com",
        "hindustantimes.com", "livemint.com", "pib.gov.in",
        "india.gov.in",
        # Global high-credibility
        "france24.com", "dw.com", "abc.net.au", "cbc.ca",
        "theglobeandmail.com",
    ]
    medium_credibility_domains = [
        "wikipedia.org", "britannica.com", "cnn.com", "theguardian.com",
        "nbcnews.com", "abcnews.go.com", "cbsnews.com", "pbs.org",
        "npr.org", "aljazeera.com",
        # Indian medium-credibility
        "timesofindia.indiatimes.com", "indiatoday.in",
        "news18.com", "firstpost.com", "theprint.in", "scroll.in",
        "deccanherald.com", "deccanchronicle.com", "telegraphindia.com",
        "business-standard.com", "economictimes.indiatimes.com",
        "moneycontrol.com", "thewire.in",
        # Tamil Nadu / South India regional
        "dtnext.in", "newindianexpress.com", "onmanorama.com",
        "dinamalar.com", "dinamani.com", "dailythanthi.com",
        "vikatan.com", "mathrubhumi.com", "manoramaonline.com",
        "thehansindia.com", "sify.com", "greatandhra.com",
        # Global medium-credibility
        "euronews.com", "scmp.com", "timesonline.co.uk",
        "independent.co.uk", "telegraph.co.uk", "sky.com",
        "foxnews.com", "usatoday.com", "thehill.com",
    ]

    url_lower = url.lower()
    for domain in high_credibility_domains:
        if domain in url_lower:
            return 0.9
    for domain in medium_credibility_domains:
        if domain in url_lower:
            return 0.7
    return 0.5

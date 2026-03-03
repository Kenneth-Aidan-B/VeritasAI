"""
VeritasAI — Serper Web Search Service
Uses the Serper.dev Google Search API for web + news evidence with geographic diversity.
Searches global, country-level, and regional news for comprehensive coverage.
"""

import logging
import asyncio
from datetime import datetime
import httpx
from backend.config import settings
from backend.models.schemas import EvidenceItem, SourceType

logger = logging.getLogger(__name__)

SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_NEWS_URL = "https://google.serper.dev/news"

# Geographic search regions for diverse coverage
GEO_REGIONS = [
    {"gl": "us", "hl": "en", "label": "US"},
    {"gl": "in", "hl": "en", "label": "India"},
    {"gl": "gb", "hl": "en", "label": "UK"},
]


async def search_serper(query: str, max_results: int = 5) -> list[EvidenceItem]:
    """
    Search Serper.dev for web + news evidence with geographic diversity.
    Runs multiple query strategies in parallel for broader, more accurate coverage.
    """
    api_key = settings.SERPER_API_KEY
    if not api_key or api_key in ("", "your_serper_api_key_here"):
        logger.warning("Serper API key not configured, skipping Serper search")
        return []

    try:
        tasks = [
            _serper_web_search(api_key, query, max_results),
            _serper_news_search(api_key, query, max_results),
            _serper_web_search(api_key, f'"{query}"', max(2, max_results // 2)),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items = []
        for r in results:
            if isinstance(r, list):
                all_items.extend(r)

        # Deduplicate by URL
        seen_urls = set()
        unique = []
        for item in all_items:
            key = item.url.lower().rstrip("/") if item.url else item.title
            if key not in seen_urls:
                seen_urls.add(key)
                unique.append(item)

        # Sort by credibility + relevance
        unique.sort(key=lambda e: (float(e.credibility_score) + float(e.relevance_score)), reverse=True)

        logger.info(f"Serper returned {len(unique)} unique results for: {query[:50]}...")
        return unique[:max_results + 3]

    except Exception as e:
        logger.error(f"Serper search failed: {e}")
        return []


async def _serper_web_search(api_key: str, query: str, max_results: int) -> list[EvidenceItem]:
    """Standard web search with geographic rotation."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results}

    try:
        async with httpx.AsyncClient(timeout=settings.EXTERNAL_API_TIMEOUT_SEC) as client:
            response = await client.post(SERPER_SEARCH_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        items = []

        for result in data.get("organic", [])[:max_results]:
            url = result.get("link", "")
            item = EvidenceItem(
                source_type=SourceType.SERPER,
                title=result.get("title", ""),
                url=url,
                snippet=result.get("snippet", "")[:500],
                full_text=result.get("snippet", ""),
                credibility_score=_estimate_domain_credibility(url),
                relevance_score=0.65,
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            )
            items.append(item)

        # Knowledge graph
        kg = data.get("knowledgeGraph")
        if kg and kg.get("description"):
            items.append(EvidenceItem(
                source_type=SourceType.SERPER,
                title=kg.get("title", "Knowledge Graph"),
                url=kg.get("descriptionLink", ""),
                snippet=kg.get("description", "")[:500],
                full_text=kg.get("description", ""),
                credibility_score=0.80,
                relevance_score=0.75,
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            ))

        # Answer box
        answer_box = data.get("answerBox")
        if answer_box:
            answer_text = answer_box.get("answer", "") or answer_box.get("snippet", "")
            if answer_text:
                items.append(EvidenceItem(
                    source_type=SourceType.SERPER,
                    title=answer_box.get("title", "Answer Box"),
                    url=answer_box.get("link", ""),
                    snippet=answer_text[:500],
                    full_text=answer_text,
                    credibility_score=0.75,
                    relevance_score=0.80,
                    stance="neutral",
                    retrieved_at=datetime.utcnow().isoformat(),
                ))

        return items
    except Exception as e:
        logger.warning(f"Serper web search failed: {e}")
        return []


async def _serper_news_search(api_key: str, query: str, max_results: int, gl: str = "") -> list[EvidenceItem]:
    """Dedicated news search for recent, geo-diverse coverage."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results}
    if gl:
        payload["gl"] = gl
        payload["hl"] = "en"

    try:
        async with httpx.AsyncClient(timeout=settings.EXTERNAL_API_TIMEOUT_SEC) as client:
            response = await client.post(SERPER_NEWS_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        items = []
        for result in data.get("news", [])[:max_results]:
            url = result.get("link", "")
            source_name = result.get("source", "")
            date_str = result.get("date", "")

            snippet = result.get("snippet", "")
            if source_name:
                snippet = f"[{source_name}] {snippet}"
            if date_str:
                snippet = f"{snippet} ({date_str})"

            item = EvidenceItem(
                source_type=SourceType.SERPER,
                title=result.get("title", ""),
                url=url,
                snippet=snippet[:500],
                full_text=snippet,
                credibility_score=_estimate_domain_credibility(url),
                relevance_score=0.70,
                stance="neutral",
                retrieved_at=datetime.utcnow().isoformat(),
            )
            items.append(item)

        logger.info(f"Serper News returned {len(items)} results")
        return items
    except Exception as e:
        logger.warning(f"Serper news search failed: {e}")
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

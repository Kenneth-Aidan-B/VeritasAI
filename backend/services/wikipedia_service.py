"""
VeritasAI — Wikipedia API Service
Retrieves reference data and knowledge base evidence from Wikipedia.
"""

import logging
from datetime import datetime
import httpx
from backend.config import settings
from backend.models.schemas import EvidenceItem, SourceType

logger = logging.getLogger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


async def search_wikipedia(query: str, max_results: int = 3) -> list[EvidenceItem]:
    """
    Search Wikipedia for reference information related to a claim.
    
    Args:
        query: The claim text to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of EvidenceItem objects from Wikipedia
    """
    try:
        # Step 1: Search for relevant Wikipedia pages
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "srprop": "snippet|titlesnippet",
        }

        async with httpx.AsyncClient(timeout=settings.EXTERNAL_API_TIMEOUT_SEC) as client:
            search_response = await client.get(
                WIKIPEDIA_SEARCH_URL, params=search_params,
                headers={"User-Agent": "VeritasAI/1.0 (fact-checking-platform; contact@veritasai.dev)"},
            )
            search_response.raise_for_status()
            search_data = search_response.json()

        results = search_data.get("query", {}).get("search", [])
        evidence_items = []

        for result in results[:max_results]:
            title = result.get("title", "")
            page_id = result.get("pageid", 0)

            if settings.FAST_MODE:
                item = EvidenceItem(
                    source_type=SourceType.WIKIPEDIA,
                    title=f"Wikipedia: {title}",
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    snippet=_clean_html(result.get("snippet", "")),
                    full_text="",
                    credibility_score=0.65,
                    relevance_score=0.5,
                    stance="neutral",
                    retrieved_at=datetime.utcnow().isoformat(),
                )
                evidence_items.append(item)
                continue

            try:
                # Step 2: Get page summary for each result
                async with httpx.AsyncClient(timeout=settings.EXTERNAL_API_TIMEOUT_SEC) as client:
                    summary_url = WIKIPEDIA_API_URL + title.replace(" ", "_")
                    summary_response = await client.get(
                        summary_url,
                        headers={"User-Agent": "VeritasAI/1.0 (https://github.com/veritasai; contact@veritasai.dev)"},
                    )
                    summary_response.raise_for_status()
                    summary_data = summary_response.json()

                extract = summary_data.get("extract", "")
                page_url = summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")

                if not page_url:
                    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

                item = EvidenceItem(
                    source_type=SourceType.WIKIPEDIA,
                    title=f"Wikipedia: {title}",
                    url=page_url,
                    snippet=extract[:500] if extract else _clean_html(result.get("snippet", "")),
                    full_text=extract,
                    credibility_score=0.7,  # Wikipedia: generally reliable but user-edited
                    relevance_score=_calculate_relevance(query, title, extract),
                    stance="neutral",  # Wikipedia is generally neutral
                    retrieved_at=datetime.utcnow().isoformat(),
                )
                evidence_items.append(item)

            except Exception as e:
                logger.warning(f"Failed to fetch Wikipedia summary for '{title}': {e}")
                # Still include search snippet
                item = EvidenceItem(
                    source_type=SourceType.WIKIPEDIA,
                    title=f"Wikipedia: {title}",
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    snippet=_clean_html(result.get("snippet", "")),
                    full_text="",
                    credibility_score=0.65,
                    relevance_score=0.5,
                    stance="neutral",
                    retrieved_at=datetime.utcnow().isoformat(),
                )
                evidence_items.append(item)

        logger.info(f"Wikipedia returned {len(evidence_items)} results for: {query[:50]}...")
        return evidence_items

    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return []


def _clean_html(html_text: str) -> str:
    """Remove HTML tags from text."""
    import re
    clean = re.sub(r"<[^>]+>", "", html_text)
    return clean.strip()


def _calculate_relevance(query: str, title: str, extract: str) -> float:
    """Calculate simple relevance score based on term overlap."""
    query_terms = set(query.lower().split())
    title_terms = set(title.lower().split())
    extract_terms = set(extract.lower().split()[:100]) if extract else set()

    title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
    extract_overlap = len(query_terms & extract_terms) / max(len(query_terms), 1)

    return min(1.0, (title_overlap * 0.4 + extract_overlap * 0.6) + 0.3)

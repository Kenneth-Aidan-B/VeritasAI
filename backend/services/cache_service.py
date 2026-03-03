"""
VeritasAI — ChromaDB Cache Service
Caches verified claims and evidence for faster re-analysis.
"""

import logging
import hashlib
import json
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

_collection = None
_client = None


def _get_collection():
    """Lazy-initialize ChromaDB collection."""
    global _collection, _client
    if _collection is None:
        try:
            import chromadb
            from backend.config import settings

            _client = chromadb.Client()  # In-memory for hackathon speed
            _collection = _client.get_or_create_collection(
                name="veritas_claims",
                metadata={"description": "Cached claim verifications"},
            )
            logger.info("ChromaDB collection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return None
    return _collection


def _hash_claim(claim_text: str) -> str:
    """Generate a deterministic hash for a claim."""
    return hashlib.sha256(claim_text.lower().strip().encode()).hexdigest()[:16]


async def get_cached_result(claim_text: str) -> Optional[dict]:
    """
    Check if a claim has been previously verified and cached.
    
    Args:
        claim_text: The atomic claim text
        
    Returns:
        Cached result dict or None
    """
    try:
        collection = _get_collection()
        if collection is None:
            return None

        claim_hash = _hash_claim(claim_text)

        results = collection.get(
            ids=[claim_hash],
            include=["metadatas", "documents"],
        )

        if results and results["ids"] and len(results["ids"]) > 0:
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            document = results["documents"][0] if results["documents"] else ""

            if metadata.get("result_json"):
                cached = json.loads(metadata["result_json"])
                cached["_cached"] = True
                cached["_cached_at"] = metadata.get("cached_at", "")
                logger.info(f"Cache HIT for claim: {claim_text[:50]}...")
                return cached

        return None

    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


async def cache_result(claim_text: str, result: dict) -> bool:
    """
    Cache a verified claim result.
    
    Args:
        claim_text: The atomic claim text
        result: The verification result to cache
        
    Returns:
        True if cached successfully
    """
    try:
        collection = _get_collection()
        if collection is None:
            return False

        claim_hash = _hash_claim(claim_text)

        # Remove non-serializable fields
        cache_data = {k: v for k, v in result.items() if not k.startswith("_")}

        collection.upsert(
            ids=[claim_hash],
            documents=[claim_text],
            metadatas=[{
                "claim_text": claim_text[:500],
                "result_json": json.dumps(cache_data, default=str),
                "cached_at": datetime.utcnow().isoformat(),
            }],
        )

        logger.info(f"Cached result for claim: {claim_text[:50]}...")
        return True

    except Exception as e:
        logger.warning(f"Cache write failed: {e}")
        return False


async def search_similar_claims(claim_text: str, n_results: int = 3) -> list[dict]:
    """
    Search for similar previously-verified claims using vector similarity.
    
    Args:
        claim_text: The claim to find similar matches for
        n_results: Number of similar claims to return
        
    Returns:
        List of similar cached results
    """
    try:
        collection = _get_collection()
        if collection is None:
            return []

        results = collection.query(
            query_texts=[claim_text],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        similar_claims = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0

                if distance < 0.5 and metadata.get("result_json"):
                    cached = json.loads(metadata["result_json"])
                    cached["_similarity"] = 1.0 - distance
                    cached["_matched_claim"] = results["documents"][0][i]
                    similar_claims.append(cached)

        return similar_claims

    except Exception as e:
        logger.warning(f"Similar claim search failed: {e}")
        return []

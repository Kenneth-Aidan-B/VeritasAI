"""
VeritasAI — Evidence Retriever Node
Queries three sources in parallel (Tavily, Google Fact Check, Wikipedia),
ranks, deduplicates, and builds an evidence agreement/disagreement matrix.
"""

import asyncio
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from backend.utils import strip_thinking_tags
from backend.config import settings
from backend.models.schemas import (
    GraphState,
    EvidenceResult,
    EvidenceItem,
    AnalysisStatus,
)
from backend.services.tavily_service import search_tavily
from backend.services.factcheck_service import search_factcheck
from backend.services.wikipedia_service import search_wikipedia
from backend.services.serper_service import search_serper
from backend.services.gemini_service import verify_with_gemini
from backend.services.cache_service import get_cached_result, search_similar_claims

logger = logging.getLogger(__name__)

# ─── Stance Classification Prompt ────────────────────────────────────────────

STANCE_CLASSIFICATION_PROMPT = """You are an evidence analysis expert. Given a CLAIM and a piece of EVIDENCE, classify the evidence's stance toward the claim.

**CLAIM:** {claim}

**EVIDENCE:** {evidence}

**Classify the stance as one of:**
- "supports" — The evidence confirms or supports the claim
- "refutes" — The evidence contradicts or refutes the claim
- "neutral" — The evidence is related but doesn't clearly support or refute

**Return ONLY one word:** supports, refutes, or neutral
"""


async def _classify_evidence_stance(
    claim_text: str, evidence: EvidenceItem, llm: ChatOpenAI
) -> str:
    """Use LLM to classify whether evidence supports, refutes, or is neutral to claim."""
    try:
        snippet = evidence.snippet[:300] if evidence.snippet else evidence.full_text[:300]
        if not snippet:
            return "neutral"

        prompt = STANCE_CLASSIFICATION_PROMPT.format(
            claim=claim_text, evidence=snippet
        )
        response = await llm.ainvoke(prompt)
        stance = strip_thinking_tags(response.content.strip()).lower()

        if stance in ("supports", "refutes", "neutral"):
            return stance
        if "support" in stance:
            return "supports"
        if "refut" in stance or "contra" in stance:
            return "refutes"
        return "neutral"
    except Exception as e:
        logger.warning(f"Stance classification failed: {e}")
        return "neutral"


def _infer_stance_fast(claim_text: str, evidence: EvidenceItem) -> str:
    """
    Fast heuristic stance classifier used when FAST_MODE is enabled.
    Uses multiple signal layers for better accuracy without LLM calls.
    """
    text = f"{evidence.title} {evidence.snippet} {evidence.full_text}".lower()
    claim_lower = claim_text.lower()

    # ── Layer 1: Strong fact-check / debunk signals ──────────────────
    # Very high precision phrases that reliably indicate stance
    strong_refute = [
        "is false", "is wrong", "is incorrect", "not true",
        "debunk", "disproven", "scientifically disproven", "no evidence",
        "no scientific evidence", "fabricat", "hoax", "conspiracy theory",
        "misleading claim", "inaccurate claim", "baseless", "unfounded",
        "false claim", "fact check: false", "pants on fire", "rated false",
        "pseudoscience", "discredited", "widely rejected",
        "no credible evidence", "lacks evidence", "flawed claim",
        "myth", "misinformation", "disinformation",
        "there is no evidence", "refuted",
        "fact check:", "isn't evidence", "don't contain", "doesn't contain",
        "do not contain", "does not contain", "no microchip",
        "bogus claim", "bogus", "rumor", "rumors",
        "misconstr", "debunking",
        "is yet to occur", "has not happened", "yet to happen",
        "not confirmed", "do not confirm", "does not confirm",
        "not corroborate", "not support", "no basis",
        "digitally altered", "manipulated", "doctored",
    ]
    strong_support = [
        "is true", "is correct", "is accurate", "confirmed by",
        "verified by", "well-established", "evidence confirms",
        "scientifically proven", "studies confirm", "research confirms",
        "widely accepted", "established fact", "scientific consensus",
        "evidence supports", "data confirms", "backed by evidence",
        "supported by evidence", "supported by science",
        "rated true", "mostly true", "well-documented fact",
    ]

    refute_hits = sum(1 for m in strong_refute if m in text)
    support_hits = sum(1 for m in strong_support if m in text)

    # ── Layer 1b: Negation-aware claim matching ─────────────────────
    # Detect when the evidence NEGATES the claim's core assertion
    # E.g., claim: "India won" but evidence: "India did not win"
    import re

    # Extract claim action words
    claim_action_words = set()
    for word in claim_lower.split():
        if word in ["won", "win", "killed", "defeated", "beat", "scored", "died",
                     "attacked", "invaded", "arrested", "resigned",
                     "destroyed", "collapsed", "winning", "defeating", "beating"]:
            claim_action_words.add(word)

    # Also add verb stems
    claim_action_stems = set()
    for w in claim_action_words:
        claim_action_stems.add(w[:3])  # "won" -> "won", "killed" -> "kil", etc.

    # Check for negation patterns that negate a claim action
    negation_prefixes = [
        "did not ", "didn't ", "has not ", "have not ", "had not ",
        "is not ", "was not ", "were not ", "cannot ", "can't ",
        "won't ", "would not ", "will not ", "never ", "not ",
    ]

    for neg in negation_prefixes:
        neg_pos = text.find(neg)
        while neg_pos >= 0:
            # Get the word(s) after the negation
            after_neg = text[neg_pos + len(neg):neg_pos + len(neg) + 30]
            after_words = after_neg.split()[:3]  # Check next 3 words
            for aw in after_words:
                aw_clean = aw.strip(".,;:!?\"'")
                if aw_clean in claim_action_words:
                    refute_hits += 3  # Very strong signal
                    break
                if len(aw_clean) >= 3 and any(aw_clean[:3] == stem for stem in claim_action_stems):
                    refute_hits += 2
                    break
            neg_pos = text.find(neg, neg_pos + 1)

    if refute_hits > support_hits and refute_hits >= 1:
        return "refutes"
    if support_hits > refute_hits and support_hits >= 1:
        return "supports"

    # ── Layer 2: Topical relevance check ─────────────────────────────
    claim_terms = [t for t in claim_lower.split() if len(t) > 3]
    overlap = sum(1 for t in claim_terms[:12] if t in text)
    if overlap == 0:
        return "neutral"

    # ── Layer 3: Claim-content corroboration ─────────────────────────
    # If the evidence restates the claim's key facts (numbers, entities),
    # it implicitly supports the claim even without "confirmed" language.
    import re
    claim_numbers = set(re.findall(r'\d[\d,\.]*', claim_lower))
    evidence_numbers = set(re.findall(r'\d[\d,\.]*', text))

    # Check for numeric agreement (e.g., claim says "300,000" and evidence says "299,792" or "300,000")
    number_match = False
    if claim_numbers:
        for cn in claim_numbers:
            cn_clean = cn.replace(",", "").replace(".", "")
            if len(cn_clean) >= 2:  # Skip single digits
                for en in evidence_numbers:
                    en_clean = en.replace(",", "").replace(".", "")
                    if cn_clean == en_clean:
                        number_match = True
                        break
                    # Approximate match (within 5%)
                    try:
                        cn_val = float(cn.replace(",", ""))
                        en_val = float(en.replace(",", ""))
                        if cn_val > 0 and abs(cn_val - en_val) / cn_val < 0.05:
                            number_match = True
                            break
                    except ValueError:
                        pass

    # Extract key entities/nouns from claim for corroboration check
    claim_key_terms = [t for t in claim_terms if len(t) > 4]
    key_overlap = sum(1 for t in claim_key_terms[:8] if t in text)
    high_overlap = key_overlap >= max(2, len(claim_key_terms) * 0.4)

    if number_match and high_overlap:
        # But first: check if evidence contains negations of claim action words
        # If evidence says "did not win" but claim says "won", don't support
        has_negation_of_claim = False
        for neg in negation_prefixes:
            neg_pos = text.find(neg)
            while neg_pos >= 0:
                after_neg = text[neg_pos + len(neg):neg_pos + len(neg) + 30]
                after_words = after_neg.split()[:3]
                for aw in after_words:
                    aw_clean = aw.strip(".,;:!?\"'")
                    if aw_clean in claim_action_words or (
                        len(aw_clean) >= 3 and any(aw_clean[:3] == stem for stem in claim_action_stems)
                    ):
                        has_negation_of_claim = True
                        break
                if has_negation_of_claim:
                    break
                neg_pos = text.find(neg, neg_pos + 1)
            if has_negation_of_claim:
                break

        if has_negation_of_claim:
            return "refutes"

        # Evidence restates the claim's core numbers in context = implicit support
        return "supports"

    # ── Layer 4: Sentiment-weighted word analysis ────────────────────
    negative_factual = [
        "false", "wrong", "incorrect", "untrue", "inaccurate",
        "misleading", "debunked", "disproven", "myth", "hoax",
        "conspiracy", "misinformation", "pseudoscience", "rejected",
    ]
    positive_factual = [
        "true", "correct", "accurate", "confirmed", "proven",
        "verified", "established", "supported", "corroborated",
        "validated", "demonstrated",
    ]

    text_words = set(text.split())
    neg_count = sum(1 for w in negative_factual if w in text_words)
    pos_count = sum(1 for w in positive_factual if w in text_words)

    if neg_count > pos_count + 1:
        return "refutes"
    if pos_count > neg_count + 1:
        return "supports"

    # ── Layer 5: High-overlap factual restatement ────────────────────
    # If most of the claim's key terms appear in the evidence and the
    # evidence is from a credible source, it's likely corroborative.
    # BUT: skip this for claims with controversy/debunk-adjacent terms
    # and skip for extraordinary claims (death, war, etc.).
    controversy_terms = [
        "vaccine", "microchip", "conspiracy", "flat", "hoax",
        "fake", "tracking", "5g", "chemtrail", "illuminati",
    ]
    extraordinary_terms = [
        "killed", "murdered", "died", "dead", "assassinated",
        "attacked", "bombed", "invaded", "war", "arrested",
        "won the", "defeated", "beat", "scored",
    ]
    claim_is_controversial = any(t in claim_lower for t in controversy_terms)
    claim_is_extraordinary = any(t in claim_lower for t in extraordinary_terms)
    if high_overlap and evidence.credibility_score >= 0.7 and not claim_is_controversial and not claim_is_extraordinary:
        return "supports"

    # ── Layer 6: Implicit refutation for extraordinary claims ────────
    # If the claim asserts a dramatic event (death, war, attack, etc.)
    # and the evidence discusses the same entity but without mentioning
    # the alleged event, the evidence implicitly contradicts the claim.
    extraordinary_event_words = [
        "killed", "murdered", "assassinated", "died", "dead", "death",
        "attacked", "bombed", "invaded", "war", "arrested", "jailed",
        "resigned", "overthrown", "coup", "shot", "executed",
        "scored", "won", "defeated", "beat", "lost",
    ]
    claim_has_event = any(e in claim_lower for e in extraordinary_event_words)

    if claim_has_event and overlap >= 2:
        # Evidence discusses the same entities
        # Check if the evidence mentions the specific event
        event_in_evidence = any(e in text for e in extraordinary_event_words if e in claim_lower)
        if not event_in_evidence:
            # Evidence discusses same entity but NOT the event → implicit refutation
            return "refutes"

    return "neutral"


def _deduplicate_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Remove duplicate evidence based on URL and content similarity."""
    seen_urls = set()
    seen_snippets = set()
    unique = []

    for item in items:
        # Skip exact URL duplicates
        if item.url and item.url in seen_urls:
            continue

        # Skip very similar snippets (first 100 chars)
        snippet_key = item.snippet[:100].lower().strip() if item.snippet else ""
        if snippet_key and snippet_key in seen_snippets:
            continue

        if item.url:
            seen_urls.add(item.url)
        if snippet_key:
            seen_snippets.add(snippet_key)
        unique.append(item)

    return unique


def _build_agreement_matrix(items: list[EvidenceItem]) -> dict:
    """Build an agreement/disagreement matrix from evidence items."""
    matrix = {
        "total": len(items),
        "supporting": 0,
        "refuting": 0,
        "neutral": 0,
        "by_source": {},
        "consensus": "unknown",
        "conflict_detected": False,
        "source_diversity": 0,
        "avg_credibility": 0.0,
        "avg_relevance": 0.0,
        "contradiction_index": 0.0,
        "robustness_score": 0.0,
    }

    for item in items:
        if item.stance == "supports":
            matrix["supporting"] += 1
        elif item.stance == "refutes":
            matrix["refuting"] += 1
        else:
            matrix["neutral"] += 1

        source_key = item.source_type.value
        if source_key not in matrix["by_source"]:
            matrix["by_source"][source_key] = {"supports": 0, "refutes": 0, "neutral": 0}
        matrix["by_source"][source_key][item.stance] += 1

    # Determine consensus
    if matrix["supporting"] > 0 and matrix["refuting"] > 0:
        matrix["conflict_detected"] = True
        if matrix["supporting"] > matrix["refuting"] * 2:
            matrix["consensus"] = "mostly_supporting"
        elif matrix["refuting"] > matrix["supporting"] * 2:
            matrix["consensus"] = "mostly_refuting"
        else:
            matrix["consensus"] = "conflicting"
    elif matrix["supporting"] > 0:
        matrix["consensus"] = "supporting"
    elif matrix["refuting"] > 0:
        matrix["consensus"] = "refuting"
    else:
        matrix["consensus"] = "insufficient"

    # Advanced robustness metrics
    if items:
        matrix["source_diversity"] = len(set(i.source_type.value for i in items))
        matrix["avg_credibility"] = round(
            sum(float(i.credibility_score) for i in items) / len(items), 3
        )
        matrix["avg_relevance"] = round(
            sum(float(i.relevance_score) for i in items) / len(items), 3
        )

        s = matrix["supporting"]
        r = matrix["refuting"]
        if s + r > 0:
            matrix["contradiction_index"] = round(min(s, r) / (s + r), 3)

        diversity_component = min(1.0, matrix["source_diversity"] / 4.0)
        quality_component = (matrix["avg_credibility"] + matrix["avg_relevance"]) / 2.0
        conflict_penalty = matrix["contradiction_index"]
        matrix["robustness_score"] = round(
            max(0.0, (0.45 * diversity_component) + (0.55 * quality_component) - (0.35 * conflict_penalty)),
            3,
        )

    return matrix


async def retrieve_evidence(state: GraphState) -> GraphState:
    """
    Evidence Retriever Node for LangGraph.
    Queries multiple sources in parallel, classifies stances, builds agreement matrix.
    """
    state.status = AnalysisStatus.RETRIEVING_EVIDENCE

    state.audit_log.append({
        "node": "evidence_retriever",
        "action": "start",
        "timestamp": datetime.utcnow().isoformat(),
    })

    if not state.extraction_result or not state.extraction_result.claims:
        logger.warning("No claims to retrieve evidence for")
        return state

    try:
        llm = None
        if not settings.FAST_MODE:
            llm = ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=settings.LLM_MODEL,
                temperature=0.0,
                max_tokens=50,
                default_headers={"HTTP-Referer": "https://openrouter.ai/", "X-Title": "VeritasAI"},
            )

        evidence_results = []

        for claim in state.extraction_result.claims:
            claim_text = claim.atomic_claim

            if not settings.FAST_MODE:
                # Check cache first
                cached = await get_cached_result(claim_text)
                if cached:
                    logger.info(f"Using cached evidence for claim {claim.id}")
                    state.audit_log.append({
                        "node": "evidence_retriever",
                        "action": "cache_hit",
                        "claim_id": claim.id,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                # Check for previously fact-checked claims (similar claims in cache)
                similar = await search_similar_claims(claim_text)
                if similar:
                    logger.info(f"Found {len(similar)} similar previously-checked claims")

            # Query all sources in parallel (Tavily, Google Fact Check, Wikipedia, Serper, Gemini)
            per_source_limit = max(1, settings.MAX_EVIDENCE_PER_SOURCE)
            tavily_task = search_tavily(claim_text, max_results=per_source_limit)
            factcheck_task = search_factcheck(claim_text, max_results=per_source_limit)
            wikipedia_task = search_wikipedia(claim_text, max_results=per_source_limit)
            serper_task = search_serper(claim_text, max_results=per_source_limit)
            gemini_task = verify_with_gemini(claim_text)

            tavily_results, factcheck_results, wiki_results, serper_results, gemini_results = await asyncio.gather(
                tavily_task, factcheck_task, wikipedia_task, serper_task, gemini_task,
                return_exceptions=True,
            )

            # Handle exceptions from parallel calls
            if isinstance(tavily_results, Exception):
                logger.error(f"Tavily failed: {tavily_results}")
                tavily_results = []
            if isinstance(factcheck_results, Exception):
                logger.error(f"Fact Check failed: {factcheck_results}")
                factcheck_results = []
            if isinstance(wiki_results, Exception):
                logger.error(f"Wikipedia failed: {wiki_results}")
                wiki_results = []
            if isinstance(serper_results, Exception):
                logger.error(f"Serper failed: {serper_results}")
                serper_results = []
            if isinstance(gemini_results, Exception):
                logger.error(f"Gemini failed: {gemini_results}")
                gemini_results = []

            # Combine all evidence
            all_evidence = (
                list(tavily_results)
                + list(factcheck_results)
                + list(wiki_results)
                + list(serper_results)
                + list(gemini_results)
            )

            # Deduplicate
            all_evidence = _deduplicate_evidence(all_evidence)

            # Keep strongest evidence only (speed + quality)
            all_evidence.sort(
                key=lambda e: (float(e.relevance_score) + float(e.credibility_score)),
                reverse=True,
            )
            all_evidence = all_evidence[:10]

            # Classify stance for all items
            # Google Fact Check already has stance from rating, but re-check with
            # heuristic if it came back as neutral (rating text may not have matched)
            for item in all_evidence:
                try:
                    if item.source_type.value == "google_factcheck" and item.stance != "neutral":
                        # Already has a real stance from the rating, keep it
                        continue
                    if item.source_type.value == "gemini" and item.stance != "neutral":
                        # Gemini already classified its own stance, keep it
                        continue
                    if settings.FAST_MODE:
                        item.stance = _infer_stance_fast(claim_text, item)
                    else:
                        item.stance = await _classify_evidence_stance(claim_text, item, llm)
                except Exception:
                    item.stance = "neutral"

            # Build agreement matrix
            agreement_matrix = _build_agreement_matrix(all_evidence)

            # Count stances
            supporting = sum(1 for e in all_evidence if e.stance == "supports")
            refuting = sum(1 for e in all_evidence if e.stance == "refutes")
            neutral = sum(1 for e in all_evidence if e.stance == "neutral")

            evidence_result = EvidenceResult(
                claim_id=claim.id,
                claim_text=claim_text,
                evidence_items=all_evidence,
                agreement_matrix=agreement_matrix,
                total_supporting=supporting,
                total_refuting=refuting,
                total_neutral=neutral,
            )
            evidence_results.append(evidence_result)

            logger.info(
                f"Claim {claim.id}: {len(all_evidence)} evidence items "
                f"(+{supporting}/-{refuting}/~{neutral})"
            )

        state.evidence_results = evidence_results

        state.audit_log.append({
            "node": "evidence_retriever",
            "action": "complete",
            "total_evidence": sum(len(er.evidence_items) for er in evidence_results),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Evidence retrieval failed: {e}")
        state.error = f"Evidence retrieval failed: {str(e)}"
        state.audit_log.append({
            "node": "evidence_retriever",
            "action": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })

    return state

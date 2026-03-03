"""
VeritasAI — Claim Extractor Node
Uses spaCy for sentence segmentation/NER and OpenRouter LLM for claim decomposition.
Implements the Claimify method: pronoun resolution, atomic claim generation, check-worthiness scoring.
"""

import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from backend.utils import strip_thinking_tags
from backend.config import settings
from backend.models.schemas import (
    AtomicClaim,
    ClaimExtractionResult,
    GraphState,
    AnalysisStatus,
)

logger = logging.getLogger(__name__)

# ─── spaCy lazy loader ──────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ─── Prompts ─────────────────────────────────────────────────────────────────

CLAIM_DECOMPOSITION_PROMPT = """You are an expert fact-checking assistant. Your task is to decompose the following text into atomic, testable claims using the Claimify method.

**Instructions:**
1. Break the text into individual sentences.
2. For each sentence, resolve all pronouns to their antecedents.
3. Decompose compound claims into atomic (single-fact) claims.
4. Rate each claim's check-worthiness on a scale from 0.0 to 1.0:
   - 1.0 = Highly check-worthy (statistical claims, historical facts, scientific assertions)
   - 0.5 = Moderately check-worthy (opinions presented as facts, generalizations)
   - 0.0 = Not check-worthy (greetings, subjective opinions clearly marked as such)
5. Only include claims with check-worthiness >= 0.3

**Input Text:**
{text}

**Entities detected by NER:** {entities}

**Output Format (strict JSON):**
{{
  "claims": [
    {{
      "id": 1,
      "original_sentence": "The original sentence from the text",
      "atomic_claim": "The decomposed, pronoun-resolved atomic claim",
      "check_worthiness": 0.85,
      "entities": ["Entity1", "Entity2"]
    }}
  ]
}}

Return ONLY valid JSON, no additional text or markdown formatting.
"""


# ─── Claim Extractor ────────────────────────────────────────────────────────

def _extract_entities(text: str) -> list[dict]:
    """Extract named entities using spaCy."""
    nlp = _get_nlp()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities


def _segment_sentences(text: str) -> list[str]:
    """Segment text into sentences using spaCy."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _fast_check_worthiness(sentence: str) -> float:
    """Lightweight heuristic check-worthiness score for fast mode."""
    s = sentence.strip()
    if not s:
        return 0.0
    lowered = s.lower()
    score = 0.45
    if any(ch.isdigit() for ch in s):
        score += 0.2
    if any(k in lowered for k in ["is", "are", "was", "were", "has", "have", "did", "will"]):
        score += 0.15
    if any(k in lowered for k in ["study", "report", "according", "evidence", "proved", "visible", "causes"]):
        score += 0.1
    return min(1.0, score)


async def extract_claims(state: GraphState) -> GraphState:
    """
    Claim Extractor Node for LangGraph.
    Decomposes input text into atomic, testable claims.
    """
    state.status = AnalysisStatus.EXTRACTING_CLAIMS
    log_entry = {
        "node": "claim_extractor",
        "action": "start",
        "timestamp": datetime.utcnow().isoformat(),
    }
    state.audit_log.append(log_entry)

    try:
        text = state.original_text

        # Step 1: NER with spaCy
        entities = _extract_entities(text)
        entity_strings = list(set(e["text"] for e in entities))
        sentences = _segment_sentences(text)

        logger.info(f"Found {len(entities)} entities, {len(sentences)} sentences")


        # Step 2: Claim decomposition
        claims = []
        if settings.FAST_MODE:
            # Deterministic, low-latency extraction path
            for idx, sent in enumerate(sentences, 1):
                sentence_entities = [e["text"] for e in entities if sent.find(e["text"]) >= 0]
                claim = AtomicClaim(
                    id=idx,
                    original_sentence=sent,
                    atomic_claim=sent,
                    check_worthiness=_fast_check_worthiness(sent),
                    entities=list(dict.fromkeys(sentence_entities)),
                )
                if claim.check_worthiness >= 0.3:
                    claims.append(claim)
        else:
            llm = ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                default_headers={"HTTP-Referer": "https://openrouter.ai/", "X-Title": "VeritasAI"},
            )

            prompt = CLAIM_DECOMPOSITION_PROMPT.format(
                text=text,
                entities=json.dumps(entity_strings),
            )

            response = await llm.ainvoke(prompt)
            response_text = strip_thinking_tags(response.content.strip())

            # Clean potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            parsed = json.loads(response_text.strip())

            # Build AtomicClaim objects
            for c in parsed.get("claims", []):
                claim = AtomicClaim(
                    id=c.get("id", len(claims) + 1),
                    original_sentence=c.get("original_sentence", ""),
                    atomic_claim=c.get("atomic_claim", ""),
                    check_worthiness=float(c.get("check_worthiness", 0.5)),
                    entities=c.get("entities", []),
                )
                if claim.check_worthiness >= 0.3:
                    claims.append(claim)

        # Sort by check-worthiness (highest first)
        claims.sort(key=lambda x: x.check_worthiness, reverse=True)

        # Limit number of claims for latency control
        claims = claims[: max(1, settings.MAX_CLAIMS)]

        # Re-index
        for i, claim in enumerate(claims):
            claim.id = i + 1

        state.extraction_result = ClaimExtractionResult(
            claims=claims,
            total_sentences=len(sentences),
            total_claims=len(claims),
        )

        state.audit_log.append({
            "node": "claim_extractor",
            "action": "complete",
            "claims_extracted": len(claims),
            "sentences_processed": len(sentences),
            "entities_found": len(entities),
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(f"Extracted {len(claims)} check-worthy claims")

    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        state.error = f"Claim extraction failed: {str(e)}"
        state.audit_log.append({
            "node": "claim_extractor",
            "action": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })

    return state

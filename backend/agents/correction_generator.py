"""
VeritasAI — Correction Generator Node
Generates accurate rewrites with factual corrections, source links, and bias/framing analysis.
"""

import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from backend.utils import strip_thinking_tags
from backend.config import settings
from backend.models.schemas import (
    GraphState,
    CorrectionResult,
    AnalysisStatus,
)

logger = logging.getLogger(__name__)


# ─── Fast-mode bias helpers ──────────────────────────────────────────────────

def _detect_loaded_language(text: str) -> list[str]:
    """Detect loaded/biased language in a claim."""
    loaded_words = [
        "allegedly", "supposedly", "claimed", "so-called", "radical",
        "extreme", "shocking", "horrifying", "devastating", "massive",
        "unprecedented", "explosive", "bombshell", "outrageous",
        "scandalous", "disastrous", "catastrophic", "alarming",
    ]
    found = [w for w in loaded_words if w in text.lower()]
    return found


def _detect_emotional_manipulation(text: str) -> str:
    """Check for emotional manipulation tactics."""
    text_lower = text.lower()
    tactics = []
    if any(w in text_lower for w in ["killed", "murdered", "died", "death"]):
        tactics.append("fear/shock appeal")
    if any(w in text_lower for w in ["everyone knows", "obviously", "clearly"]):
        tactics.append("false consensus")
    if any(w in text_lower for w in ["always", "never", "all", "none"]):
        tactics.append("absolutist framing")
    if any(w in text_lower for w in ["they", "them", "those people"]):
        tactics.append("us-vs-them framing")
    return "; ".join(tactics) if tactics else "No strong emotional manipulation detected"


def _estimate_bias_score(text: str) -> float:
    """Estimate an overall bias score (0-1)."""
    loaded = _detect_loaded_language(text)
    score = min(1.0, len(loaded) * 0.15)
    text_lower = text.lower()
    if any(w in text_lower for w in ["killed", "murdered", "attacked", "bombed"]):
        score += 0.2
    if any(w in text_lower for w in ["always", "never", "all", "every", "only"]):
        score += 0.1
    return min(1.0, round(score, 2))

CORRECTION_PROMPT = """You are an expert fact-checker and editor. Given a claim that has been verified, generate a factual correction and bias analysis.

**ORIGINAL CLAIM:** {claim}

**VERDICT:** {verdict} (Confidence: {confidence}%)

**KEY EVIDENCE:**
{evidence}

**REASONING:** {reasoning}

**Instructions:**
1. If the claim is REFUTED or LIKELY_REFUTED, rewrite it to be factually accurate
2. If the claim is SUPPORTED, confirm it with additional context
3. If INSUFFICIENT_EVIDENCE, note what additional information would be needed
4. Analyze the original claim for bias, loaded language, or emotional manipulation
5. Provide source links for every correction made

**Output Format (strict JSON):**
{{
    "corrected_text": "The factually corrected version of the claim (or confirmation if accurate)",
    "corrections_made": [
        {{
            "original_fragment": "The part of the claim that was incorrect",
            "corrected_fragment": "The factually accurate version",
            "explanation": "Why this correction was needed",
            "source_url": "URL supporting this correction"
        }}
    ],
    "bias_analysis": {{
        "loaded_language": ["List of loaded/biased words or phrases found"],
        "emotional_manipulation": "Description of any emotional manipulation tactics",
        "framing_bias": "Description of how the framing may bias the reader",
        "overall_bias_score": 0.3,
        "bias_direction": "neutral|left|right|sensationalist|minimizing"
    }},
    "source_links": ["url1", "url2"]
}}

Return ONLY valid JSON, no additional text or markdown.
"""


async def generate_corrections(state: GraphState) -> GraphState:
    """
    Correction Generator Node for LangGraph.
    Generates factual corrections, bias analysis, and source-backed rewrites.
    """
    state.status = AnalysisStatus.GENERATING_CORRECTION

    state.audit_log.append({
        "node": "correction_generator",
        "action": "start",
        "timestamp": datetime.utcnow().isoformat(),
    })

    if not state.verdict_results:
        logger.warning("No verdicts to generate corrections for")
        return state

    if settings.FAST_MODE:
        correction_results = []
        for verdict in state.verdict_results:
            matching_evidence = next((er for er in state.evidence_results if er.claim_id == verdict.claim_id), None)
            urls = [e.url for e in (matching_evidence.evidence_items if matching_evidence else []) if e.url][:5]

            # Build meaningful corrections from evidence
            corrections_made = []
            evidence_snippets = []
            if matching_evidence:
                for item in matching_evidence.evidence_items[:5]:
                    if item.snippet:
                        evidence_snippets.append(item.snippet[:200])

            if verdict.verdict.value in ("REFUTED", "LIKELY_REFUTED"):
                # Extract key facts from evidence to build a correction
                # Prefer authoritative sources (Tavily AI Answer, Google Fact Check) for corrections
                all_items = matching_evidence.evidence_items if matching_evidence else []
                refuting_items = [e for e in all_items if e.stance == "refutes"]
                neutral_items = [e for e in all_items if e.stance == "neutral"]

                # Find best correction source: authoritative first, then highest credibility
                authoritative_refuting = [
                    e for e in refuting_items
                    if e.source_type.value == "google_factcheck"
                    or e.source_type.value == "gemini"
                    or (e.source_type.value == "tavily" and "AI Answer" in e.title)
                ]
                best_evidence = authoritative_refuting or refuting_items or neutral_items

                if best_evidence:
                    # Use the most credible evidence snippet as the correction basis
                    best = max(best_evidence, key=lambda e: e.credibility_score)
                    snippet_clean = best.snippet[:300].strip()

                    corrected_text = (
                        f"⚠️ This claim is FALSE. "
                        f"According to {best.source_type.value.replace('_', ' ').title()}: "
                        f'"{snippet_clean}"'
                    )

                    corrections_made.append({
                        "original_fragment": verdict.claim_text,
                        "corrected_fragment": snippet_clean[:200],
                        "explanation": f"The original claim is contradicted by evidence from {best.source_type.value}",
                        "source_url": best.url or "",
                    })
                else:
                    corrected_text = (
                        f"⚠️ This claim is FALSE. "
                        f"No credible source supports this assertion. "
                        f"The claim appears to be fabricated or unverified misinformation."
                    )

                    corrections_made.append({
                        "original_fragment": verdict.claim_text,
                        "corrected_fragment": "No credible evidence supports this claim",
                        "explanation": "No reputable source was found that corroborates this claim",
                        "source_url": "",
                    })

            elif verdict.verdict.value in ("SUPPORTED", "LIKELY_SUPPORTED"):
                supporting_items = [
                    e for e in (matching_evidence.evidence_items if matching_evidence else [])
                    if e.stance == "supports"
                ]
                if supporting_items:
                    best = max(supporting_items, key=lambda e: e.credibility_score)
                    corrected_text = (
                        f"✅ This claim is CORRECT. "
                        f"Confirmed by {best.source_type.value.replace('_', ' ').title()}: "
                        f'"{best.snippet[:300].strip()}"'
                    )
                else:
                    corrected_text = (
                        f"✅ This claim appears to be correct based on available evidence."
                    )
            else:
                corrected_text = (
                    f"⚪ Evidence is currently insufficient to fully verify or refute this claim. "
                    f"Additional fact-checking from primary sources is recommended."
                )

            correction_results.append(
                CorrectionResult(
                    claim_id=verdict.claim_id,
                    original_text=verdict.claim_text,
                    corrected_text=corrected_text,
                    corrections_made=corrections_made,
                    bias_analysis={
                        "loaded_language": _detect_loaded_language(verdict.claim_text),
                        "emotional_manipulation": _detect_emotional_manipulation(verdict.claim_text),
                        "framing_bias": "Not fully assessed in fast mode",
                        "overall_bias_score": _estimate_bias_score(verdict.claim_text),
                        "bias_direction": "neutral",
                    },
                    source_links=urls,
                    clarification_notes=[
                        "Interpret this result with uncertainty context and confidence band",
                        "Review conflicting evidence and ambiguity notes before final use",
                    ] + (verdict.ambiguity_notes[:2] if hasattr(verdict, "ambiguity_notes") else []),
                )
            )

        state.correction_results = correction_results
        state.audit_log.append({
            "node": "correction_generator",
            "action": "complete_fast_mode",
            "corrections": len(correction_results),
            "timestamp": datetime.utcnow().isoformat(),
        })
        return state

    try:
        llm = ChatOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            default_headers={"HTTP-Referer": "https://openrouter.ai/", "X-Title": "VeritasAI"},
        )

        correction_results = []

        for verdict in state.verdict_results:
            # Find matching evidence
            matching_evidence = None
            for er in state.evidence_results:
                if er.claim_id == verdict.claim_id:
                    matching_evidence = er
                    break

            evidence_text = ""
            source_urls = []
            if matching_evidence:
                for i, item in enumerate(matching_evidence.evidence_items[:6], 1):
                    evidence_text += (
                        f"[Source {i}] {item.title}\n"
                        f"  URL: {item.url}\n"
                        f"  Content: {item.snippet[:200]}\n\n"
                    )
                    if item.url:
                        source_urls.append(item.url)

            # Find matching explanation
            reasoning_text = verdict.reasoning_chain
            for expl in state.explanation_results:
                if expl.claim_id == verdict.claim_id:
                    reasoning_text = "\n".join(expl.step_by_step_reasoning)
                    break

            prompt = CORRECTION_PROMPT.format(
                claim=verdict.claim_text,
                verdict=verdict.verdict.value,
                confidence=verdict.confidence,
                evidence=evidence_text or "No evidence available.",
                reasoning=reasoning_text,
            )

            response = await llm.ainvoke(prompt)
            response_text = strip_thinking_tags(response.content.strip())

            # Clean markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            try:
                parsed = json.loads(response_text.strip())
            except json.JSONDecodeError:
                logger.error("Failed to parse correction JSON")
                parsed = {
                    "corrected_text": verdict.claim_text,
                    "corrections_made": [],
                    "bias_analysis": {
                        "loaded_language": [],
                        "emotional_manipulation": "Unable to analyze",
                        "framing_bias": "Unable to analyze",
                        "overall_bias_score": 0.0,
                        "bias_direction": "neutral",
                    },
                    "source_links": source_urls,
                    "clarification_notes": [
                        "Insufficient structured correction output; rely on evidence citations",
                        "Collect additional context before operational decision-making",
                    ],
                }

            correction = CorrectionResult(
                claim_id=verdict.claim_id,
                original_text=verdict.claim_text,
                corrected_text=parsed.get("corrected_text", verdict.claim_text),
                corrections_made=parsed.get("corrections_made", []),
                bias_analysis=parsed.get("bias_analysis", {}),
                source_links=parsed.get("source_links", source_urls),
                clarification_notes=parsed.get("clarification_notes", []),
            )
            correction_results.append(correction)

        state.correction_results = correction_results

        state.audit_log.append({
            "node": "correction_generator",
            "action": "complete",
            "corrections": len(correction_results),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Correction generation failed: {e}")
        state.error = f"Correction generation failed: {str(e)}"
        state.audit_log.append({
            "node": "correction_generator",
            "action": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })

    return state

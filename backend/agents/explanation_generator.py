"""
VeritasAI — Explanation Generator Node
Produces transparent reasoning chains with inline evidence citations.
Shows step-by-step logic: claim → evidence → reasoning → verdict.
"""

import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from backend.utils import strip_thinking_tags
from backend.config import settings
from backend.models.schemas import (
    GraphState,
    ExplanationResult,
    AnalysisStatus,
)

logger = logging.getLogger(__name__)

EXPLANATION_PROMPT = """You are an expert fact-checker creating a transparent, step-by-step explanation of how a verdict was reached for a claim.

**CLAIM:** {claim}

**VERDICT:** {verdict} (Confidence: {confidence}%)

**REASONING CHAIN FROM CLASSIFIER:** {reasoning}

**KEY EVIDENCE:**
{evidence}

**CONFLICTING EVIDENCE:**
{conflicts}

**Instructions:**
Create a detailed, transparent explanation that includes:
1. Step-by-step reasoning with numbered steps
2. Inline citations referencing specific evidence [Source N]
3. Clear documentation of assumptions made
4. Explicit mention of limitations
5. Acknowledgment of any conflicting evidence
6. Recommended next checks when uncertainty remains

**Output Format (strict JSON):**
{{
    "step_by_step_reasoning": [
        "Step 1: [Analysis of the claim's core assertion]",
        "Step 2: [Evidence from Source 1 indicates... [Source 1]]",
        "Step 3: [Cross-referencing with Source 2 shows... [Source 2]]",
        "Step 4: [Weighing the evidence, considering credibility scores...]",
        "Step 5: [Final determination based on the preponderance of evidence]"
    ],
    "evidence_citations": [
        {{"source": "Source name", "url": "url", "relevance": "How this evidence was used"}},
        {{"source": "Source name", "url": "url", "relevance": "How this evidence was used"}}
    ],
    "assumptions": [
        "Relying on web sources available as of analysis date",
        "English-language content only",
        "Claims evaluated independently"
    ],
    "limitations": [
        "Limited to text-based verification",
        "Cannot verify images or videos",
        "Source credibility scores are estimated"
    ],
    "recommended_next_checks": [
        "Check primary-source documents from official organizations",
        "Verify whether the claim is time-bound or region-specific",
        "Cross-check against at least one independent fact-checking source"
    ]
}}

Return ONLY valid JSON, no additional text or markdown.
"""


async def generate_explanations(state: GraphState) -> GraphState:
    """
    Explanation Generator Node for LangGraph.
    Creates transparent reasoning chains with citations.
    """
    state.status = AnalysisStatus.GENERATING_EXPLANATION

    state.audit_log.append({
        "node": "explanation_generator",
        "action": "start",
        "timestamp": datetime.utcnow().isoformat(),
    })

    if not state.verdict_results:
        logger.warning("No verdicts to explain")
        return state

    if settings.FAST_MODE:
        explanation_results = []
        for verdict in state.verdict_results:
            matching_evidence = next((er for er in state.evidence_results if er.claim_id == verdict.claim_id), None)
            citations = []
            if matching_evidence:
                for item in matching_evidence.evidence_items[:4]:
                    citations.append({
                        "source": item.source_type.value,
                        "url": item.url,
                        "relevance": f"stance={item.stance}, credibility={item.credibility_score:.0%}",
                    })

            explanation_results.append(
                ExplanationResult(
                    claim_id=verdict.claim_id,
                    claim_text=verdict.claim_text,
                    step_by_step_reasoning=[
                        f"Step 1: Retrieved evidence from {int(matching_evidence.agreement_matrix.get('source_diversity', 0)) if matching_evidence else 0} independent sources ({len(matching_evidence.evidence_items) if matching_evidence else 0} items total).",
                        f"Step 2: Classified stances — {matching_evidence.total_supporting if matching_evidence else 0} supporting, {matching_evidence.total_refuting if matching_evidence else 0} refuting, {matching_evidence.total_neutral if matching_evidence else 0} neutral.",
                        f"Step 3: Applied veracity mapping for verdict {verdict.verdict.value} at {verdict.confidence:.0f}% confidence.",
                        f"Step 4: {'Debate between advocate and skeptic agents further validated the verdict.' if '[DEBATE ENHANCED]' in (verdict.reasoning_chain or '') else 'Evidence distribution used for final determination.'}",
                    ],
                    evidence_citations=citations,
                    assumptions=[
                        "Evidence is limited to currently reachable web sources",
                        "Fast mode uses deterministic aggregation for latency",
                    ],
                    limitations=[
                        "No long-form LLM explanation generation in fast mode",
                        "Coverage may be reduced due to evidence limits",
                    ],
                    recommended_next_checks=[
                        "Increase source diversity (≥2 independent source types)",
                        "Look for primary-source or official documentation",
                        "Re-run with debate enabled for controversial claims",
                    ],
                )
            )

        state.explanation_results = explanation_results
        state.audit_log.append({
            "node": "explanation_generator",
            "action": "complete_fast_mode",
            "explanations": len(explanation_results),
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

        explanation_results = []

        for verdict in state.verdict_results:
            # Find matching evidence
            matching_evidence = None
            for er in state.evidence_results:
                if er.claim_id == verdict.claim_id:
                    matching_evidence = er
                    break

            evidence_text = ""
            if matching_evidence:
                for i, item in enumerate(matching_evidence.evidence_items[:8], 1):
                    evidence_text += (
                        f"[Source {i}] {item.source_type.value.upper()} — {item.title}\n"
                        f"  URL: {item.url}\n"
                        f"  Snippet: {item.snippet[:200]}\n"
                        f"  Credibility: {item.credibility_score:.0%} | Stance: {item.stance}\n\n"
                    )

            conflicts_text = "\n".join(
                f"- {c}" for c in verdict.conflicting_evidence
            ) if verdict.conflicting_evidence else "No conflicting evidence detected."

            prompt = EXPLANATION_PROMPT.format(
                claim=verdict.claim_text,
                verdict=verdict.verdict.value,
                confidence=verdict.confidence,
                reasoning=verdict.reasoning_chain,
                evidence=evidence_text or "No evidence available.",
                conflicts=conflicts_text,
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
                logger.error(f"Failed to parse explanation JSON")
                parsed = {
                    "step_by_step_reasoning": [verdict.reasoning_chain],
                    "evidence_citations": [],
                    "assumptions": ["Relying on web sources available as of analysis date"],
                    "limitations": ["Explanation parsing failed, showing raw reasoning"],
                    "recommended_next_checks": [
                        "Validate against official or primary sources",
                        "Collect more independent evidence before final judgment",
                    ],
                }

            explanation = ExplanationResult(
                claim_id=verdict.claim_id,
                claim_text=verdict.claim_text,
                step_by_step_reasoning=parsed.get("step_by_step_reasoning", []),
                evidence_citations=parsed.get("evidence_citations", []),
                assumptions=parsed.get("assumptions", []),
                limitations=parsed.get("limitations", []),
                recommended_next_checks=parsed.get("recommended_next_checks", []),
            )
            explanation_results.append(explanation)

        state.explanation_results = explanation_results

        state.audit_log.append({
            "node": "explanation_generator",
            "action": "complete",
            "explanations": len(explanation_results),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        state.error = f"Explanation generation failed: {str(e)}"
        state.audit_log.append({
            "node": "explanation_generator",
            "action": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })

    return state

"""
VeritasAI — Multi-Agent Debate Feature
Two LLM agents argue for/against a claim, a judge agent decides.
Based on "Can LLMs Produce Faithful Explanations For Fact-checking via Multi-Agent Debate" (2024).
"""

import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from backend.utils import strip_thinking_tags
from backend.config import settings
from backend.models.schemas import (
    GraphState,
    DebateResult,
    DebateArgument,
    VerdictLabel,
    AnalysisStatus,
)

logger = logging.getLogger(__name__)

# ─── Debate Prompts ──────────────────────────────────────────────────────────

ADVOCATE_PROMPT = """You are the ADVOCATE agent in a fact-checking debate. Your role is to argue that the following claim is TRUE and SUPPORTED by evidence.

**CLAIM:** {claim}

**AVAILABLE EVIDENCE:**
{evidence}

**Previous debate arguments:**
{previous_args}

**Instructions:**
- Make the strongest possible case that this claim is true
- Cite specific evidence to support your argument
- Address any counterarguments from the skeptic
- Be intellectually honest — if the evidence genuinely doesn't support the claim, acknowledge weaknesses

**Output Format (strict JSON):**
{{
    "argument": "Your detailed argument for why this claim is supported",
    "evidence_cited": ["Key evidence point 1", "Key evidence point 2"],
    "confidence": 0.75
}}

Return ONLY valid JSON.
"""

SKEPTIC_PROMPT = """You are the SKEPTIC agent in a fact-checking debate. Your role is to argue that the following claim is FALSE or UNSUPPORTED by evidence.

**CLAIM:** {claim}

**AVAILABLE EVIDENCE:**
{evidence}

**Previous debate arguments:**
{previous_args}

**Instructions:**
- Make the strongest possible case that this claim is false or unsupported
- Cite specific evidence that contradicts the claim
- Point out weaknesses in the advocate's arguments
- Be intellectually honest — if the evidence genuinely supports the claim, acknowledge strengths

**Output Format (strict JSON):**
{{
    "argument": "Your detailed argument for why this claim is refuted or unsupported",
    "evidence_cited": ["Key evidence point 1", "Key evidence point 2"],
    "confidence": 0.75
}}

Return ONLY valid JSON.
"""

JUDGE_PROMPT = """You are the JUDGE agent in a fact-checking debate. You must weigh both sides of the debate and render a final verdict.

**CLAIM:** {claim}

**ADVOCATE'S ARGUMENTS (arguing claim is TRUE):**
{advocate_args}

**SKEPTIC'S ARGUMENTS (arguing claim is FALSE):**
{skeptic_args}

**ORIGINAL CLASSIFIER VERDICT:** {original_verdict} (Confidence: {original_confidence}%)

**AVAILABLE EVIDENCE:**
{evidence}

**Instructions:**
1. Carefully weigh both sides of the debate
2. Consider the quality and specificity of arguments
3. Consider the evidence cited by each side
4. Determine whether the debate changes the original verdict
5. Use the 5-level veracity scale

**Output Format (strict JSON):**
{{
    "verdict": "SUPPORTED|LIKELY_SUPPORTED|INSUFFICIENT_EVIDENCE|LIKELY_REFUTED|REFUTED",
    "confidence": 80.0,
    "reasoning": "Detailed reasoning explaining why you chose this verdict after considering both sides",
    "debate_enhanced": true
}}

Return ONLY valid JSON.
"""


async def run_debate(state: GraphState) -> GraphState:
    """
    Multi-Agent Debate Node for LangGraph.
    Two agents argue for/against each claim, a judge decides.
    Supports both FAST_MODE (deterministic) and LLM-based debate.
    """
    if not state.enable_debate:
        logger.info("Debate disabled, skipping")
        return state

    state.status = AnalysisStatus.DEBATING

    state.audit_log.append({
        "node": "debate_agents",
        "action": "start",
        "timestamp": datetime.utcnow().isoformat(),
    })

    if not state.verdict_results or not state.evidence_results:
        logger.warning("No verdicts or evidence for debate")
        return state

    # ── FAST_MODE: deterministic debate using evidence analysis ──────
    if settings.FAST_MODE:
        debate_results = []

        for verdict in state.verdict_results:
            # Find evidence for this claim
            matching_evidence = next(
                (er for er in state.evidence_results if er.claim_id == verdict.claim_id), None
            )
            if not matching_evidence:
                continue

            items = matching_evidence.evidence_items
            supporting_items = [e for e in items if e.stance == "supports"]
            refuting_items = [e for e in items if e.stance == "refutes"]
            neutral_items = [e for e in items if e.stance == "neutral"]

            # Build Advocate arguments from supporting + neutral evidence
            advocate_evidence = supporting_items or neutral_items[:2]
            advocate_arg_text = (
                f"Based on {len(supporting_items)} supporting evidence item(s), "
                f"the claim appears to have factual basis. "
            )
            if supporting_items:
                advocate_arg_text += "Key supporting sources: " + "; ".join(
                    f'"{e.title}" ({e.source_type.value}, credibility {e.credibility_score:.0%})'
                    for e in supporting_items[:3]
                )
            else:
                advocate_arg_text += (
                    "However, no direct supporting evidence was found. "
                    "The advocate notes that absence of refutation is not the same as refutation."
                )
            advocate_conf = min(0.9, 0.3 + len(supporting_items) * 0.15)

            advocate_args = [
                DebateArgument(
                    agent_role="advocate",
                    argument=advocate_arg_text,
                    evidence_cited=[e.title for e in advocate_evidence[:3]],
                    confidence=advocate_conf,
                ),
                DebateArgument(
                    agent_role="advocate",
                    argument=(
                        f"In round 2, the advocate maintains that "
                        f"{'the supporting evidence from credible sources (' + str(len(supporting_items)) + ' items) outweighs concerns' if supporting_items else 'the claim should not be dismissed without strong counter-evidence'}. "
                        f"Source diversity across {matching_evidence.agreement_matrix.get('source_diversity', 0)} source types adds reliability."
                    ),
                    evidence_cited=[e.title for e in supporting_items[:2]],
                    confidence=advocate_conf,
                ),
            ]

            # Build Skeptic arguments from refuting + contradicting evidence
            skeptic_evidence = refuting_items or neutral_items[:2]
            skeptic_arg_text = (
                f"Based on {len(refuting_items)} refuting evidence item(s), "
                f"this claim faces significant credibility challenges. "
            )
            if refuting_items:
                skeptic_arg_text += "Key refuting sources: " + "; ".join(
                    f'"{e.title}" ({e.source_type.value}, credibility {e.credibility_score:.0%})'
                    for e in refuting_items[:3]
                )
            else:
                skeptic_arg_text += (
                    "While no direct refutation was found, the skeptic notes the "
                    "absence of credible supporting evidence is itself a concern, "
                    "especially for extraordinary claims."
                )
            skeptic_conf = min(0.9, 0.3 + len(refuting_items) * 0.15)

            skeptic_args = [
                DebateArgument(
                    agent_role="skeptic",
                    argument=skeptic_arg_text,
                    evidence_cited=[e.title for e in skeptic_evidence[:3]],
                    confidence=skeptic_conf,
                ),
                DebateArgument(
                    agent_role="skeptic",
                    argument=(
                        f"In round 2, the skeptic emphasizes that "
                        f"{'the refuting evidence is strong and comes from credible sources' if refuting_items else 'no credible source corroborates this claim'}. "
                        f"The contradiction index is {matching_evidence.agreement_matrix.get('contradiction_index', 0):.2f} "
                        f"and evidence robustness is {matching_evidence.agreement_matrix.get('robustness_score', 0):.2f}."
                    ),
                    evidence_cited=[e.title for e in refuting_items[:2]],
                    confidence=skeptic_conf,
                ),
            ]

            # Judge decision based on evidence balance
            s_count = len(supporting_items)
            r_count = len(refuting_items)

            # Check for authoritative sources (Tavily AI Answer, Google Fact Check)
            auth_refuting = sum(1 for e in refuting_items if (
                e.source_type.value == "google_factcheck" or
                e.source_type.value == "gemini" or
                (e.source_type.value == "tavily" and "AI Answer" in e.title)
            ))
            auth_supporting = sum(1 for e in supporting_items if (
                e.source_type.value == "google_factcheck" or
                e.source_type.value == "gemini" or
                (e.source_type.value == "tavily" and "AI Answer" in e.title)
            ))

            # Authoritative sources carry decisive weight
            if auth_refuting > 0 and auth_supporting == 0:
                # If authoritative + regular refuting evidence, boost to REFUTED
                total_refuting_weight = auth_refuting * 2 + r_count  # Auth sources count double
                if total_refuting_weight >= 3 or auth_refuting >= 2:
                    judge_verdict = VerdictLabel.REFUTED
                else:
                    judge_verdict = VerdictLabel.LIKELY_REFUTED
                judge_confidence = 78.0 + min(15.0, total_refuting_weight * 3.0)
                judge_reasoning = (
                    f"After weighing both sides, {auth_refuting} authoritative source(s) "
                    f"(Fact Check / AI Answer) directly refute the claim. "
                    f"While {s_count} generic evidence items appear to support, "
                    f"authoritative fact-check sources take precedence as they specifically address the claim."
                )
                debate_enhanced = verdict.verdict != judge_verdict
            elif auth_supporting > 0 and auth_refuting == 0 and r_count == 0:
                judge_verdict = VerdictLabel.SUPPORTED if auth_supporting >= 2 else VerdictLabel.LIKELY_SUPPORTED
                judge_confidence = 78.0 + min(15.0, auth_supporting * 5.0)
                judge_reasoning = (
                    f"After weighing both sides, {auth_supporting} authoritative source(s) "
                    f"directly confirm the claim with no contradicting evidence."
                )
                debate_enhanced = verdict.verdict != judge_verdict
            elif r_count > s_count:
                judge_verdict = VerdictLabel.REFUTED if r_count >= 2 else VerdictLabel.LIKELY_REFUTED
                judge_confidence = 65.0 + min(25.0, r_count * 5.0)
                judge_reasoning = (
                    f"After weighing both sides, the refuting evidence ({r_count} items) "
                    f"outweighs supporting evidence ({s_count} items). "
                    f"The skeptic's arguments are more substantiated by credible sources."
                )
                debate_enhanced = verdict.verdict != judge_verdict
            elif s_count > r_count:
                judge_verdict = VerdictLabel.SUPPORTED if s_count >= 2 else VerdictLabel.LIKELY_SUPPORTED
                judge_confidence = 65.0 + min(25.0, s_count * 5.0)
                judge_reasoning = (
                    f"After weighing both sides, the supporting evidence ({s_count} items) "
                    f"outweighs refuting evidence ({r_count} items). "
                    f"The advocate's arguments are better substantiated."
                )
                debate_enhanced = verdict.verdict != judge_verdict
            elif s_count == 0 and r_count == 0:
                # No supporting or refuting → check if claim is extraordinary
                from backend.agents.veracity_classifier import _is_extraordinary_claim
                is_extra, category = _is_extraordinary_claim(verdict.claim_text)
                if is_extra:
                    judge_verdict = VerdictLabel.LIKELY_REFUTED
                    judge_confidence = 65.0
                    judge_reasoning = (
                        f"Neither side presented strong evidence. However, this is an extraordinary "
                        f"claim ({category}) with no corroborating evidence from any source. "
                        f"Extraordinary claims require extraordinary evidence, which is absent."
                    )
                    debate_enhanced = True
                else:
                    judge_verdict = verdict.verdict
                    judge_confidence = verdict.confidence
                    judge_reasoning = (
                        "Neither side presented compelling evidence. "
                        "The original verdict stands due to insufficient evidence on both sides."
                    )
                    debate_enhanced = False
            else:
                judge_verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
                judge_confidence = 45.0
                judge_reasoning = (
                    f"Evidence is evenly split ({s_count} supporting vs {r_count} refuting). "
                    f"Neither side conclusively wins the debate. More evidence is needed."
                )
                debate_enhanced = verdict.verdict != judge_verdict

            debate_result = DebateResult(
                claim_id=verdict.claim_id,
                claim_text=verdict.claim_text,
                advocate_arguments=advocate_args,
                skeptic_arguments=skeptic_args,
                judge_verdict=judge_verdict,
                judge_reasoning=judge_reasoning,
                judge_confidence=judge_confidence,
                debate_enhanced_verdict=debate_enhanced,
            )
            debate_results.append(debate_result)

            # Update verdict if debate changed it
            if debate_enhanced:
                verdict.verdict = judge_verdict
                verdict.confidence = judge_confidence
                verdict.reasoning_chain += (
                    f"\n\n[DEBATE ENHANCED] After multi-agent debate: {judge_reasoning}"
                )

            logger.info(
                f"Fast debate for claim {verdict.claim_id}: "
                f"Judge verdict = {judge_verdict.value} "
                f"(enhanced: {debate_enhanced})"
            )

        state.debate_results = debate_results
        state.audit_log.append({
            "node": "debate_agents",
            "action": "complete_fast_mode",
            "claims_debated": len(debate_results),
            "timestamp": datetime.utcnow().isoformat(),
        })
        return state

    # ── Standard LLM-based debate ────────────────────────────────────

    # ── Standard LLM-based debate ────────────────────────────────────

    try:
        llm = ChatOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=0.3,  # Slightly higher temp for diverse arguments
            max_tokens=settings.LLM_MAX_TOKENS,
            default_headers={"HTTP-Referer": "https://openrouter.ai/", "X-Title": "VeritasAI"},
        )

        debate_results = []

        # Debate all claims when debate is enabled
        claims_to_debate = list(state.verdict_results)

        for verdict in claims_to_debate:
            # Find evidence
            evidence_text = ""
            for er in state.evidence_results:
                if er.claim_id == verdict.claim_id:
                    for i, item in enumerate(er.evidence_items[:6], 1):
                        evidence_text += (
                            f"[{i}] {item.source_type.value}: {item.title}\n"
                            f"    {item.snippet[:200]}\n"
                            f"    Stance: {item.stance} | Credibility: {item.credibility_score:.0%}\n\n"
                        )
                    break

            # Run 2 rounds of debate
            advocate_args = []
            skeptic_args = []

            for round_num in range(2):
                previous_args = ""
                if advocate_args or skeptic_args:
                    for aa in advocate_args:
                        previous_args += f"ADVOCATE: {aa.argument}\n"
                    for sa in skeptic_args:
                        previous_args += f"SKEPTIC: {sa.argument}\n"

                # Advocate turn
                advocate_prompt = ADVOCATE_PROMPT.format(
                    claim=verdict.claim_text,
                    evidence=evidence_text or "No evidence available.",
                    previous_args=previous_args or "None yet (first round).",
                )
                advocate_response = await llm.ainvoke(advocate_prompt)
                advocate_parsed = _parse_debate_response(strip_thinking_tags(advocate_response.content))

                advocate_arg = DebateArgument(
                    agent_role="advocate",
                    argument=advocate_parsed.get("argument", ""),
                    evidence_cited=advocate_parsed.get("evidence_cited", []),
                    confidence=float(advocate_parsed.get("confidence", 0.5)),
                )
                advocate_args.append(advocate_arg)

                # Skeptic turn
                previous_args += f"ADVOCATE (Round {round_num + 1}): {advocate_arg.argument}\n"

                skeptic_prompt = SKEPTIC_PROMPT.format(
                    claim=verdict.claim_text,
                    evidence=evidence_text or "No evidence available.",
                    previous_args=previous_args,
                )
                skeptic_response = await llm.ainvoke(skeptic_prompt)
                skeptic_parsed = _parse_debate_response(strip_thinking_tags(skeptic_response.content))

                skeptic_arg = DebateArgument(
                    agent_role="skeptic",
                    argument=skeptic_parsed.get("argument", ""),
                    evidence_cited=skeptic_parsed.get("evidence_cited", []),
                    confidence=float(skeptic_parsed.get("confidence", 0.5)),
                )
                skeptic_args.append(skeptic_arg)

            # Judge's verdict
            advocate_text = "\n".join(
                f"Round {i + 1}: {a.argument}" for i, a in enumerate(advocate_args)
            )
            skeptic_text = "\n".join(
                f"Round {i + 1}: {s.argument}" for i, s in enumerate(skeptic_args)
            )

            judge_prompt = JUDGE_PROMPT.format(
                claim=verdict.claim_text,
                advocate_args=advocate_text,
                skeptic_args=skeptic_text,
                original_verdict=verdict.verdict.value,
                original_confidence=verdict.confidence,
                evidence=evidence_text or "No evidence available.",
            )
            judge_response = await llm.ainvoke(judge_prompt)
            judge_parsed = _parse_debate_response(strip_thinking_tags(judge_response.content))

            # Parse judge verdict
            judge_verdict_str = judge_parsed.get("verdict", "INSUFFICIENT_EVIDENCE").upper().replace(" ", "_")
            try:
                judge_verdict = VerdictLabel(judge_verdict_str)
            except ValueError:
                judge_verdict = verdict.verdict

            debate_result = DebateResult(
                claim_id=verdict.claim_id,
                claim_text=verdict.claim_text,
                advocate_arguments=advocate_args,
                skeptic_arguments=skeptic_args,
                judge_verdict=judge_verdict,
                judge_reasoning=judge_parsed.get("reasoning", ""),
                judge_confidence=float(judge_parsed.get("confidence", verdict.confidence)),
                debate_enhanced_verdict=judge_parsed.get("debate_enhanced", False),
            )
            debate_results.append(debate_result)

            # Update the verdict if the debate changed it
            if debate_result.debate_enhanced_verdict:
                verdict.verdict = judge_verdict
                verdict.confidence = debate_result.judge_confidence
                verdict.reasoning_chain += (
                    f"\n\n[DEBATE ENHANCED] After multi-agent debate: "
                    f"{judge_parsed.get('reasoning', '')}"
                )

            logger.info(
                f"Debate for claim {verdict.claim_id}: "
                f"Judge verdict = {judge_verdict.value} "
                f"(enhanced: {debate_result.debate_enhanced_verdict})"
            )

        state.debate_results = debate_results

        state.audit_log.append({
            "node": "debate_agents",
            "action": "complete",
            "claims_debated": len(debate_results),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Multi-agent debate failed: {e}")
        state.audit_log.append({
            "node": "debate_agents",
            "action": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Don't set state.error — debate failure is non-fatal

    return state


def _parse_debate_response(response_text: str) -> dict:
    """Parse debate agent response, handling markdown and malformed JSON."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {
            "argument": response_text[:500],
            "evidence_cited": [],
            "confidence": 0.5,
        }

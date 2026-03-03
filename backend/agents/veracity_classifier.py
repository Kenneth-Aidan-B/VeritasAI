"""
VeritasAI — Veracity Classifier Node
Classifies each claim on a 5-level veracity scale with calibrated confidence
using chain-of-thought (CoT) prompting with OpenRouter LLM.
"""

import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from backend.utils import strip_thinking_tags
from backend.config import settings
from backend.models.schemas import (
    GraphState,
    VerdictResult,
    VerdictLabel,
    AnalysisStatus,
)

logger = logging.getLogger(__name__)


def _confidence_band(confidence: float) -> dict:
    """Create a calibrated confidence interval for transparency."""
    c = max(0.0, min(100.0, float(confidence)))
    spread = 8.0 if c >= 80 else (12.0 if c >= 60 else 18.0)
    level = "high" if c >= 80 else ("medium" if c >= 60 else "low")
    return {
        "low": max(0.0, round(c - spread, 1)),
        "high": min(100.0, round(c + spread, 1)),
        "level": level,
    }


def _decision_logic_from_evidence(evidence_result, confidence: float) -> dict:
    """Build structured decision-logic summary from evidence distribution."""
    by_source = (evidence_result.agreement_matrix or {}).get("by_source", {})
    source_diversity = len(by_source.keys()) if isinstance(by_source, dict) else 0
    conflict_detected = evidence_result.total_supporting > 0 and evidence_result.total_refuting > 0
    low_volume = len(evidence_result.evidence_items) < 3
    bias_risk = "high" if source_diversity <= 1 else ("medium" if source_diversity == 2 else "low")

    return {
        "supporting_count": evidence_result.total_supporting,
        "refuting_count": evidence_result.total_refuting,
        "neutral_count": evidence_result.total_neutral,
        "source_diversity": source_diversity,
        "conflict_detected": conflict_detected,
        "bias_risk": bias_risk,
        "incomplete_context": low_volume or confidence < 60,
        "evidence_volume": len(evidence_result.evidence_items),
    }


def _apply_safeguard_calibration(
    claim_text: str,
    verdict_label: VerdictLabel,
    confidence: float,
    evidence_result,
) -> tuple[VerdictLabel, float, list[str], dict]:
    """Calibrate confidence using robust guardrails for ambiguity/conflict/bias risk."""
    c = float(confidence)
    safeguards_triggered = []

    agreement = evidence_result.agreement_matrix or {}
    source_diversity = int(agreement.get("source_diversity", 0))
    contradiction_index = float(agreement.get("contradiction_index", 0.0))
    robustness_score = float(agreement.get("robustness_score", 0.0))
    volume = len(evidence_result.evidence_items)

    if source_diversity <= 1:
        c -= 8
        safeguards_triggered.append("single_source_risk")

    if volume < 3:
        c -= 6
        safeguards_triggered.append("low_evidence_volume")

    if contradiction_index >= 0.35:
        c -= 10
        safeguards_triggered.append("high_contradiction")

    lowered = (claim_text or "").lower()
    if any(k in lowered for k in ["always", "never", "all", "only", "every"]):
        c -= 4
        safeguards_triggered.append("absolute_claim_language")

    if robustness_score >= 0.78 and source_diversity >= 2 and contradiction_index < 0.2:
        c += 5
        safeguards_triggered.append("high_robustness_boost")

    c = max(5.0, min(98.0, c))

    # Only force INSUFFICIENT_EVIDENCE if confidence is very low
    if c < 30 and verdict_label != VerdictLabel.INSUFFICIENT_EVIDENCE:
        verdict_label = VerdictLabel.INSUFFICIENT_EVIDENCE
        safeguards_triggered.append("forced_insufficient_evidence")

    diagnostics = {
        "source_diversity": source_diversity,
        "contradiction_index": contradiction_index,
        "robustness_score": robustness_score,
        "safeguards_triggered": safeguards_triggered,
        "controversy_score": round(min(1.0, contradiction_index + (0.2 if "absolute_claim_language" in safeguards_triggered else 0.0)), 3),
    }
    return verdict_label, c, safeguards_triggered, diagnostics

# ─── Classification Prompt ───────────────────────────────────────────────────

VERACITY_CLASSIFICATION_PROMPT = """You are an expert fact-checker. Analyze the following claim against the provided evidence and classify its veracity.

**CLAIM:** {claim}

**EVIDENCE:**
{evidence}

**AGREEMENT MATRIX:**
- Supporting evidence: {supporting} items
- Refuting evidence: {refuting} items  
- Neutral evidence: {neutral} items
- Consensus: {consensus}

**Instructions:**
1. Use chain-of-thought reasoning to analyze each piece of evidence
2. Consider source credibility scores
3. Identify any conflicting evidence
4. Classify the claim using the 5-level veracity scale
5. Provide a calibrated confidence score (0-100%)

**Veracity Scale:**
- SUPPORTED: Strong evidence confirms the claim
- LIKELY_SUPPORTED: Moderate evidence leans toward confirmation
- INSUFFICIENT_EVIDENCE: Not enough data to determine
- LIKELY_REFUTED: Moderate evidence leans toward refutation
- REFUTED: Strong evidence contradicts the claim

**Output Format (strict JSON):**
{{
    "verdict": "SUPPORTED|LIKELY_SUPPORTED|INSUFFICIENT_EVIDENCE|LIKELY_REFUTED|REFUTED",
    "confidence": 85.0,
    "confidence_band": {{"low": 78.0, "high": 90.0, "level": "high|medium|low"}},
    "reasoning_chain": "Step 1: ... Step 2: ... Step 3: ... Therefore: ...",
    "conflicting_evidence": ["Description of conflict 1", "Description of conflict 2"],
    "key_evidence": ["Most important evidence point 1", "Most important evidence point 2"],
    "uncertainty_factors": ["Source disagreement", "Low evidence volume"],
    "ambiguity_notes": ["Claim uses broad wording", "Time period is unspecified"],
    "decision_logic": {{
        "supporting_count": 0,
        "refuting_count": 0,
        "neutral_count": 0,
        "source_diversity": 0,
        "conflict_detected": false,
        "bias_risk": "low|medium|high",
        "incomplete_context": true
    }}
}}

**Important:**
- Never present uncertain results as definitive
- INSUFFICIENT_EVIDENCE is a valid, first-class verdict
- If confidence is below 40%, lean toward INSUFFICIENT_EVIDENCE
- Show uncertainty ranges, not just point estimates
- Flag ALL conflicting evidence explicitly

Return ONLY valid JSON, no additional text or markdown.
"""


async def classify_veracity(state: GraphState) -> GraphState:
    """
    Veracity Classifier Node for LangGraph.
    Classifies each claim on a 5-level scale with CoT reasoning.
    """
    state.status = AnalysisStatus.CLASSIFYING

    state.audit_log.append({
        "node": "veracity_classifier",
        "action": "start",
        "timestamp": datetime.utcnow().isoformat(),
    })

    if not state.extraction_result or not state.evidence_results:
        logger.warning("No claims or evidence to classify")
        return state

    try:
        llm = None
        if not settings.FAST_MODE:
            llm = ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                default_headers={"HTTP-Referer": "https://openrouter.ai/", "X-Title": "VeritasAI"},
            )

        verdict_results = []

        for evidence_result in state.evidence_results:
            claim_text = evidence_result.claim_text

            if settings.FAST_MODE:
                (
                    verdict_label,
                    confidence,
                    reasoning_chain,
                    conflicts,
                    key_evidence,
                    uncertainty_factors,
                    ambiguity_notes,
                    decision_logic,
                ) = _fast_verdict_from_evidence(evidence_result)

                verdict_label, confidence, safeguards, diagnostics = _apply_safeguard_calibration(
                    claim_text,
                    verdict_label,
                    confidence,
                    evidence_result,
                )
                for s in safeguards:
                    if s not in uncertainty_factors:
                        uncertainty_factors.append(s)
                decision_logic.update(diagnostics)

                verdict = VerdictResult(
                    claim_id=evidence_result.claim_id,
                    claim_text=claim_text,
                    verdict=verdict_label,
                    confidence=confidence,
                    reasoning_chain=reasoning_chain,
                    conflicting_evidence=conflicts,
                    key_evidence=key_evidence,
                    confidence_band=_confidence_band(confidence),
                    uncertainty_factors=uncertainty_factors,
                    ambiguity_notes=ambiguity_notes,
                    decision_logic=decision_logic,
                )
                verdict_results.append(verdict)
                logger.info(
                    f"Claim {evidence_result.claim_id}: {verdict_label.value} "
                    f"({confidence:.1f}% confidence, fast mode)"
                )
                continue

            # Format evidence for the prompt
            evidence_text = _format_evidence(evidence_result)
            agreement = evidence_result.agreement_matrix

            prompt = VERACITY_CLASSIFICATION_PROMPT.format(
                claim=claim_text,
                evidence=evidence_text,
                supporting=evidence_result.total_supporting,
                refuting=evidence_result.total_refuting,
                neutral=evidence_result.total_neutral,
                consensus=agreement.get("consensus", "unknown"),
            )

            response = await llm.ainvoke(prompt)
            response_text = strip_thinking_tags(response.content.strip())

            # Clean markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            try:
                parsed = json.loads(response_text.strip())
            except json.JSONDecodeError:
                logger.error(f"Failed to parse verdict JSON: {response_text[:200]}")
                parsed = {
                    "verdict": "INSUFFICIENT_EVIDENCE",
                    "confidence": 30.0,
                    "confidence_band": {"low": 12.0, "high": 48.0, "level": "low"},
                    "reasoning_chain": "Failed to parse LLM response. Defaulting to insufficient evidence.",
                    "conflicting_evidence": [],
                    "key_evidence": [],
                    "uncertainty_factors": ["Model response format error"],
                    "ambiguity_notes": ["Unable to parse structured response"],
                    "decision_logic": {},
                }

            # Map verdict string to enum
            verdict_str = parsed.get("verdict", "INSUFFICIENT_EVIDENCE").upper().replace(" ", "_")
            try:
                verdict_label = VerdictLabel(verdict_str)
            except ValueError:
                verdict_label = VerdictLabel.INSUFFICIENT_EVIDENCE

            confidence = float(parsed.get("confidence", 50.0))
            confidence_band = parsed.get("confidence_band") or _confidence_band(confidence)
            uncertainty_factors = parsed.get("uncertainty_factors", [])
            ambiguity_notes = parsed.get("ambiguity_notes", [])
            decision_logic = parsed.get("decision_logic") or _decision_logic_from_evidence(evidence_result, confidence)

            verdict_label, confidence, safeguards, diagnostics = _apply_safeguard_calibration(
                claim_text,
                verdict_label,
                confidence,
                evidence_result,
            )
            for s in safeguards:
                if s not in uncertainty_factors:
                    uncertainty_factors.append(s)
            decision_logic.update(diagnostics)
            confidence_band = _confidence_band(confidence)

            # Calibration: if very low confidence, override to INSUFFICIENT_EVIDENCE
            if confidence < 30.0 and verdict_label not in (
                VerdictLabel.INSUFFICIENT_EVIDENCE,
            ):
                verdict_label = VerdictLabel.INSUFFICIENT_EVIDENCE

            verdict = VerdictResult(
                claim_id=evidence_result.claim_id,
                claim_text=claim_text,
                verdict=verdict_label,
                confidence=min(100.0, max(0.0, confidence)),
                reasoning_chain=parsed.get("reasoning_chain", ""),
                conflicting_evidence=parsed.get("conflicting_evidence", []),
                key_evidence=parsed.get("key_evidence", []),
                confidence_band=confidence_band,
                uncertainty_factors=uncertainty_factors,
                ambiguity_notes=ambiguity_notes,
                decision_logic=decision_logic,
            )
            verdict_results.append(verdict)

            logger.info(
                f"Claim {evidence_result.claim_id}: {verdict_label.value} "
                f"({confidence:.1f}% confidence)"
            )

        state.verdict_results = verdict_results

        state.audit_log.append({
            "node": "veracity_classifier",
            "action": "complete",
            "verdicts": len(verdict_results),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Veracity classification failed: {e}")
        state.error = f"Veracity classification failed: {str(e)}"
        state.audit_log.append({
            "node": "veracity_classifier",
            "action": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })

    return state


def _format_evidence(evidence_result) -> str:
    """Format evidence items into a readable string for the LLM prompt."""
    lines = []
    for i, item in enumerate(evidence_result.evidence_items[:10], 1):
        source = item.source_type.value.upper()
        cred = f"{item.credibility_score:.0%}"
        stance = item.stance.upper()
        lines.append(
            f"[{i}] SOURCE: {source} | CREDIBILITY: {cred} | STANCE: {stance}\n"
            f"    TITLE: {item.title}\n"
            f"    SNIPPET: {item.snippet[:200]}\n"
            f"    URL: {item.url}\n"
        )
    return "\n".join(lines) if lines else "No evidence available."


def _is_extraordinary_claim(claim_text: str) -> tuple[bool, str]:
    """
    Detect if a claim makes an extraordinary assertion that would be major news
    if true (death, attack, war, disaster, etc.). Such claims, when unsupported
    by ANY evidence, should be treated as fabricated rather than 'insufficient'.
    Returns (is_extraordinary, category).
    """
    claim_lower = claim_text.lower()

    # Category patterns: (category_name, keyword_list)
    extraordinary_patterns = [
        ("death_claim", [
            "killed", "murdered", "assassinated", "died", "dead", "death of",
            "passed away", "shot dead", "executed", "bombing killed",
            "massacre", "genocide", "slaughtered",
        ]),
        ("war_claim", [
            "declared war", "invaded", "attacked", "bombed", "nuked",
            "launched missiles", "war between", "military attack",
            "airstrike", "ground invasion",
        ]),
        ("disaster_claim", [
            "earthquake destroyed", "tsunami hit", "volcano erupted",
            "city destroyed", "collapsed", "massive explosion",
        ]),
        ("arrest_claim", [
            "arrested", "jailed", "imprisoned", "sentenced to",
            "found guilty", "convicted of",
        ]),
        ("resignation_claim", [
            "resigned", "stepped down", "impeached", "overthrown", "coup",
            "removed from office",
        ]),
        ("score_claim", [
            "scored", "won the match", "defeated", "beat", "lost the match",
            "final score", "won the series", "won the cup", "won the tournament",
            "cricket", "football", "soccer", "basketball", "tennis",
        ]),
        ("election_claim", [
            "won the election", "elected as", "became president",
            "became prime minister", "lost the election", "election results",
        ]),
        ("economic_claim", [
            "stock market crashed", "economy collapsed", "currency devalued",
            "bankruptcy", "went bankrupt", "financial crisis",
        ]),
    ]

    for category, keywords in extraordinary_patterns:
        if any(kw in claim_lower for kw in keywords):
            return True, category

    return False, ""


def _check_evidence_contradicts_claim(claim_text: str, evidence_items) -> tuple[bool, float, str]:
    """
    CONSERVATIVE check: only returns implicitly_refuted=True when evidence
    contains strong *positive signals* that directly contradict the claim.

    Example: claim 'Modi killed by Iran' → evidence shows Modi alive, meeting
    world leaders, giving speeches → implicitly refuted.

    This does NOT fire just because evidence discusses the same entities.
    That was the old bug — topic overlap != contradiction.

    Returns (implicitly_refuted, confidence, reasoning).
    """
    claim_lower = claim_text.lower()

    # ── Only check death / assassination / catastrophe claims ─────────
    death_keywords = [
        "killed", "died", "dead", "assassinated", "murdered", "passed away",
        "shot dead", "execution", "executed", "fatal",
    ]
    is_death_claim = any(kw in claim_lower for kw in death_keywords)

    if not is_death_claim:
        # For non-death claims, we do NOT attempt implicit refutation.
        # Topic overlap without explicit stance is NOT a contradiction.
        return False, 0.0, ""

    # ── Extract person names from the claim ───────────────────────────
    # Look for capitalized multi-word names in the ORIGINAL claim text
    import re
    name_pattern = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim_text)
    person_names = [n.lower() for n in name_pattern if len(n) > 2]

    if not person_names:
        return False, 0.0, ""

    # ── Look for "alive signals" — evidence the person is alive ───────
    alive_signals = [
        "met with", "meeting with", "meets with", "spoke at", "speaking at",
        "attended", "attending", "announced", "announces", "addressed",
        "visited", "visiting", "inaugurated", "inaugurates", "launched",
        "signed", "signing", "released", "released statement",
        "tweeted", "posted on", "gave a speech", "is alive",
        "is safe", "is well", "appeared", "arrives", "arrived",
        "chaired", "presided", "led the", "congratulated",
        "participated", "will attend", "will visit", "is scheduled",
        "denied", "denies reports", "dismissed rumors",
    ]

    alive_evidence_count = 0
    alive_details = []

    for item in evidence_items:
        text = f"{item.title} {item.snippet}".lower()

        # Check if evidence mentions any of the claimed persons
        mentions_person = any(name in text for name in person_names)
        if not mentions_person:
            continue

        # Check if evidence contains alive signals for that person
        found_signals = [sig for sig in alive_signals if sig in text]
        if found_signals:
            alive_evidence_count += 1
            alive_details.append(f"'{item.title[:60]}' contains: {', '.join(found_signals[:3])}")

    # ── Decision: require multiple alive signals to refute ────────────
    if alive_evidence_count >= 3:
        return True, 80.0 + min(10.0, alive_evidence_count * 2.0), (
            f"Death/harm claim about {', '.join(person_names)}, but {alive_evidence_count} "
            f"evidence items show the person alive and active. "
            f"Details: {'; '.join(alive_details[:3])}"
        )
    elif alive_evidence_count >= 2:
        return True, 70.0, (
            f"Death/harm claim about {', '.join(person_names)}, but {alive_evidence_count} "
            f"evidence items contain alive signals. {'; '.join(alive_details[:2])}"
        )

    return False, 0.0, ""


def _fast_verdict_from_evidence(evidence_result) -> tuple[VerdictLabel, float, str, list[str], list[str], list[str], list[str], dict]:
    """Low-latency deterministic verdict mapping from evidence counts."""
    s = evidence_result.total_supporting
    r = evidence_result.total_refuting
    n = evidence_result.total_neutral
    total = max(1, s + r + n)

    net = s - r
    non_neutral = s + r
    strength = abs(net) / total if total > 0 else 0

    # ── Authoritative source override ─────────────────────────────────
    # Tavily AI Answer and Google Fact Check are high-authority sources that
    # directly answer the claim query. If they refute/support, they should
    # carry decisive weight over generic web evidence.
    authoritative_refuting = 0
    authoritative_supporting = 0
    for item in evidence_result.evidence_items:
        is_authoritative = (
            item.source_type.value == "google_factcheck"
            or item.source_type.value == "gemini"
            or (item.source_type.value == "tavily" and "AI Answer" in item.title)
        )
        if is_authoritative:
            if item.stance == "refutes":
                authoritative_refuting += 1
            elif item.stance == "supports":
                authoritative_supporting += 1

    # If authoritative sources refute while generic ones support, trust authoritative
    if authoritative_refuting > 0 and authoritative_supporting == 0:
        # Authoritative sources clearly refute — override the simple count
        if authoritative_refuting >= 2:
            verdict = VerdictLabel.REFUTED
            confidence = 85.0 + min(10.0, authoritative_refuting * 3.0)
        else:
            verdict = VerdictLabel.LIKELY_REFUTED
            confidence = 72.0

        # Build early return values
        matrix = evidence_result.agreement_matrix or {}
        avg_cred = float(matrix.get("avg_credibility", 0.5))
        conflicts = [f"Authoritative sources ({authoritative_refuting}) refute but {s} generic items appear to support."]
        key_evidence = [f"Supporting: {s}", f"Refuting: {r}", f"Neutral: {n}",
                        f"Authoritative Refuting: {authoritative_refuting}"]
        reasoning = (
            f"Step 1: Collected {total} evidence items. "
            f"Step 2: {authoritative_refuting} authoritative source(s) (Fact Check / AI Answer) directly refute the claim. "
            f"Step 3: Despite {s} generic items appearing to support, authoritative sources take precedence. "
            f"Therefore: {verdict.value} with {confidence:.0f}% confidence."
        )
        uncertainty_factors = []
        if s > 0:
            uncertainty_factors.append(f"Some generic evidence appears supporting ({s} items) but is outweighed by authoritative sources")
        ambiguity_notes = []
        claim_lower = (evidence_result.claim_text or "").lower()
        if any(k in claim_lower for k in ["always", "never", "all", "only", "every"]):
            ambiguity_notes.append("Absolute wording increases ambiguity risk")
        decision_logic = _decision_logic_from_evidence(evidence_result, confidence)
        decision_logic["authoritative_override"] = True
        return (verdict, max(0.0, min(100.0, confidence)), reasoning, conflicts,
                key_evidence, uncertainty_factors, ambiguity_notes, decision_logic)

    if authoritative_supporting > 0 and authoritative_refuting == 0 and r == 0:
        # Authoritative sources clearly support
        verdict = VerdictLabel.SUPPORTED if authoritative_supporting >= 2 else VerdictLabel.LIKELY_SUPPORTED
        confidence = 82.0 + min(10.0, authoritative_supporting * 3.0)
        matrix = evidence_result.agreement_matrix or {}
        reasoning = (
            f"Step 1: Collected {total} evidence items. "
            f"Step 2: {authoritative_supporting} authoritative source(s) directly confirm the claim. "
            f"Therefore: {verdict.value} with {confidence:.0f}% confidence."
        )
        decision_logic = _decision_logic_from_evidence(evidence_result, confidence)
        decision_logic["authoritative_override"] = True
        return (verdict, max(0.0, min(100.0, confidence)), reasoning, [],
                [f"Supporting: {s}", f"Refuting: {r}"], [], [], decision_logic)

    # ── Standard verdict logic ────────────────────────────────────────
    # The key insight: if we have ANY refuting evidence and ZERO supporting,
    # that's a strong signal for refutation (and vice versa).
    # The old logic required net >= 2 which was far too conservative.

    if s == 0 and r == 0:
        # ── Fabricated / extraordinary claim detection ────────────────
        # If no evidence supports OR refutes, and the claim makes an
        # extraordinary assertion (death, war, major event), it's likely fabricated.
        is_extraordinary, category = _is_extraordinary_claim(evidence_result.claim_text)
        implicitly_refuted, impl_conf, impl_reason = _check_evidence_contradicts_claim(
            evidence_result.claim_text, evidence_result.evidence_items
        )

        if is_extraordinary and implicitly_refuted:
            # Extraordinary claim with alive signals contradicting = REFUTED
            verdict = VerdictLabel.REFUTED
            confidence = impl_conf
        elif is_extraordinary and total >= 3 and n >= 3:
            # Extraordinary claim with decent evidence volume but ZERO corroboration
            # This is suspicious — major events would have corroborating evidence
            verdict = VerdictLabel.LIKELY_REFUTED
            confidence = 65.0
        elif is_extraordinary and total >= 2:
            # Some evidence but no corroboration — lean toward insufficient
            # Don't jump to LIKELY_REFUTED just because it's extraordinary
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
            confidence = 45.0
        elif is_extraordinary and total < 2:
            # Extraordinary claim with barely any evidence
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
            confidence = 38.0
        else:
            # Regular claim with only neutral evidence
            verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
            confidence = 35.0
    elif r > 0 and s == 0:
        # Only refuting, no supporting — clear refutation
        if r >= 3:
            verdict = VerdictLabel.REFUTED
            confidence = 82.0 + min(10.0, r * 2.0)
        elif r >= 2:
            verdict = VerdictLabel.REFUTED
            confidence = 72.0
        else:
            verdict = VerdictLabel.LIKELY_REFUTED
            confidence = 62.0
    elif s > 0 and r == 0:
        # Only supporting, no refuting — clear support
        if s >= 3:
            verdict = VerdictLabel.SUPPORTED
            confidence = 82.0 + min(10.0, s * 2.0)
        elif s >= 2:
            verdict = VerdictLabel.SUPPORTED
            confidence = 72.0
        else:
            verdict = VerdictLabel.LIKELY_SUPPORTED
            confidence = 62.0
    elif net >= 2:
        # Strong net support despite some refutation
        verdict = VerdictLabel.LIKELY_SUPPORTED
        confidence = 60.0 + min(25.0, strength * 30.0)
    elif net <= -2:
        # Strong net refutation despite some support
        verdict = VerdictLabel.LIKELY_REFUTED
        confidence = 60.0 + min(25.0, strength * 30.0)
    elif abs(net) == 1:
        # Slight lean one way — use LIKELY_ with moderate confidence
        if net > 0:
            verdict = VerdictLabel.LIKELY_SUPPORTED
            confidence = 52.0
        else:
            verdict = VerdictLabel.LIKELY_REFUTED
            confidence = 52.0
    else:
        # Equal supporting and refuting — genuine conflict
        verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        confidence = 40.0

    # ── Boost confidence if evidence is high quality ──────────────────
    matrix = evidence_result.agreement_matrix or {}
    avg_cred = float(matrix.get("avg_credibility", 0.5))
    if avg_cred >= 0.8 and non_neutral > 0:
        confidence += 5.0
    elif avg_cred < 0.4 and non_neutral > 0:
        confidence -= 5.0

    conflicts = []
    if s > 0 and r > 0:
        conflicts.append(
            f"Conflicting evidence detected: {s} supporting vs {r} refuting items."
        )

    uncertainty_factors = []
    if total < 3:
        uncertainty_factors.append("Low evidence volume")
    if s > 0 and r > 0:
        uncertainty_factors.append("Conflicting evidence across sources")
    if n > 0 and n >= max(s, r) and non_neutral > 0:
        uncertainty_factors.append("Significant neutral evidence alongside decisive evidence")
    elif n > 0 and non_neutral == 0:
        uncertainty_factors.append("Only neutral evidence found — no clear stance detected")

    ambiguity_notes = []
    claim_lower = (evidence_result.claim_text or "").lower()
    if any(k in claim_lower for k in ["always", "never", "all", "only", "every"]):
        ambiguity_notes.append("Absolute wording increases ambiguity risk")
    if not any(ch.isdigit() for ch in claim_lower):
        ambiguity_notes.append("Claim lacks specific quantitative context")

    key_evidence = [
        f"Supporting: {s}",
        f"Refuting: {r}",
        f"Neutral: {n}",
    ]

    reasoning = (
        f"Step 1: Collected {total} evidence items from {int(matrix.get('source_diversity', 0))} sources. "
        f"Step 2: Classified stances — {s} supporting, {r} refuting, {n} neutral. "
        f"Step 3: Net direction is {'supporting' if net > 0 else 'refuting' if net < 0 else 'balanced'} "
        f"(net={net}, strength={strength:.2f}). "
        f"Step 4: Average source credibility is {avg_cred:.2f}. "
        f"Therefore: {verdict.value} with {confidence:.0f}% confidence."
    )

    decision_logic = _decision_logic_from_evidence(evidence_result, confidence)
    return (
        verdict,
        max(0.0, min(100.0, confidence)),
        reasoning,
        conflicts,
        key_evidence,
        uncertainty_factors,
        ambiguity_notes,
        decision_logic,
    )

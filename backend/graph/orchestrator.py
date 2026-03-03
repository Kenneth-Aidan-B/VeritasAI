"""
VeritasAI — LangGraph Orchestrator
Defines the stateful graph workflow that chains all agent nodes together.
Pipeline: Claim Extractor → Evidence Retriever → Veracity Classifier → Debate → Explanation → Correction
"""

import logging
from datetime import datetime
from typing import Optional, Callable, Any

from langgraph.graph import StateGraph, END

from backend.models.schemas import (
    GraphState,
    FullAnalysisResult,
    ClaimAnalysis,
    AnalysisStatus,
)
from backend.config import settings
from backend.agents.claim_extractor import extract_claims
from backend.agents.evidence_retriever import retrieve_evidence
from backend.agents.veracity_classifier import classify_veracity
from backend.agents.explanation_generator import generate_explanations
from backend.agents.correction_generator import generate_corrections
from backend.agents.debate_agents import run_debate
from backend.services.cache_service import cache_result

logger = logging.getLogger(__name__)


# ─── Wrapper nodes that call async agents ────────────────────────────────────

async def node_extract_claims(state: dict) -> dict:
    """LangGraph node wrapper for claim extraction."""
    gs = GraphState(**state)
    gs = await extract_claims(gs)
    return gs.model_dump()


async def node_retrieve_evidence(state: dict) -> dict:
    """LangGraph node wrapper for evidence retrieval."""
    gs = GraphState(**state)
    gs = await retrieve_evidence(gs)
    return gs.model_dump()


async def node_classify_veracity(state: dict) -> dict:
    """LangGraph node wrapper for veracity classification."""
    gs = GraphState(**state)
    gs = await classify_veracity(gs)
    return gs.model_dump()


async def node_run_debate(state: dict) -> dict:
    """LangGraph node wrapper for multi-agent debate."""
    gs = GraphState(**state)
    gs = await run_debate(gs)
    return gs.model_dump()


async def node_generate_explanations(state: dict) -> dict:
    """LangGraph node wrapper for explanation generation."""
    gs = GraphState(**state)
    gs = await generate_explanations(gs)
    return gs.model_dump()


async def node_generate_corrections(state: dict) -> dict:
    """LangGraph node wrapper for correction generation."""
    gs = GraphState(**state)
    gs = await generate_corrections(gs)
    return gs.model_dump()


async def node_finalize(state: dict) -> dict:
    """Finalize the analysis: cache results, set completed status."""
    gs = GraphState(**state)
    gs.status = AnalysisStatus.COMPLETED

    # Cache each claim result
    for verdict in gs.verdict_results:
        try:
            await cache_result(verdict.claim_text, verdict.model_dump())
        except Exception as e:
            logger.warning(f"Failed to cache claim {verdict.claim_id}: {e}")

    gs.audit_log.append({
        "node": "finalize",
        "action": "complete",
        "timestamp": datetime.utcnow().isoformat(),
    })

    return gs.model_dump()


# ─── Conditional edges ──────────────────────────────────────────────────────

def should_continue_after_extraction(state: dict) -> str:
    """Check if claim extraction succeeded and has claims to process."""
    gs = GraphState(**state)
    if gs.error:
        return "error"
    if not gs.extraction_result or not gs.extraction_result.claims:
        return "no_claims"
    return "continue"


def should_debate(state: dict) -> str:
    """Check if multi-agent debate should be run."""
    gs = GraphState(**state)
    if gs.error:
        return "skip_debate"
    if gs.enable_debate:
        return "debate"
    return "skip_debate"


# ─── Build the Graph ─────────────────────────────────────────────────────────

def build_verification_graph() -> StateGraph:
    """
    Build the LangGraph verification pipeline.
    
    Pipeline:
    START → extract_claims → retrieve_evidence → classify_veracity
        → [conditional: debate] → generate_explanations → generate_corrections → finalize → END
    """
    # Define the graph with dict state (LangGraph requirement)
    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("extract_claims", node_extract_claims)
    workflow.add_node("retrieve_evidence", node_retrieve_evidence)
    workflow.add_node("classify_veracity", node_classify_veracity)
    workflow.add_node("run_debate", node_run_debate)
    workflow.add_node("generate_explanations", node_generate_explanations)
    workflow.add_node("generate_corrections", node_generate_corrections)
    workflow.add_node("finalize", node_finalize)

    # Set entry point
    workflow.set_entry_point("extract_claims")

    # Add edges
    workflow.add_conditional_edges(
        "extract_claims",
        should_continue_after_extraction,
        {
            "continue": "retrieve_evidence",
            "no_claims": "finalize",
            "error": "finalize",
        },
    )

    workflow.add_edge("retrieve_evidence", "classify_veracity")

    workflow.add_conditional_edges(
        "classify_veracity",
        should_debate,
        {
            "debate": "run_debate",
            "skip_debate": "generate_explanations",
        },
    )

    workflow.add_edge("run_debate", "generate_explanations")
    workflow.add_edge("generate_explanations", "generate_corrections")
    workflow.add_edge("generate_corrections", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


# ─── Compiled graph singleton ────────────────────────────────────────────────

_compiled_graph = None


def get_compiled_graph():
    """Get or create the compiled LangGraph."""
    global _compiled_graph
    if _compiled_graph is None:
        workflow = build_verification_graph()
        _compiled_graph = workflow.compile()
        logger.info("LangGraph verification pipeline compiled successfully")
    return _compiled_graph


# ─── Run Analysis ───────────────────────────────────────────────────────────

async def run_analysis(
    text: str,
    session_id: str = "",
    enable_debate: bool = True,
    progress_callback: Optional[Callable] = None,
) -> FullAnalysisResult:
    """
    Run the full verification pipeline on input text.
    
    Args:
        text: The text to analyze
        session_id: WebSocket session ID for progress tracking
        enable_debate: Whether to enable multi-agent debate
        progress_callback: Optional async callback for progress updates
        
    Returns:
        FullAnalysisResult with complete analysis
    """
    graph = get_compiled_graph()

    # Initialize state
    initial_state = GraphState(
        original_text=text,
        session_id=session_id,
        enable_debate=enable_debate,
        status=AnalysisStatus.PENDING,
        audit_log=[{
            "node": "orchestrator",
            "action": "start",
            "text_length": len(text),
            "timestamp": datetime.utcnow().isoformat(),
        }],
    )

    try:
        # Run the graph
        final_state_dict = await graph.ainvoke(initial_state.model_dump())
        final_state = GraphState(**final_state_dict)

        # Build the full analysis result
        result = _build_full_result(final_state)

        if progress_callback:
            await progress_callback({
                "type": "complete",
                "session_id": session_id,
                "result": result.model_dump(),
            })

        return result

    except Exception as e:
        logger.error(f"Analysis pipeline failed: {e}")
        return FullAnalysisResult(
            session_id=session_id,
            original_text=text,
            status=AnalysisStatus.ERROR,
            error=str(e),
            responsible_ai_card=_build_responsible_ai_card(),
        )


def _build_full_result(state: GraphState) -> FullAnalysisResult:
    """Build FullAnalysisResult from final graph state."""
    claim_analyses = []

    if state.extraction_result:
        for claim in state.extraction_result.claims:
            # Find matching results for this claim
            evidence = next(
                (e for e in state.evidence_results if e.claim_id == claim.id), None
            )
            verdict = next(
                (v for v in state.verdict_results if v.claim_id == claim.id), None
            )
            debate = next(
                (d for d in state.debate_results if d.claim_id == claim.id), None
            )
            explanation = next(
                (e for e in state.explanation_results if e.claim_id == claim.id), None
            )
            correction = next(
                (c for c in state.correction_results if c.claim_id == claim.id), None
            )

            claim_analysis = ClaimAnalysis(
                claim=claim,
                evidence=evidence,
                verdict=verdict,
                debate=debate,
                explanation=explanation,
                correction=correction,
            )
            claim_analyses.append(claim_analysis)

    return FullAnalysisResult(
        session_id=state.session_id,
        original_text=state.original_text,
        status=state.status,
        claims=claim_analyses,
        responsible_ai_card=_build_responsible_ai_card(),
        system_diagnostics=_build_system_diagnostics(state),
        audit_log=state.audit_log,
        completed_at=datetime.utcnow().isoformat(),
        error=state.error,
    )


def _build_system_diagnostics(state: GraphState) -> dict:
    """Build runtime diagnostics to support transparency and operational robustness."""
    node_counts = {}
    for entry in state.audit_log:
        node = entry.get("node", "unknown")
        node_counts[node] = node_counts.get(node, 0) + 1

    conflict_claims = 0
    low_context_claims = 0
    total_claims = len(state.verdict_results)
    for v in state.verdict_results:
        logic = getattr(v, "decision_logic", {}) or {}
        if logic.get("conflict_detected"):
            conflict_claims += 1
        if logic.get("incomplete_context"):
            low_context_claims += 1

    return {
        "pipeline_mode": "fast" if settings.FAST_MODE else ("debate" if state.enable_debate else "standard"),
        "claims_processed": total_claims,
        "conflict_claim_ratio": round((conflict_claims / total_claims), 3) if total_claims else 0.0,
        "incomplete_context_ratio": round((low_context_claims / total_claims), 3) if total_claims else 0.0,
        "node_activity": node_counts,
        "resilience_features": [
            "bounded external timeouts",
            "uncertainty-aware confidence calibration",
            "conflict and ambiguity surfacing",
            "structured decision-logic trace",
        ],
    }


def _build_responsible_ai_card() -> dict:
    """Build the Responsible AI card with assumptions, limitations, and bias acknowledgments."""
    return {
        "assumptions": [
            "Relying on web sources available as of analysis date",
            "English-language content only",
            "Claims evaluated independently; cross-claim context may be limited",
            "Source credibility scores are estimated based on domain reputation",
            "LLM reasoning may contain biases from training data",
        ],
        "limitations": [
            "Satire/sarcasm detection is limited",
            "Image/video verification is not supported (text-only)",
            "Non-English content analysis is not supported",
            "Real-time social media ingestion at scale is not implemented",
            "Very recent events may not have sufficient web coverage",
            "Paywalled content cannot be accessed",
        ],
        "confidence_calibration": [
            "Uncertain results are never presented as definitive",
            "INSUFFICIENT_EVIDENCE is a first-class verdict, not a failure",
            "Confidence scores reflect the strength of available evidence",
            "Multi-agent debate is used to stress-test uncertain verdicts",
        ],
        "bias_acknowledgment": [
            "Evidence sources may carry political/ideological bias",
            "The LLM has potential biases from training data",
            "Web search results may be influenced by algorithmic ranking",
            "Source credibility scoring is based on general domain reputation",
        ],
        "transparency": [
            "Full audit log of every API call and reasoning step is available",
            "Every verdict includes a transparent reasoning chain",
            "Evidence citations are provided with source URLs",
            "The multi-agent debate feature provides multiple perspectives",
        ],
        "scalability": [
            "Pipeline is node-based and horizontally scalable at service boundaries",
            "External retrieval is parallelized with bounded timeouts",
            "Fast mode provides lower-latency, lower-cost inference for high throughput",
        ],
        "controversial_content_handling": [
            "Conflicting information is explicitly surfaced, not hidden",
            "Ambiguous or incomplete-context claims are marked with uncertainty notes",
            "Low-confidence outcomes are routed toward INSUFFICIENT_EVIDENCE",
        ],
    }

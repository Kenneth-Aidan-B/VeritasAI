"""
VeritasAI Pydantic Models / Schemas
Defines all data structures used across the pipeline.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


# ─── Enums ───────────────────────────────────────────────────────────────────

class VerdictLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    LIKELY_SUPPORTED = "LIKELY_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LIKELY_REFUTED = "LIKELY_REFUTED"
    REFUTED = "REFUTED"


class SourceType(str, Enum):
    TAVILY = "tavily"
    GOOGLE_FACTCHECK = "google_factcheck"
    WIKIPEDIA = "wikipedia"
    SERPER = "serper"
    GEMINI = "gemini"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING_CLAIMS = "extracting_claims"
    RETRIEVING_EVIDENCE = "retrieving_evidence"
    CLASSIFYING = "classifying"
    DEBATING = "debating"
    GENERATING_EXPLANATION = "generating_explanation"
    GENERATING_CORRECTION = "generating_correction"
    COMPLETED = "completed"
    ERROR = "error"


# ─── Input Models ────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Input request for fact-checking analysis."""
    text: str = Field(..., description="Text or URL to analyze")
    enable_debate: bool = Field(default=True, description="Enable multi-agent debate")
    session_id: Optional[str] = Field(default=None, description="Session ID for WebSocket tracking")


# ─── Claim Models ────────────────────────────────────────────────────────────

class AtomicClaim(BaseModel):
    """A single atomic, testable claim extracted from the input."""
    id: int = Field(..., description="Claim index")
    original_sentence: str = Field(..., description="Original sentence the claim was extracted from")
    atomic_claim: str = Field(..., description="Decomposed atomic claim with pronouns resolved")
    check_worthiness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Score indicating how check-worthy this claim is (0-1)"
    )
    entities: list[str] = Field(default_factory=list, description="Named entities in the claim")


class ClaimExtractionResult(BaseModel):
    """Output of the Claim Extractor node."""
    claims: list[AtomicClaim] = Field(default_factory=list)
    total_sentences: int = 0
    total_claims: int = 0


# ─── Evidence Models ─────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """A single piece of evidence from any source."""
    source_type: SourceType
    title: str = ""
    url: str = ""
    snippet: str = ""
    full_text: str = ""
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    stance: str = Field(default="neutral", description="supports, refutes, or neutral")
    retrieved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EvidenceResult(BaseModel):
    """Evidence gathered for a single claim."""
    claim_id: int
    claim_text: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    agreement_matrix: dict = Field(default_factory=dict)
    total_supporting: int = 0
    total_refuting: int = 0
    total_neutral: int = 0


# ─── Verdict Models ──────────────────────────────────────────────────────────

class VerdictResult(BaseModel):
    """Veracity classification result for a single claim."""
    claim_id: int
    claim_text: str
    verdict: VerdictLabel
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage")
    reasoning_chain: str = ""
    conflicting_evidence: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    confidence_band: dict = Field(default_factory=dict, description="Calibrated low/high confidence band")
    uncertainty_factors: list[str] = Field(default_factory=list, description="Key uncertainty drivers")
    ambiguity_notes: list[str] = Field(default_factory=list, description="Detected ambiguity and incomplete-context notes")
    decision_logic: dict = Field(default_factory=dict, description="Structured decision logic and weighting summary")


# ─── Debate Models ───────────────────────────────────────────────────────────

class DebateArgument(BaseModel):
    """A single argument in the multi-agent debate."""
    agent_role: str = Field(..., description="'advocate' or 'skeptic'")
    argument: str = ""
    evidence_cited: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class DebateResult(BaseModel):
    """Result of the multi-agent debate for a claim."""
    claim_id: int
    claim_text: str
    advocate_arguments: list[DebateArgument] = Field(default_factory=list)
    skeptic_arguments: list[DebateArgument] = Field(default_factory=list)
    judge_verdict: VerdictLabel = VerdictLabel.INSUFFICIENT_EVIDENCE
    judge_reasoning: str = ""
    judge_confidence: float = 50.0
    debate_enhanced_verdict: bool = False


# ─── Explanation & Correction Models ─────────────────────────────────────────

class ExplanationResult(BaseModel):
    """Transparent reasoning chain for a claim."""
    claim_id: int
    claim_text: str
    step_by_step_reasoning: list[str] = Field(default_factory=list)
    evidence_citations: list[dict] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_checks: list[str] = Field(default_factory=list)


class CorrectionResult(BaseModel):
    """Factual correction and bias analysis for a claim."""
    claim_id: int
    original_text: str
    corrected_text: str = ""
    corrections_made: list[dict] = Field(default_factory=list)
    bias_analysis: dict = Field(default_factory=dict)
    source_links: list[str] = Field(default_factory=list)
    clarification_notes: list[str] = Field(default_factory=list)


# ─── Full Analysis Result ────────────────────────────────────────────────────

class ClaimAnalysis(BaseModel):
    """Complete analysis result for a single claim."""
    claim: AtomicClaim
    evidence: Optional[EvidenceResult] = None
    verdict: Optional[VerdictResult] = None
    debate: Optional[DebateResult] = None
    explanation: Optional[ExplanationResult] = None
    correction: Optional[CorrectionResult] = None


class FullAnalysisResult(BaseModel):
    """Complete analysis result for the entire input."""
    session_id: str = ""
    original_text: str = ""
    status: AnalysisStatus = AnalysisStatus.PENDING
    claims: list[ClaimAnalysis] = Field(default_factory=list)
    responsible_ai_card: dict = Field(default_factory=dict)
    system_diagnostics: dict = Field(default_factory=dict)
    audit_log: list[dict] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None


# ─── Graph State ─────────────────────────────────────────────────────────────

class GraphState(BaseModel):
    """State object passed through the LangGraph pipeline."""
    original_text: str = ""
    session_id: str = ""
    enable_debate: bool = True
    status: AnalysisStatus = AnalysisStatus.PENDING
    extraction_result: Optional[ClaimExtractionResult] = None
    evidence_results: list[EvidenceResult] = Field(default_factory=list)
    verdict_results: list[VerdictResult] = Field(default_factory=list)
    debate_results: list[DebateResult] = Field(default_factory=list)
    explanation_results: list[ExplanationResult] = Field(default_factory=list)
    correction_results: list[CorrectionResult] = Field(default_factory=list)
    audit_log: list[dict] = Field(default_factory=list)
    error: Optional[str] = None
    progress_callback: Optional[object] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


# ─── WebSocket Messages ─────────────────────────────────────────────────────

class WSMessage(BaseModel):
    """WebSocket message format."""
    type: str = "status_update"
    session_id: str = ""
    status: AnalysisStatus = AnalysisStatus.PENDING
    data: dict = Field(default_factory=dict)
    message: str = ""
    progress: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

"""
models.py — Pydantic request/response schemas.

This is the single source of truth for the API contract.
Written for Pydantic v1 (which installs cleanly on Python 3.15).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OptimizationLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    STYLE = "style"


class IssueCategory(str, Enum):
    STYLE = "style"
    COMPLEXITY = "complexity"
    SECURITY = "security"
    SMELL = "smell"
    SYNTAX = "syntax"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=100_000,
                      description="Python source code to review")

    @field_validator("code")
    @classmethod
    def strip_code(cls, v: str) -> str:
        return v.strip()


class RefactorRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=100_000,
                      description="Python source code to refactor")
    level: OptimizationLevel = Field(OptimizationLevel.MEDIUM,
                                     description="Refactoring aggressiveness")
    use_ai: bool = Field(True, description="Whether to call the AI enhancement layer")

    @field_validator("code")
    @classmethod
    def strip_code(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Sub-models used inside responses
# ---------------------------------------------------------------------------

class Issue(BaseModel):
    line: Optional[int] = Field(None, description="1-based line number, if known")
    column: Optional[int] = Field(None, description="1-based column number, if known")
    severity: IssueSeverity
    category: IssueCategory
    rule: str = Field(..., description="Machine-readable rule identifier, e.g. 'W0612'")
    message: str = Field(..., description="Human-readable explanation")


class QualityBreakdown(BaseModel):
    style: int = Field(..., ge=0, le=100)
    complexity: int = Field(..., ge=0, le=100)
    security: int = Field(..., ge=0, le=100)
    maintainability: int = Field(..., ge=0, le=100)
    big_o: Optional[int] = Field(
        None, ge=0, le=100,
        description="Score derived from the predicted Big-O class, scaled by "
                     "prediction confidence. None if the complexity model is "
                     "unavailable or gave no prediction — the overall score "
                     "then falls back to the other three categories.",
    )


class QualityScore(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Overall quality score 0–100")
    grade: str = Field(..., description="Letter grade A–F")
    breakdown: QualityBreakdown


class AppliedRule(BaseModel):
    rule: str = Field(..., description="Rule identifier")
    line: Optional[int] = None
    description: str = Field(..., description="Plain-English explanation of the change")
    applied: bool = Field(
        True,
        description=(
            "True if the code was actually modified. False marks advice that was "
            "reported but not auto-applied (e.g. range(len(x)) -> enumerate), so the "
            "summary does not claim changes that were never made."
        ),
    )


class AISuggestion(BaseModel):
    explanation: str
    diff: str = Field(..., description="Unified diff of the suggestion")
    validated: bool = Field(..., description="True if ast.parse passed on the suggested code")
    verified: Optional[bool] = Field(
        None,
        description="True if differential testing (app/verify/) confirmed the suggestion "
                     "behaves the same as the original on generated inputs; False if a "
                     "genuine behavioural disagreement was found. None if verification "
                     "wasn't attempted (ambiguous which function to test, or verification "
                     "itself failed to run) — None is NOT evidence of correctness.",
    )
    verification_note: Optional[str] = Field(
        None,
        description="Why verification wasn't attempted, or what disagreed, when verified is not True.",
    )


class SpeedSample(BaseModel):
    """One measured data point: both versions run on the same synthetic
    input of the given size, timed inside a sandboxed subprocess (see
    app/verify/differential.py:measure_speedup) so interpreter startup
    noise doesn't swamp real differences. Milliseconds, not seconds — the
    measured gaps at realistic sizes are often sub-millisecond."""
    size: int = Field(..., description="Synthetic input size used for this measurement")
    original_ms: float = Field(..., description="Original code's measured time, in milliseconds")
    refactored_ms: float = Field(..., description="Refactored code's measured time, in milliseconds")


class PerformanceComparison(BaseModel):
    measured: bool = Field(
        ..., description="True if a real timing comparison was made. False means it wasn't "
                          "attempted or didn't complete — see note — NOT that the two versions "
                          "run at the same speed.",
    )
    samples: List[SpeedSample] = Field(
        default_factory=list,
        description="Empty when measured is False. Multiple sizes, not one run, so a constant-"
                     "factor speedup can be told apart from an actual complexity-class change.",
    )
    speedup: Optional[float] = Field(
        None,
        description="original_ms / refactored_ms at the largest measured size — the single "
                     "headline number, e.g. 4.2 means ~4.2x faster. None when measured is False.",
    )
    complexity_class_likely_changed: bool = Field(
        False,
        description="True if the speed gap widens with input size rather than staying roughly "
                     "constant, i.e. more than a constant-factor cleanup happened.",
    )
    note: str = Field(
        "",
        description="Why measurement wasn't attempted or didn't complete, when measured is "
                     "False (e.g. no scalable parameter, code didn't verify as equivalent). "
                     "Empty when measured is True.",
    )


class FunctionComplexity(BaseModel):
    """Big-O for one function, analysed on its own."""
    name: str = Field(..., description="Function name")
    line: Optional[int] = Field(None, description="Line where the function is defined")
    complexity: str = Field(..., description="Predicted Big-O for this function alone")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: List[str] = Field(default_factory=list)
    suggestion: Optional[str] = None
    is_dominant: bool = Field(
        False,
        description="True for the function that determines the module's overall complexity",
    )


class ComplexityPrediction(BaseModel):
    before: str = Field(..., description="Predicted Big-O before refactor, e.g. 'O(n²)'")
    after: str = Field(..., description="Predicted Big-O after refactor")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: List[str] = Field(
        default_factory=list,
        description="Plain-language structural evidence behind the prediction",
    )
    suggestion: Optional[str] = Field(
        None,
        description="The single highest-value change that would lower the complexity",
    )
    functions: List[FunctionComplexity] = Field(
        default_factory=list,
        description=(
            "Per-function breakdown. Analysing a whole file as one unit blends "
            "structures that belong to different functions and produces a "
            "low-confidence average, so each function is predicted separately "
            "and the worst one drives the overall figure."
        ),
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ReviewResponse(BaseModel):
    valid: bool = Field(..., description="False if the code has syntax errors")
    syntax_error: Optional[str] = Field(None, description="Syntax error if valid=false")
    issues: List[Issue] = Field(default_factory=list)
    quality_score: Optional[QualityScore] = None
    summary: str = Field(..., description="Plain-English one-line summary")


class RefactorResponse(BaseModel):
    original_code: str
    refactored_code: str
    diff: str = Field(..., description="Unified diff between original and refactored code")
    applied_rules: List[AppliedRule] = Field(default_factory=list)
    ai_suggestions: List[AISuggestion] = Field(default_factory=list)
    ai_available: bool = Field(..., description="False if AI layer was unreachable or disabled")
    correctness_concerns: List[str] = Field(
        default_factory=list,
        description="Suspected logic/correctness issues the AI noticed, distinct from "
                     "refactoring suggestions. Only an LLM given the code can plausibly "
                     "reason about intent vs. implementation — the deterministic engines "
                     "(rules, ML complexity model) never attempt this. Always empty when "
                     "ai_available is False.",
    )
    complexity: Optional[ComplexityPrediction] = None
    performance: Optional[PerformanceComparison] = Field(
        None,
        description="Measured (not predicted) runtime comparison between original_code and "
                     "refactored_code, distinct from the ML-predicted 'complexity' field above "
                     "— that's a Big-O class guess, this is an actual sandboxed stopwatch. None "
                     "when refactored_code is identical to original_code (nothing to compare) "
                     "or the syntax gate rejected the input.",
    )
    summary: str


class HealthResponse(BaseModel):
    status: str = "ok"
    ai_configured: bool
    version: str = "1.0.0"

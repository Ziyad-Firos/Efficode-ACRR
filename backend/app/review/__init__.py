"""
review/__init__.py — Review engine coordinator.

run_all_checks() is the single public function called by main.py.
It runs style, complexity, security, and smell checks concurrently
using asyncio.gather with per-check timeouts, then assembles the
full ReviewResponse including the quality score.

Pipeline:
  style.py    → flake8  (subprocess, 10s timeout)
  complexity.py → radon  (in-process)
  security.py → bandit  (subprocess, 15s timeout)
  smells.py   → custom AST visitors (in-process)
  → score assembly
  → ReviewResponse
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

from app.models import (
    Issue,
    IssueCategory,
    IssueSeverity,
    QualityBreakdown,
    QualityScore,
    ReviewResponse,
)
from app.review.style import analyze_style
from app.review.complexity import analyze_complexity, get_maintainability_index
from app.review.security import analyze_security
from app.review.smells import analyze_smells
from app.ml.complexity_predictor import predict_dominant_label

logger = logging.getLogger("acrr.review")

# Per-check timeout in seconds (belt-and-suspenders on top of subprocess timeouts)
_CHECK_TIMEOUT = 20


async def _run_in_executor(fn, *args):
    """Run a synchronous check function in a thread pool without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def run_all_checks(code: str) -> ReviewResponse:
    """
    Run all four review checks concurrently and return a unified ReviewResponse.

    Each check is isolated: if one raises or times out, the others
    still complete and the response is still returned.

    Args:
        code: Python source string (already syntax-validated by main.py).

    Returns:
        ReviewResponse with issues, quality score, and summary.
    """
    # Run all checks concurrently
    results = await asyncio.gather(
        _safe_check("style",      _run_in_executor, analyze_style,      code),
        _safe_check("complexity", _run_in_executor, analyze_complexity,  code),
        _safe_check("security",   _run_in_executor, analyze_security,    code),
        _safe_check("smells",     _run_in_executor, analyze_smells,      code),
        return_exceptions=False,
    )

    style_issues, complexity_issues, security_issues, smell_issues = results

    all_issues: List[Issue] = (
        style_issues + complexity_issues + security_issues + smell_issues
    )

    # Sort by line number (None lines go last)
    all_issues.sort(key=lambda i: (i.line is None, i.line or 0))

    # Quality score
    mi = get_maintainability_index(code)
    big_o = await _safe_predict(code)
    quality_score = _compute_quality_score(
        all_issues, mi, big_o
    )

    # Human-readable summary
    summary = _build_summary(all_issues, quality_score)

    logger.info(
        "Review complete: %d total issues (style=%d, complexity=%d, security=%d, smells=%d) score=%d",
        len(all_issues),
        len(style_issues),
        len(complexity_issues),
        len(security_issues),
        len(smell_issues),
        quality_score.score,
    )

    return ReviewResponse(
        valid=True,
        issues=all_issues,
        quality_score=quality_score,
        summary=summary,
    )


async def _safe_check(name: str, runner, fn, *args) -> List[Issue]:
    """
    Run a single check with a timeout. Returns empty list on any failure.
    Logs the failure but never propagates the exception.
    """
    try:
        return await asyncio.wait_for(runner(fn, *args), timeout=_CHECK_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("%s check timed out after %ss", name, _CHECK_TIMEOUT)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s check failed: %s", name, exc)
        return []


async def _safe_predict(code: str) -> Optional[Tuple[str, float]]:
    """predict_label(), isolated the same way the four checks above are —
    None (not an exception) on timeout, missing model, or any failure."""
    try:
        return await asyncio.wait_for(
            _run_in_executor(predict_dominant_label, code), timeout=_CHECK_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.warning("complexity prediction timed out after %ss", _CHECK_TIMEOUT)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("complexity prediction failed: %s", exc)
        return None


# ── Quality score assembly ─────────────────────────────────────────────────

# Base score (before confidence scaling) for each predicted complexity
# class. O(n) and below are the normal case for real code and are not
# penalised; O(n log n) is the standard cost of a sort-based algorithm and
# is barely penalised. O(n²) and up are where the review previously had
# nothing to say -- the ML complexity engine could call a nested-loop
# function "O(n^3+) (78% conf.)" in the same response that graded it A,
# because the grade never looked at the prediction at all.
_BIG_O_BASE_SCORE = {
    "O(1)": 100,
    "O(log n)": 100,
    "O(n)": 100,
    "O(n log n)": 95,
    "O(n²)": 75,
    "O(n³+)": 45,
    "O(2^n)": 15,
}


def _big_o_score(prediction: Optional[Tuple[str, float]]) -> Optional[int]:
    """
    Confidence-scaled score for the predicted complexity class, or None if
    there is no prediction to score.

    A low-confidence guess should not swing the grade as hard as a
    high-confidence one -- the model's own calibration work (see
    app/ml/audit.py) showed it is honestly under-confident, so a 46%
    confidence O(n³+) guess (main() in the New Year Chaos test case) is
    blended much closer to neutral than a 99% one.
    """
    if prediction is None:
        return None
    label, confidence = prediction
    base = _BIG_O_BASE_SCORE.get(label)
    if base is None:
        return None
    return int(100 - confidence * (100 - base))


def _compute_quality_score(
    issues: List[Issue],
    maintainability_index: float,
    big_o_prediction: Optional[Tuple[str, float]] = None,
) -> QualityScore:
    """
    Derive a 0–100 quality score from issue counts and maintainability index.

    Scoring model:
      - Start at 100
      - Deduct per issue by category × severity weight
      - Blend with radon's maintainability index and the predicted Big-O
        class for the final score

    Deduction table (points per issue):
      ERROR   WARNING   INFO/STYLE
      style      8         4         1
      complexity 10        5         2
      security   15        8         3
      smell       4        3         1
    """
    deductions = {
        (IssueCategory.STYLE,      IssueSeverity.ERROR):   10,
        (IssueCategory.STYLE,      IssueSeverity.WARNING):  5,
        (IssueCategory.STYLE,      IssueSeverity.INFO):     2,
        (IssueCategory.STYLE,      IssueSeverity.STYLE):    2,
        (IssueCategory.COMPLEXITY, IssueSeverity.ERROR):   20,
        (IssueCategory.COMPLEXITY, IssueSeverity.WARNING): 12,
        (IssueCategory.COMPLEXITY, IssueSeverity.INFO):     4,
        (IssueCategory.SECURITY,   IssueSeverity.ERROR):   35,
        (IssueCategory.SECURITY,   IssueSeverity.WARNING): 20,
        (IssueCategory.SECURITY,   IssueSeverity.INFO):    12,
        (IssueCategory.SMELL,      IssueSeverity.ERROR):   10,
        (IssueCategory.SMELL,      IssueSeverity.WARNING):  6,
        (IssueCategory.SMELL,      IssueSeverity.INFO):     2,
    }

    # Per-category scores
    cat_totals: dict[IssueCategory, int] = {cat: 100 for cat in IssueCategory}

    for issue in issues:
        key = (issue.category, issue.severity)
        pts = deductions.get(key, 1)
        cat_totals[issue.category] = max(0, cat_totals[issue.category] - pts)

    style_score      = cat_totals[IssueCategory.STYLE]
    complexity_score = cat_totals[IssueCategory.COMPLEXITY]
    security_score   = cat_totals[IssueCategory.SECURITY]

    # Maintainability: remap radon's MI (0–100) to 0–100, but cap at 80
    # so short bad code doesn't get a free 25-point boost from MI alone
    maint_score = min(80, int(maintainability_index * 1.0))

    big_o_score = _big_o_score(big_o_prediction)

    # Weighted overall — security still drives the grade.
    # With a prediction:    style 20%, complexity 20%, security 30%, maintainability 10%, big_o 20%
    # Without (unavailable): style 25%, complexity 25%, security 35%, maintainability 15% -- the
    # original weights, unchanged, so a missing/failed prediction never lowers a score that
    # would otherwise have been computed without it.
    if big_o_score is None:
        overall = int(
            style_score      * 0.25
            + complexity_score * 0.25
            + security_score   * 0.35
            + maint_score      * 0.15
        )
    else:
        overall = int(
            style_score      * 0.20
            + complexity_score * 0.20
            + security_score   * 0.30
            + maint_score      * 0.10
            + big_o_score       * 0.20
        )

    grade = _score_to_grade(overall)

    return QualityScore(
        score=overall,
        grade=grade,
        breakdown=QualityBreakdown(
            style=style_score,
            complexity=complexity_score,
            security=security_score,
            maintainability=maint_score,
            big_o=big_o_score,
        ),
    )


def _score_to_grade(score: int) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _build_summary(issues: List[Issue], score: QualityScore) -> str:
    if not issues:
        return f"No issues found. Quality grade: {score.grade} ({score.score}/100)."

    counts: dict[IssueSeverity, int] = {}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1

    parts = []
    if counts.get(IssueSeverity.ERROR):
        parts.append(f"{counts[IssueSeverity.ERROR]} error{'s' if counts[IssueSeverity.ERROR] > 1 else ''}")
    if counts.get(IssueSeverity.WARNING):
        parts.append(f"{counts[IssueSeverity.WARNING]} warning{'s' if counts[IssueSeverity.WARNING] > 1 else ''}")
    info_total = counts.get(IssueSeverity.INFO, 0) + counts.get(IssueSeverity.STYLE, 0)
    if info_total:
        parts.append(f"{info_total} suggestion{'s' if info_total > 1 else ''}")

    issues_str = ", ".join(parts)
    return f"{len(issues)} issue{'s' if len(issues) > 1 else ''} found ({issues_str}). Grade: {score.grade} ({score.score}/100)."

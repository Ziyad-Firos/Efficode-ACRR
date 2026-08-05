"""
review/complexity.py — Cyclomatic complexity analysis via radon.

Radon gives real, per-function cyclomatic complexity (McCabe).
NOT the same as Big-O — cyclomatic complexity counts the number
of independent paths through a function (branches + loops + 1).

Grade scale (standard Radon):
  A  1–5    low risk, simple
  B  6–10   moderate risk
  C  11–15  high risk, complex
  D  16–20  very high risk
  E/F 21+   untestable / unstable

Returns structured Issues for functions that exceed the threshold.
"""

from __future__ import annotations

import logging
from typing import List, Any

from app.models import Issue, IssueCategory, IssueSeverity

logger = logging.getLogger("acrr.review.complexity")

# Minimum complexity that triggers a warning — CC 5+ is already meaningful
_WARN_THRESHOLD = 5    # grade B+: warrants a note
_ERROR_THRESHOLD = 10  # grade C+: genuinely complex


def analyze_complexity(code: str) -> List[Issue]:
    """
    Analyze cyclomatic complexity of each function/method in code.

    Args:
        code: Python source string (already syntax-validated).

    Returns:
        List of Issue objects for functions exceeding the threshold.
        Empty list on tool failure.
    """
    try:
        from radon.complexity import cc_visit, ComplexityVisitor  # type: ignore
        from radon.metrics import mi_visit                          # type: ignore
    except ImportError:
        logger.warning("radon not installed — complexity check skipped. Install with: pip install radon")
        return []

    issues: List[Issue] = []

    try:
        blocks = cc_visit(code)
        for block in blocks:
            complexity: int = block.complexity
            if complexity > _ERROR_THRESHOLD:
                severity = IssueSeverity.ERROR
                label = "untestable"
                grade = "F"
            elif complexity > _WARN_THRESHOLD:
                severity = IssueSeverity.WARNING
                label = "too complex"
                grade = "D" if complexity <= 15 else "E"
            else:
                continue  # within acceptable range, no issue

            issues.append(Issue(
                line=block.lineno,
                column=None,
                severity=severity,
                category=IssueCategory.COMPLEXITY,
                rule="CC001",
                message=(
                    f"Function '{block.name}' has cyclomatic complexity {complexity} "
                    f"(grade {grade}) — {label}. "
                    f"Consider splitting it into smaller functions."
                ),
            ))

        logger.debug("complexity check: %d issues found", len(issues))

    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error in complexity check: %s", exc)

    return issues


def get_complexity_metrics(code: str) -> dict[str, Any]:
    """
    Return raw complexity metrics for all functions.
    Used by the ML predictor to build feature vectors.

    Returns:
        Dict mapping function_name → {complexity, grade, lineno}
    """
    try:
        from radon.complexity import cc_visit  # type: ignore
    except ImportError:
        return {}

    metrics: dict[str, Any] = {}
    try:
        for block in cc_visit(code):
            cc = block.complexity
            if cc <= 5:
                grade = "A"
            elif cc <= 10:
                grade = "B"
            elif cc <= 15:
                grade = "C"
            elif cc <= 20:
                grade = "D"
            elif cc <= 25:
                grade = "E"
            else:
                grade = "F"
            metrics[block.name] = {
                "complexity": cc,
                "grade": grade,
                "lineno": block.lineno,
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute complexity metrics: %s", exc)

    return metrics


def get_maintainability_index(code: str) -> float:
    """
    Return the Maintainability Index (0–100) for the whole module.
    Higher is better. Used for the quality score.
    Returns 50.0 as a neutral default on failure.
    """
    try:
        from radon.metrics import mi_visit  # type: ignore
        mi = mi_visit(code, multi=True)
        return round(float(mi), 1)
    except Exception:  # noqa: BLE001
        return 50.0

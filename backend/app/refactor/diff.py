"""
refactor/diff.py — Unified diff generation.

Generates a human-readable unified diff between original and
refactored code using Python's standard library difflib.
No extra dependencies — difflib is built-in.
"""

from __future__ import annotations

import difflib
import logging

logger = logging.getLogger("acrr.refactor.diff")


def make_diff(original: str, refactored: str, context_lines: int = 3) -> str:
    """
    Generate a unified diff between original and refactored code.

    Args:
        original:      The original source code string.
        refactored:    The refactored source code string.
        context_lines: Number of unchanged context lines around each change.

    Returns:
        A unified diff string. Empty string if the code is identical.
    """
    if original == refactored:
        return ""

    original_lines  = original.splitlines(keepends=True)
    refactored_lines = refactored.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        original_lines,
        refactored_lines,
        fromfile="original.py",
        tofile="refactored.py",
        n=context_lines,
    ))

    return "".join(diff_lines)


def make_html_diff(original: str, refactored: str) -> str:
    """
    Generate an HTML side-by-side diff (for rich frontend rendering).
    Returns an HTML string. Falls back to empty string on error.
    """
    try:
        differ = difflib.HtmlDiff(wrapcolumn=80)
        return differ.make_table(
            original.splitlines(),
            refactored.splitlines(),
            fromdesc="Original",
            todesc="Refactored",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HTML diff generation failed: %s", exc)
        return ""

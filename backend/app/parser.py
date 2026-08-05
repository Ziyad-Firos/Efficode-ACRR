"""
parser.py — Safe ast.parse wrapper.

This is the GATE for every pipeline. Code that fails to parse never
reaches review or refactor logic. All modules call this first.

Returns a ParseResult dataclass instead of raising exceptions,
so callers handle the failure path explicitly instead of catching
randomly-shaped errors.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("acrr.parser")


@dataclass
class ParseResult:
    valid: bool
    tree: Optional[ast.AST] = field(default=None, repr=False)
    error: Optional[str] = None
    error_line: Optional[int] = None
    error_col: Optional[int] = None


def parse_code(code: str) -> ParseResult:
    """
    Attempt to parse Python source code into an AST.

    Args:
        code: Raw Python source string.

    Returns:
        ParseResult with valid=True and the AST on success,
        or valid=False with a human-readable error message on failure.

    Never raises. All exceptions are caught and returned as ParseResult.
    """
    if not code or not code.strip():
        return ParseResult(valid=False, error="Empty code submitted.")

    try:
        tree = ast.parse(code)
        return ParseResult(valid=True, tree=tree)

    except SyntaxError as exc:
        # SyntaxError gives us line/col info — expose it
        msg = exc.msg or "Invalid syntax"
        logger.debug("Syntax error in submitted code: %s at line %s col %s",
                     msg, exc.lineno, exc.offset)
        return ParseResult(
            valid=False,
            error=f"{msg} (line {exc.lineno})",
            error_line=exc.lineno,
            error_col=exc.offset,
        )

    except ValueError as exc:
        # ast.parse raises ValueError for null bytes in source
        logger.debug("ValueError during parse: %s", exc)
        return ParseResult(valid=False, error=f"Invalid source: {exc}")

    except Exception as exc:  # noqa: BLE001
        # Belt-and-suspenders: should not happen in practice
        logger.warning("Unexpected parse error: %s", exc)
        return ParseResult(valid=False, error=f"Parse failed: {exc}")


def safe_unparse(tree: ast.AST) -> Optional[str]:
    """
    Convert an AST back to source code using ast.unparse (Python 3.9+).
    Returns None if unparsing fails.

    Used by the refactor engine after transformations.
    """
    try:
        return ast.unparse(tree)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ast.unparse failed: %s", exc)
        return None


def is_valid_python(code: str) -> bool:
    """Convenience helper — True if code parses without errors."""
    return parse_code(code).valid
